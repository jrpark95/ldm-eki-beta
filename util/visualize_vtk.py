#!/usr/bin/env python3
"""
VTK Particle Distribution Visualization and GIF Generator

This script reads VTK files from LDM-EKI simulation output and creates
geographic visualizations of particle distributions, including animated GIFs.

Usage:
    python3 util/visualize_vtk.py [options]

Examples:
    # Generate GIF from prior simulation
    python3 util/visualize_vtk.py --mode prior --start 1 --end 100 --step 5

    # Generate GIF from ensemble simulation
    python3 util/visualize_vtk.py --mode ensemble --start 1 --end 100 --step 5

    # Generate single plot
    python3 util/visualize_vtk.py --single output/plot_vtk_prior/plot_00050.vtk
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from scipy.ndimage import gaussian_filter
import io
from PIL import Image
import imageio.v2 as imageio
import os
import datetime
import re
import argparse
import sys

# Default simulation parameters
# Note: These should ideally be read from config files, but we use defaults here
DEFAULT_BASE_TIME = datetime.datetime(2011, 3, 14, 0, 0, 0)
DEFAULT_DT = 100  # seconds per timestep

# Default region extent for visualization (can be overridden)
# Format: [lon_min, lon_max, lat_min, lat_max]
DEFAULT_REGION_EXTENT = [136, 150, 32, 42]  # Japan region (Fukushima area)


def plot_particle_distribution(vtk_filename,
                               region_extent=None,
                               bins=(400, 400),
                               use_log_scale=True,
                               base_time=None,
                               dt=None,
                               sigma=2.0):
    """
    Generate a geographic plot of particle distribution from VTK file.

    Args:
        vtk_filename: Path to VTK file
        region_extent: [lon_min, lon_max, lat_min, lat_max] for plot bounds
        bins: (nx, ny) histogram bins for particle density
        use_log_scale: Use logarithmic color scale if True
        base_time: Base simulation time (datetime object)
        dt: Time step duration in seconds

    Returns:
        matplotlib Figure object or None if failed
    """
    try:
        mesh = pv.read(vtk_filename)
    except Exception as e:
        print(f"[Error] Failed to read {vtk_filename}: {e}")
        return None

    points = mesh.points

    if points is None or points.size == 0:
        print(f"[Skip] {vtk_filename}: No point data.")
        return None

    # Filter: valid longitude range, remove NaN/Inf
    valid_lon_mask = (points[:, 0] >= 0) & (points[:, 0] < 180.0)
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[valid_lon_mask & finite_mask]

    if points.size == 0:
        print(f"[Skip] {vtk_filename}: No valid points after filtering.")
        return None

    lons = points[:, 0]
    lats = points[:, 1]

    # Additional safety: check for NaN/Inf
    valid_coords = np.isfinite(lons) & np.isfinite(lats)
    lons = lons[valid_coords]
    lats = lats[valid_coords]

    if len(lons) == 0 or len(lats) == 0:
        print(f"[Skip] {vtk_filename}: No valid lat/lon values.")
        return None

    # Set visualization region based on actual particle positions
    if region_extent is None:
        # Use actual particle extent with a small margin
        lon_min, lon_max = np.min(lons), np.max(lons)
        lat_min, lat_max = np.min(lats), np.max(lats)

        # Add 5% margin on each side
        lon_margin = (lon_max - lon_min) * 0.05
        lat_margin = (lat_max - lat_min) * 0.05
        lon_min -= lon_margin
        lon_max += lon_margin
        lat_min -= lat_margin
        lat_max += lat_margin

        # Make the extent square (equal width and height)
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min

        if lon_range > lat_range:
            # Expand latitude to match longitude
            lat_center = (lat_min + lat_max) / 2
            lat_min = lat_center - lon_range / 2
            lat_max = lat_center + lon_range / 2
        else:
            # Expand longitude to match latitude
            lon_center = (lon_min + lon_max) / 2
            lon_min = lon_center - lat_range / 2
            lon_max = lon_center + lat_range / 2
    else:
        lon_min, lon_max, lat_min, lat_max = region_extent

    try:
        H, lon_edges, lat_edges = np.histogram2d(
            lons, lats, bins=bins, range=[[lon_min, lon_max], [lat_min, lat_max]]
        )
    except ValueError as e:
        print(f"[Skip] {vtk_filename}: histogram2d error - {e}")
        return None

    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
    Lon, Lat = np.meshgrid(lon_centers, lat_centers)

    # Apply Gaussian smoothing for smoother contours
    if sigma > 0:
        H = gaussian_filter(H, sigma=sigma)

    # Start plotting with square aspect ratio
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    # Set extent based on particle distribution (or user-specified region)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.STATES, edgecolor="black", facecolor="none")
    ax.add_feature(cfeature.LAKES, facecolor="lightblue")
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, color="blue")

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, color="gray")
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 10, "color": "black"}
    gl.ylabel_style = {"size": 10, "color": "black"}
    gl.right_labels = False
    gl.top_labels = False

    # Custom colormap (same as original)
    colors = [
        "#fffffc", "#c0e9fc", "#83c4f1", "#5099cf", "#49a181",
        "#6bbc51", "#69bd50", "#d3e158", "#feaf43", "#f96127",
        "#e1342a", "#9f2b2f", "#891a19"
    ]
    cmap = mcolors.ListedColormap(colors)

    # Handle case where H.max() is very small or zero
    if H.max() < 1e-10:
        print(f"[Warning] {vtk_filename}: All histogram values are near zero or empty.")
        boundaries = np.linspace(0, 1, len(colors) + 1)
    elif use_log_scale and H.max() > 1:
        # Use logarithmic scale, starting from 1 (10^0)
        boundaries = np.logspace(0, np.log10(H.max()), len(colors) + 1)
    else:
        # Use linear scale
        boundaries = np.linspace(0, H.max(), len(colors) + 1)

    # Ensure boundaries are strictly increasing
    if len(np.unique(boundaries)) < len(boundaries):
        print(f"[Warning] {vtk_filename}: Boundaries not unique, using linear scale.")
        boundaries = np.linspace(H.min(), H.max() + 1e-10, len(colors) + 1)

    norm = mcolors.BoundaryNorm(boundaries, ncolors=len(colors), clip=True)

    contour = ax.contourf(
        Lon, Lat, H.T,
        levels=boundaries,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label("Particle Count", fontsize=16)

    # Extract timestamp from filename
    # Supports patterns: plot_00001.vtk, Cs-137_00015.vtk, etc.
    match = re.search(r'_(\d{5})\.vtk$', vtk_filename)
    if match:
        time_step_idx = int(match.group(1))

        # Use provided base_time and dt, or defaults
        if base_time is None:
            base_time = DEFAULT_BASE_TIME
        if dt is None:
            dt = DEFAULT_DT

        time_seconds = time_step_idx * dt
        sim_time = base_time + datetime.timedelta(seconds=time_seconds)
        time_str = sim_time.strftime("%B %d, %Y, %H:%M UTC")
        particle_info = f" | Particles: {len(lons):,}"
    else:
        time_str = "Unknown Time"
        particle_info = f" | Particles: {len(lons):,}"

    ax.set_title(f"LDM-EKI Simulation - {time_str}{particle_info}", fontsize=18, weight='bold')
    plt.tight_layout()
    return fig


def create_gif_from_vtk_series(vtk_directory,
                               filename_pattern="plot_{:05d}.vtk",
                               start=1,
                               end=100,
                               step=1,
                               output_gif="particle_distribution.gif",
                               region_extent=None,
                               base_time=None,
                               dt=None,
                               **plot_kwargs):
    """
    Create animated GIF from series of VTK files.

    Args:
        vtk_directory: Directory containing VTK files
        filename_pattern: Pattern for VTK filenames (e.g., "plot_{:05d}.vtk")
        start: Starting timestep index
        end: Ending timestep index
        step: Step size for timesteps
        output_gif: Output GIF filename
        region_extent: Geographic extent for plots
        base_time: Base simulation time
        dt: Time step duration in seconds
        **plot_kwargs: Additional arguments passed to plot_particle_distribution
    """
    images = []

    print(f"\n{'='*70}")
    print(f"Generating GIF from VTK files:")
    print(f"  Directory: {vtk_directory}")
    print(f"  Range: {start} to {end} (step {step})")
    print(f"  Output: {output_gif}")
    print(f"{'='*70}\n")

    # If region_extent is not specified, compute maximum extent from all VTK files
    if region_extent is None:
        print("[Info] Computing maximum extent from all VTK files...")
        lon_min_global, lon_max_global = float('inf'), float('-inf')
        lat_min_global, lat_max_global = float('inf'), float('-inf')

        for t in range(start, end + 1, step):
            vtk_filename = os.path.join(vtk_directory, filename_pattern.format(t))
            if not os.path.exists(vtk_filename):
                continue

            try:
                mesh = pv.read(vtk_filename)
                points = mesh.points
                if points is None or points.size == 0:
                    continue

                # Filter valid points
                valid_lon_mask = (points[:, 0] >= 0) & (points[:, 0] < 180.0)
                finite_mask = np.isfinite(points).all(axis=1)
                points = points[valid_lon_mask & finite_mask]

                if points.size == 0:
                    continue

                lons = points[:, 0]
                lats = points[:, 1]
                valid_coords = np.isfinite(lons) & np.isfinite(lats)
                lons = lons[valid_coords]
                lats = lats[valid_coords]

                if len(lons) > 0 and len(lats) > 0:
                    lon_min_global = min(lon_min_global, np.min(lons))
                    lon_max_global = max(lon_max_global, np.max(lons))
                    lat_min_global = min(lat_min_global, np.min(lats))
                    lat_max_global = max(lat_max_global, np.max(lats))

            except Exception as e:
                continue

        if lon_min_global != float('inf'):
            # Add 5% margin
            lon_margin = (lon_max_global - lon_min_global) * 0.05
            lat_margin = (lat_max_global - lat_min_global) * 0.05
            lon_min_global -= lon_margin
            lon_max_global += lon_margin
            lat_min_global -= lat_margin
            lat_max_global += lat_margin

            # Make the extent square (equal width and height)
            lon_range = lon_max_global - lon_min_global
            lat_range = lat_max_global - lat_min_global

            if lon_range > lat_range:
                # Expand latitude to match longitude
                lat_center = (lat_min_global + lat_max_global) / 2
                lat_min_global = lat_center - lon_range / 2
                lat_max_global = lat_center + lon_range / 2
            else:
                # Expand longitude to match latitude
                lon_center = (lon_min_global + lon_max_global) / 2
                lon_min_global = lon_center - lat_range / 2
                lon_max_global = lon_center + lat_range / 2

            region_extent = [lon_min_global, lon_max_global, lat_min_global, lat_max_global]
            print(f"[Info] Global extent (square): lon=[{lon_min_global:.2f}, {lon_max_global:.2f}], "
                  f"lat=[{lat_min_global:.2f}, {lat_max_global:.2f}]")
            print(f"[Info] Extent size: {lon_max_global - lon_min_global:.2f}° × {lat_max_global - lat_min_global:.2f}°")
        else:
            print("[Warning] Could not determine extent from VTK files, using default.")
            region_extent = DEFAULT_REGION_EXTENT

    print(f"[Info] Using fixed extent for all frames: {region_extent}\n")

    for t in range(start, end + 1, step):
        vtk_filename = os.path.join(vtk_directory, filename_pattern.format(t))

        if not os.path.exists(vtk_filename):
            print(f"[Skip] File not found: {vtk_filename}")
            continue

        print(f"Processing [{t:05d}]: {os.path.basename(vtk_filename)}")

        fig = plot_particle_distribution(
            vtk_filename,
            region_extent=region_extent,
            base_time=base_time,
            dt=dt,
            **plot_kwargs
        )

        if fig is None:
            continue

        # Convert figure to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
        plt.close(fig)

    if images:
        print(f"\n[Success] Saving GIF with {len(images)} frames...")
        images[0].save(
            output_gif,
            save_all=True,
            append_images=images[1:],
            duration=300,  # 300ms per frame
            loop=0
        )
        print(f"[Success] GIF saved as {output_gif}")
        print(f"          File size: {os.path.getsize(output_gif) / 1024 / 1024:.2f} MB")
    else:
        print("[Warning] No valid frames were generated. GIF not created.")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate geographic visualizations from VTK files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate GIF from prior simulation (timesteps 1-100, every 5 steps)
  python3 util/visualize_vtk.py --mode prior --start 1 --end 100 --step 5

  # Generate GIF from ensemble simulation
  python3 util/visualize_vtk.py --mode ensemble --start 1 --end 50 --step 2

  # Generate single plot from specific VTK file
  python3 util/visualize_vtk.py --single output/plot_vtk_prior/plot_00050.vtk

  # Custom region extent (lon_min, lon_max, lat_min, lat_max)
  python3 util/visualize_vtk.py --mode prior --start 1 --end 100 --step 5 \\
      --extent 135 145 35 40

  # Custom output filename
  python3 util/visualize_vtk.py --mode prior --start 1 --end 100 --step 5 \\
      --output my_simulation.gif
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        '--mode',
        choices=['prior', 'ensemble'],
        help='Simulation mode: prior (true simulation) or ensemble (default: auto-detect)'
    )
    mode_group.add_argument(
        '--single',
        metavar='VTK_FILE',
        help='Generate single plot from specific VTK file'
    )

    # Time range arguments
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        help='Starting timestep index (default: 1)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=100,
        help='Ending timestep index (default: 100)'
    )
    parser.add_argument(
        '--step',
        type=int,
        default=5,
        help='Step size for timesteps (default: 5)'
    )

    # Output options
    parser.add_argument(
        '--output',
        metavar='FILENAME',
        help='Output GIF filename (default: auto-generated based on mode)'
    )
    parser.add_argument(
        '--output-dir',
        default='output/results',
        help='Output directory for GIF (default: output/results)'
    )

    # Visualization options
    parser.add_argument(
        '--extent',
        nargs=4,
        type=float,
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        help=f'Geographic extent (default: {DEFAULT_REGION_EXTENT})'
    )
    parser.add_argument(
        '--bins',
        nargs=2,
        type=int,
        default=[400, 400],
        metavar=('NX', 'NY'),
        help='Histogram bins for particle density (default: 400 400)'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=2.0,
        help='Gaussian smoothing sigma for smoother contours (default: 2.0, use 0 to disable)'
    )
    parser.add_argument(
        '--linear-scale',
        action='store_true',
        help='Use linear color scale instead of logarithmic'
    )

    # Time parameters
    parser.add_argument(
        '--base-time',
        metavar='YYYY-MM-DD-HH:MM:SS',
        help=f'Base simulation time (default: {DEFAULT_BASE_TIME.strftime("%Y-%m-%d %H:%M:%S")})'
    )
    parser.add_argument(
        '--dt',
        type=int,
        default=DEFAULT_DT,
        help=f'Time step duration in seconds (default: {DEFAULT_DT})'
    )

    args = parser.parse_args()

    # Parse base_time if provided
    base_time = DEFAULT_BASE_TIME
    if args.base_time:
        try:
            base_time = datetime.datetime.strptime(args.base_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"[Error] Invalid base time format: {args.base_time}")
            print("        Expected format: YYYY-MM-DD HH:MM:SS")
            sys.exit(1)

    # Set region extent (None means auto-detect from particles)
    region_extent = args.extent if args.extent else None

    # Auto-detect mode if not specified
    if not args.mode and not args.single:
        print("[Info] No mode specified, auto-detecting available VTK files...\n")

        # Check which directories exist and have VTK files
        prior_dir = 'output/plot_vtk_prior'
        ensemble_dir = 'output/plot_vtk_ens'

        prior_exists = os.path.exists(prior_dir) and len([f for f in os.listdir(prior_dir) if f.endswith('.vtk')]) > 0
        ensemble_exists = os.path.exists(ensemble_dir) and len([f for f in os.listdir(ensemble_dir) if f.endswith('.vtk')]) > 0

        if prior_exists:
            args.mode = 'prior'
            print(f"[Info] Found VTK files in {prior_dir}, using mode: prior")
        elif ensemble_exists:
            args.mode = 'ensemble'
            print(f"[Info] Found VTK files in {ensemble_dir}, using mode: ensemble")
        else:
            print("[Error] No VTK files found in output/plot_vtk_prior or output/plot_vtk_ens")
            print("        Run a simulation first or specify --single with a VTK file path.")
            sys.exit(1)

    # Handle single plot mode
    if args.single:
        if not os.path.exists(args.single):
            print(f"[Error] File not found: {args.single}")
            sys.exit(1)

        print(f"Generating single plot from: {args.single}")
        fig = plot_particle_distribution(
            args.single,
            region_extent=region_extent,
            bins=tuple(args.bins),
            use_log_scale=not args.linear_scale,
            base_time=base_time,
            dt=args.dt,
            sigma=args.sigma
        )

        if fig is not None:
            output_png = args.single.replace('.vtk', '_plot.png')
            fig.savefig(output_png, dpi=150, bbox_inches='tight')
            print(f"[Success] Plot saved as {output_png}")
            plt.show()
        else:
            print("[Error] Failed to generate plot")
            sys.exit(1)
        return

    # Handle GIF mode
    if args.mode == 'prior':
        vtk_directory = 'output/plot_vtk_prior'
        default_output = 'particle_distribution_prior.gif'
    else:  # ensemble
        vtk_directory = 'output/plot_vtk_ens'
        default_output = 'particle_distribution_ensemble.gif'

    if not os.path.exists(vtk_directory):
        print(f"[Error] Directory not found: {vtk_directory}")
        print("        Run a simulation first to generate VTK files.")
        sys.exit(1)

    # Auto-detect end timestep if default value is used
    if args.end == 100:  # default value
        # Find the maximum timestep from existing VTK files
        vtk_files = [f for f in os.listdir(vtk_directory) if f.endswith('.vtk')]
        if vtk_files:
            # Extract timestep numbers from filenames (e.g., plot_00050.vtk -> 50)
            timesteps = []
            for f in vtk_files:
                match = re.search(r'_(\d{5})\.vtk$', f)
                if match:
                    timesteps.append(int(match.group(1)))

            if timesteps:
                detected_end = max(timesteps)
                print(f"[Info] Auto-detected end timestep: {detected_end} (found {len(timesteps)} VTK files)")
                args.end = detected_end

    # Prepare output path
    output_gif = args.output if args.output else default_output
    if not output_gif.endswith('.gif'):
        output_gif += '.gif'

    os.makedirs(args.output_dir, exist_ok=True)
    output_gif = os.path.join(args.output_dir, output_gif)

    # Generate GIF
    create_gif_from_vtk_series(
        vtk_directory=vtk_directory,
        filename_pattern="plot_{:05d}.vtk",
        start=args.start,
        end=args.end,
        step=args.step,
        output_gif=output_gif,
        region_extent=region_extent,
        bins=tuple(args.bins),
        use_log_scale=not args.linear_scale,
        base_time=base_time,
        dt=args.dt,
        sigma=args.sigma
    )


if __name__ == "__main__":
    main()
