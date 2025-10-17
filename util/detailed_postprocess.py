#!/usr/bin/env python3
"""
Detailed Post-Processing Utility for LDM-EKI

This script generates comprehensive analysis outputs including:
1. Extracted debug data from eki_debug_data.npz (text only)
2. Individual plots from all_receptors_comparison.png
3. Clean input configuration summary

Usage:
    python3 util/detailed_postprocess.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
from datetime import datetime
import re
import struct

# Add src/eki to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'eki'))

from eki_debug_logger import load_debug_data


def ensure_output_dir(subdir='detailed'):
    """Create output directory structure"""
    base_dir = os.path.join('output', 'results', subdir)
    subdirs = ['debug_data', 'plots', 'config']

    for sd in subdirs:
        path = os.path.join(base_dir, sd)
        os.makedirs(path, exist_ok=True)

    return base_dir


def extract_debug_data(output_dir):
    """Extract and save debug data from npz archive (text only)"""
    print("\n" + "="*70)
    print("EXTRACTING DEBUG DATA")
    print("="*70)

    data = load_debug_data()

    if not data:
        print("⚠️  No debug data found in logs/debug/eki_debug_data.npz")
        return None

    debug_dir = os.path.join(output_dir, 'debug_data')

    for key, arr in data.items():
        print(f"\n📦 Processing: {key}")
        print(f"   Shape: {arr.shape}, dtype: {arr.dtype}")

        # Save as text only (first 100 values for large arrays)
        txt_path = os.path.join(debug_dir, f'{key}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Array: {key}\n")
            f.write(f"Shape: {arr.shape}\n")
            f.write(f"dtype: {arr.dtype}\n")
            f.write(f"Min: {arr.min():.6e}\n")
            f.write(f"Max: {arr.max():.6e}\n")
            f.write(f"Mean: {arr.mean():.6e}\n")
            f.write(f"Std: {arr.std():.6e}\n")
            f.write(f"\nData (first 100 values):\n")
            flat = arr.flatten()
            for i, val in enumerate(flat[:100]):
                f.write(f"{i:6d}: {val:.12e}\n")
            if len(flat) > 100:
                f.write(f"\n... ({len(flat) - 100} more values)\n")
        print(f"   ✓ Saved: {txt_path}")

    return data


def parse_config_value(line):
    """Parse configuration value from line"""
    if ':' in line:
        return line.split(':', 1)[1].strip()
    elif '=' in line:
        return line.split('=', 1)[1].strip()
    return line.strip()


def create_config_summary(output_dir):
    """Create clean configuration summary"""
    print("\n" + "="*70)
    print("GENERATING CONFIGURATION SUMMARY")
    print("="*70)

    config_dir = os.path.join(output_dir, 'config')
    summary_path = os.path.join(config_dir, 'input_summary.md')

    with open(summary_path, 'w') as f:
        f.write(f"# LDM-EKI Input Configuration Summary\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # EKI Settings
        f.write("## EKI Configuration\n\n")
        try:
            with open('input/eki.conf', 'r') as cfg:
                for line in cfg:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'EKI_TIME_INTERVAL' in line:
                        f.write(f"- **Time Interval:** {parse_config_value(line)} minutes\n")
                    elif 'ENSEMBLE_SIZE' in line:
                        f.write(f"- **Ensemble Size:** {parse_config_value(line)}\n")
                    elif 'ITERATION' in line and 'MAX' in line:
                        f.write(f"- **Max Iterations:** {parse_config_value(line)}\n")
                    elif 'ADAPTIVE' in line:
                        f.write(f"- **Adaptive:** {parse_config_value(line)}\n")
                    elif 'LOCALIZED' in line:
                        f.write(f"- **Localized:** {parse_config_value(line)}\n")
        except:
            f.write("*EKI configuration not found*\n")

        # Receptor Settings
        f.write("\n## Receptor Configuration\n\n")
        try:
            with open('input/receptor.conf', 'r') as cfg:
                for line in cfg:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'NUM_RECEPTORS' in line:
                        f.write(f"- **Number of Receptors:** {parse_config_value(line)}\n")
                        break
            # Read receptor locations
            f.write("- **Receptor Locations:**\n")
            with open('input/receptor.conf', 'r') as cfg:
                in_locations = False
                for line in cfg:
                    line = line.strip()
                    if 'RECEPTOR_LOCATIONS' in line:
                        in_locations = True
                        continue
                    if in_locations and line and not line.startswith('#'):
                        if '=' in line:
                            break
                        parts = line.split()
                        if len(parts) >= 2:
                            f.write(f"  - Lat: {parts[0]}, Lon: {parts[1]}\n")
        except:
            f.write("*Receptor configuration not found*\n")

        # Source Settings
        f.write("\n## Source Configuration\n\n")
        try:
            with open('input/source.conf', 'r') as cfg:
                for line in cfg:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'SOURCE_LOCATION' in line:
                        parts = parse_config_value(line).split()
                        if len(parts) >= 3:
                            f.write(f"- **Source Location:** Lat={parts[0]}, Lon={parts[1]}, Alt={parts[2]}m\n")
        except:
            f.write("*Source configuration not found*\n")

        # Nuclide Settings
        f.write("\n## Nuclide Configuration\n\n")
        try:
            with open('input/nuclides.conf', 'r') as cfg:
                for line in cfg:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'PRIMARY_NUCLIDE' in line:
                        f.write(f"- **Primary Nuclide:** {parse_config_value(line)}\n")
                    elif 'DECAY_CONSTANT' in line:
                        f.write(f"- **Decay Constant:** {parse_config_value(line)} s⁻¹\n")
        except:
            f.write("*Nuclide configuration not found*\n")

        # Simulation Settings
        f.write("\n## Simulation Settings\n\n")
        try:
            with open('input/setting.txt', 'r') as cfg:
                for line in cfg:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if 'Time_end' in line:
                        f.write(f"- **Simulation Duration:** {parse_config_value(line)} seconds\n")
                    elif 'dt' in line and 'Time' not in line:
                        f.write(f"- **Time Step:** {parse_config_value(line)} seconds\n")
                    elif 'Number_of_particles' in line:
                        f.write(f"- **Particles:** {parse_config_value(line)}\n")
        except:
            f.write("*Simulation settings not found*\n")

    print(f"✓ Configuration summary: {summary_path}")
    return summary_path


def generate_individual_plots(output_dir):
    """Generate individual plots from original data (same as compare_all_receptors.py)"""
    print("\n" + "="*70)
    print("GENERATING INDIVIDUAL PLOTS FROM DATA")
    print("="*70)

    # Import functions from compare_all_receptors
    sys.path.insert(0, 'util')
    from compare_all_receptors import (
        load_eki_settings,
        parse_single_particle_counts,
        parse_ensemble_particle_counts,
        load_ensemble_doses_from_shm,
        load_eki_iterations,
        load_true_emissions,
        plot_emission_estimates
    )

    plots_dir = os.path.join(output_dir, 'plots')

    # Load settings
    eki_settings = load_eki_settings()
    num_receptors = eki_settings['num_receptors']
    num_timesteps = eki_settings['num_timesteps']
    time_interval = eki_settings['time_interval']

    print(f"Configuration: {num_receptors} receptors, {num_timesteps} timesteps, {time_interval}min interval")

    # Load data
    single_data = parse_single_particle_counts(num_receptors=num_receptors)
    ensemble_particle_data = parse_ensemble_particle_counts(num_receptors=num_receptors)
    ensemble_doses = load_ensemble_doses_from_shm()
    eki_iterations = load_eki_iterations()
    true_emissions = load_true_emissions()

    # Prepare arrays
    times = np.arange(1, num_timesteps + 1) * time_interval
    single_counts = [np.zeros(num_timesteps) for _ in range(num_receptors)]
    single_doses = [np.zeros(num_timesteps) for _ in range(num_receptors)]

    # Fill single mode data
    if single_data:
        for i, obs in enumerate(single_data[:num_timesteps]):
            for r in range(num_receptors):
                receptor_key = f'R{r+1}'
                if f'{receptor_key}_count' in obs:
                    single_counts[r][i] = obs[f'{receptor_key}_count']
                if f'{receptor_key}_dose' in obs:
                    single_doses[r][i] = obs[f'{receptor_key}_dose']

    # Process ensemble dose data
    if ensemble_doses is not None:
        ens_dose_mean = ensemble_doses.mean(axis=0)
        ens_dose_std = ensemble_doses.std(axis=0)
    else:
        ens_dose_mean = np.zeros((num_receptors, num_timesteps))
        ens_dose_std = np.zeros((num_receptors, num_timesteps))

    # Process ensemble particle data
    ens_counts_mean = [np.zeros(num_timesteps) for _ in range(num_receptors)]
    ens_counts_std = [np.zeros(num_timesteps) for _ in range(num_receptors)]

    for obs_idx, data in ensemble_particle_data.items():
        if obs_idx < num_timesteps:
            for r in range(num_receptors):
                receptor_key = f'R{r+1}'
                if f'{receptor_key}_count' in data and data[f'{receptor_key}_count']:
                    ens_counts_mean[r][obs_idx] = np.mean(data[f'{receptor_key}_count'])
                    ens_counts_std[r][obs_idx] = np.std(data[f'{receptor_key}_count'])

    # Colors
    single_color = '#2E86C1'
    ensemble_color = '#E74C3C'

    # Receptor titles
    receptor_titles = []
    for i in range(num_receptors):
        if i < len(eki_settings['receptor_locations']):
            lat, lon = eki_settings['receptor_locations'][i]
            receptor_titles.append(f'R{i+1} ({lat:.1f}, {lon:.1f})')
        else:
            receptor_titles.append(f'R{i+1}')

    plot_count = 0

    # Generate individual particle plots
    for r in range(num_receptors):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(times, single_counts[r], 'o-', color=single_color,
                linewidth=2, markersize=5, label='Single Mode', alpha=0.9)
        ax.plot(times, ens_counts_mean[r], 's--', color=ensemble_color,
                linewidth=2, markersize=4, label='Ensemble Mean', alpha=0.9)
        ax.fill_between(times,
                        ens_counts_mean[r] - ens_counts_std[r],
                        ens_counts_mean[r] + ens_counts_std[r],
                        color=ensemble_color, alpha=0.2)
        ax.set_xlabel('Time (minutes)', fontsize=11)
        ax.set_ylabel('Particle Count', fontsize=11)
        ax.set_title(f'{receptor_titles[r]} PARTICLES', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plot_path = os.path.join(plots_dir, f'R{r+1}_particles.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ R{r+1} particles: {plot_path}")
        plot_count += 1

    # Generate individual dose plots
    for r in range(num_receptors):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(times, single_doses[r], 'o-', color=single_color,
                linewidth=2, markersize=5, label='Single Mode', alpha=0.9)
        ax.plot(times, ens_dose_mean[r], 's--', color=ensemble_color,
                linewidth=2, markersize=4, label='Ensemble Mean', alpha=0.9)
        ax.fill_between(times,
                        ens_dose_mean[r] - ens_dose_std[r],
                        ens_dose_mean[r] + ens_dose_std[r],
                        color=ensemble_color, alpha=0.2)
        ax.set_xlabel('Time (minutes)', fontsize=11)
        ax.set_ylabel('Dose (Sv)', fontsize=11)
        ax.set_yscale('log')
        ax.set_title(f'{receptor_titles[r]} DOSE (log scale)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plot_path = os.path.join(plots_dir, f'R{r+1}_dose.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ R{r+1} dose: {plot_path}")
        plot_count += 1

    # Generate emission estimates plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_emission_estimates(ax, eki_iterations, true_emissions, num_timesteps, time_interval)

    plot_path = os.path.join(plots_dir, 'emission_estimates.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Emission estimates: {plot_path}")
    plot_count += 1

    print(f"\n✓ Total plots generated: {plot_count}")


def create_summary_report(output_dir, debug_data):
    """Create final summary report"""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)

    report_path = os.path.join(output_dir, 'README.md')

    with open(report_path, 'w') as f:
        f.write("# LDM-EKI Detailed Analysis Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Contents\n\n")
        f.write("This directory contains detailed analysis results from LDM-EKI simulation:\n\n")
        f.write("- **`config/`** - Input configuration summary\n")
        f.write("- **`debug_data/`** - Extracted debug data (text format)\n")
        f.write("- **`plots/`** - Individual receptor and emission plots\n\n")

        f.write("---\n\n")

        f.write("## Debug Data\n\n")
        if debug_data:
            f.write(f"Total arrays: {len(debug_data)}\n\n")
            f.write("| Array Name | Shape | Size (MB) |\n")
            f.write("|------------|-------|----------|\n")
            for key, arr in debug_data.items():
                size_mb = arr.nbytes / (1024 * 1024)
                f.write(f"| `{key}` | {arr.shape} | {size_mb:.2f} |\n")
            f.write("\nAll arrays saved as text files in `debug_data/` directory.\n")
        else:
            f.write("No debug data available.\n")

        f.write("\n---\n\n")

        f.write("## Plots\n\n")
        f.write("Individual plots extracted from all_receptors_comparison.png:\n\n")
        f.write("- `R1_particles.png`, `R2_particles.png`, `R3_particles.png` - Particle counts\n")
        f.write("- `R1_dose.png`, `R2_dose.png`, `R3_dose.png` - Dose observations\n")
        f.write("- `emission_estimates.png` - Emission rate estimates\n\n")

        f.write("---\n\n")
        f.write("## How to Load Debug Data\n\n")
        f.write("```python\n")
        f.write("import numpy as np\n\n")
        f.write("# Load from original NPZ archive\n")
        f.write("data = dict(np.load('logs/debug/eki_debug_data.npz'))\n")
        f.write("prior_state = data['prior_state']\n")
        f.write("initial_obs = data['initial_observation']\n")
        f.write("```\n\n")

    print(f"✓ Summary report: {report_path}")
    return report_path


def main():
    """Main post-processing function"""
    print("\n" + "="*70)
    print("LDM-EKI DETAILED POST-PROCESSING")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # Create output directory
    output_dir = ensure_output_dir('detailed')
    print(f"Output directory: {output_dir}")

    # Step 1: Extract debug data
    debug_data = extract_debug_data(output_dir)

    # Step 2: Create config summary
    create_config_summary(output_dir)

    # Step 3: Generate individual plots from data
    generate_individual_plots(output_dir)

    # Step 4: Create summary report
    summary_path = create_summary_report(output_dir, debug_data)

    # Final message
    print("\n" + "="*70)
    print("POST-PROCESSING COMPLETE!")
    print("="*70)
    print(f"\n📂 All results saved to: {output_dir}")
    print(f"📄 Summary report: {summary_path}")
    print("\nGenerated files:")
    print(f"  • Debug data: {output_dir}/debug_data/ (text files)")
    print(f"  • Individual plots: {output_dir}/plots/")
    print(f"  • Configuration: {output_dir}/config/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
