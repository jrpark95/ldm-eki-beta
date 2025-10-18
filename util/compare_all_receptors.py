#!/usr/bin/env python3
"""
REAL comparison with all receptors shown separately
Dynamically generates multiple pages (3 receptors per page)
1. Particle counts for receptors
2. Observation dose for receptors
3. Emission estimates from EKI iterations (shown on all pages)

Dependencies:
    - numpy: Array operations
    - matplotlib: Plotting and visualization
    - struct: Binary data reading
    - re: Regular expression parsing
    - os, sys: File system operations

Input files required:
    - input/receptor.conf: Receptor configuration
    - input/eki.conf: EKI settings (timestep interval)
    - /dev/shm/ldm_eki_ensemble_observations_*: Observation data (binary)
    - logs/eki_iterations/*.pkl: Iteration results

Output:
    - output/results/all_receptors_comparison.png
"""
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import re
import os

def load_receptor_settings(receptor_file='input/receptor.conf'):
    """Load receptor configuration from receptor.conf"""
    receptor_settings = {
        'num_receptors': 3,
        'receptor_locations': []
    }

    try:
        with open(receptor_file, 'r') as f:
            lines = f.readlines()
            in_receptor_locations = False

            for line in lines:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Normalize separator: convert ':' to '=' for uniform parsing
                if ':' in line and '=' not in line:
                    line = line.replace(':', '=', 1)

                # Parse number of receptors
                if 'NUM_RECEPTORS' in line:
                    receptor_settings['num_receptors'] = int(line.split('=')[1].strip())

                # Parse receptor locations
                elif 'RECEPTOR_LOCATIONS' in line:
                    in_receptor_locations = True
                    continue

                if in_receptor_locations:
                    # Stop if we hit another section
                    if '=' in line and not line.startswith('#'):
                        in_receptor_locations = False
                    # Parse location line
                    elif not line.startswith('#'):
                        try:
                            parts = line.split()
                            if len(parts) >= 2:
                                lat, lon = float(parts[0]), float(parts[1])
                                receptor_settings['receptor_locations'].append((lat, lon))
                        except:
                            pass

    except FileNotFoundError:
        print(f"Warning: Could not load receptor settings from {receptor_file}: File not found")
        print("Using default values")
        return None
    except Exception as e:
        print(f"Warning: Could not load receptor settings from {receptor_file}: {e}")
        print("Using default values")
        return None

    return receptor_settings


def load_eki_settings(settings_file='input/eki.conf'):
    """Load EKI settings from configuration file"""
    settings = {
        'time_interval': 15.0,
        'num_receptors': 3,
        'receptor_locations': [],
        'num_timesteps': None  # Will be calculated from TRUE_EMISSION_SERIES
    }

    # Load receptor settings from separate file
    receptor_settings = load_receptor_settings()
    if receptor_settings:
        settings['num_receptors'] = receptor_settings['num_receptors']
        settings['receptor_locations'] = receptor_settings['receptor_locations']

    try:
        with open(settings_file, 'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Normalize separator: convert ':' to '=' for uniform parsing
                if ':' in line and '=' not in line:
                    line = line.replace(':', '=', 1)

                # Parse time interval
                if 'EKI_TIME_INTERVAL' in line:
                    settings['time_interval'] = float(line.split('=')[1].strip())

        # Calculate number of timesteps from TRUE_EMISSION_SERIES length
        if settings['num_timesteps'] is None:
            # Count TRUE_EMISSION_SERIES entries
            true_emissions = load_true_emissions()
            if true_emissions is not None and len(true_emissions) > 0:
                settings['num_timesteps'] = len(true_emissions)
            else:
                # Fallback: try to read from setting.txt
                try:
                    with open('input/setting.txt', 'r') as f:
                        for line in f:
                            if 'Time_end' in line:
                                time_end_seconds = float(line.split(':')[1].strip())
                                settings['num_timesteps'] = int((time_end_seconds / 60) / settings['time_interval'])
                                break
                except:
                    settings['num_timesteps'] = 24  # Safe default

        print(f"[CONFIG] Loaded EKI settings from eki.conf and receptor.conf:")
        print(f"  - Time interval: {settings['time_interval']} minutes")
        print(f"  - Num timesteps: {settings['num_timesteps']}")
        print(f"  - Num receptors: {settings['num_receptors']} (from receptor.conf)")
        print(f"  - Receptor locations: {len(settings['receptor_locations'])} found (from receptor.conf)")

    except Exception as e:
        print(f"Warning: Could not load settings from {settings_file}: {e}")
        print("Using default values")

    return settings

def parse_single_particle_counts(log_file='logs/ldm_eki_simulation.log', num_receptors=3):
    """Parse single mode particle counts - dynamic receptor support (default 3 for backward compatibility)"""
    observations = []

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Build dynamic regex pattern for all receptors
                # Pattern: [EKI_OBS] Observation X at t=Ys: R1=dose(count) R2=...
                if '[EKI_OBS]' in line and 'Observation' in line:
                    obs_entry = {'time_min': 0}

                    # Extract timestamp
                    time_match = re.search(r'at t=(\d+)s', line)
                    if time_match:
                        obs_entry['time_min'] = int(time_match.group(1)) / 60.0

                    # Extract all receptor data
                    for r in range(1, num_receptors + 1):
                        pattern = rf'R{r}=([\d.e+-]+)\((\d+)p\)'
                        match = re.search(pattern, line)
                        if match:
                            obs_entry[f'R{r}_dose'] = float(match.group(1))
                            obs_entry[f'R{r}_count'] = int(match.group(2))

                    if len(obs_entry) > 1:  # Has data beyond time
                        observations.append(obs_entry)

        print(f"Loaded {len(observations)} single mode observations with particle counts")
    except Exception as e:
        print(f"Error loading single particle counts: {e}")

    return observations

def parse_ensemble_particle_counts(log_file='logs/ldm_eki_simulation.log', num_receptors=3):
    """Parse ensemble particle counts from log - dynamic receptor support (default 3 for backward compatibility)"""
    ensemble_data = {}

    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Pattern: [EKI_ENSEMBLE_OBS] EnsX obsY: R1=... R2=...
                if '[EKI_ENSEMBLE_OBS]' in line:
                    # Try new format first: [EKI_ENSEMBLE_OBS] obsY at t=Xs: R1=countp R2=countp...
                    new_format_match = re.search(r'obs(\d+) at t=(\d+)s:', line)
                    if new_format_match:
                        obs_idx = int(new_format_match.group(1)) - 1  # Convert to 0-based

                        if obs_idx not in ensemble_data:
                            ensemble_data[obs_idx] = {}
                            for r in range(1, num_receptors + 1):
                                ensemble_data[obs_idx][f'R{r}_count'] = []

                        # Extract all receptor particle counts (already averaged in C++)
                        for r in range(1, num_receptors + 1):
                            pattern = rf'R{r}=(\d+)p'
                            match = re.search(pattern, line)
                            if match:
                                # Store as single-element array for compatibility with mean/std calculation
                                ensemble_data[obs_idx][f'R{r}_count'] = [int(match.group(1))]

                    else:
                        # Fallback to old format: [EKI_ENSEMBLE_OBS] EnsX obsY: R1=dose(count) R2=...
                        ens_match = re.search(r'Ens(\d+) obs(\d+):', line)
                        if ens_match:
                            ens_id = int(ens_match.group(1))
                            obs_idx = int(ens_match.group(2)) - 1  # Convert to 0-based

                            if obs_idx not in ensemble_data:
                                ensemble_data[obs_idx] = {}
                                for r in range(1, num_receptors + 1):
                                    ensemble_data[obs_idx][f'R{r}_dose'] = []
                                    ensemble_data[obs_idx][f'R{r}_count'] = []

                            # Extract all receptor data
                            for r in range(1, num_receptors + 1):
                                pattern = rf'R{r}=([\d.e+-]+)\((\d+)p\)'
                                match = re.search(pattern, line)
                                if match:
                                    ensemble_data[obs_idx][f'R{r}_dose'].append(float(match.group(1)))
                                    ensemble_data[obs_idx][f'R{r}_count'].append(int(match.group(2)))

        print(f"Loaded ensemble data for {len(ensemble_data)} observation points")
    except Exception as e:
        print(f"Error loading ensemble particle counts: {e}")

    return ensemble_data

def load_ensemble_doses_from_shm():
    """Load ensemble dose data from shared memory"""
    try:
        with open('/dev/shm/ldm_eki_ensemble_obs_config', 'rb') as f:
            config_data = f.read(12)
            num_ensembles, num_receptors, num_timesteps = struct.unpack('<iii', config_data)

        total_elements = num_ensembles * num_receptors * num_timesteps
        with open('/dev/shm/ldm_eki_ensemble_obs_data', 'rb') as f:
            flat_data = struct.unpack(f'<{total_elements}f', f.read(total_elements * 4))

        # After dimension fix, C++ sends data as [ensemble, timestep, receptor]
        observations = np.array(flat_data).reshape((num_ensembles, num_timesteps, num_receptors))

        # But visualization expects [ensemble, receptor, timestep], so transpose
        observations = np.transpose(observations, (0, 2, 1))  # Swap timestep and receptor dimensions

        print(f"Loaded ensemble doses: {num_ensembles} ensembles x {num_receptors} receptors x {num_timesteps} timesteps")
        print(f"  (transposed from timestep-major to receptor-major for visualization)")
        return observations
    except Exception as e:
        print(f"Error loading ensemble doses from shm: {e}")
        return None

def load_eki_iterations():
    """Load EKI iteration data for emission estimates"""
    iterations = []
    iteration_dir = 'logs/eki_iterations'

    if os.path.exists(iteration_dir):
        files = sorted([f for f in os.listdir(iteration_dir) if f.startswith('iteration_') and f.endswith('.npy')])
        for file in files:
            try:
                data = np.load(os.path.join(iteration_dir, file))
                iterations.append(data)
            except:
                pass
        print(f"Loaded {len(iterations)} EKI iterations")

    return iterations

def load_true_emissions():
    """Load true emission profile from eki.conf"""
    emissions = []
    try:
        with open('input/eki.conf', 'r') as f:
            in_section = False
            for line in f:
                # Support both ':' and '=' separators
                if 'TRUE_EMISSION_SERIES' in line and (':' in line or '=' in line):
                    in_section = True
                    continue
                if in_section:
                    line = line.strip()
                    # Stop at next section (key=value or key: value)
                    if not line or '=' in line or ':' in line:
                        if emissions:
                            break
                    else:
                        try:
                            emissions.append(float(line))
                        except ValueError:
                            pass
    except:
        pass

    return np.array(emissions) if emissions else None

def plot_emission_estimates(ax, eki_iterations, true_emissions, num_timesteps, time_interval):
    """Plot emission estimates on given axis"""

    # Plot true emissions
    if true_emissions is not None and len(true_emissions) > 0:
        total_duration_minutes = num_timesteps * time_interval
        emission_times = np.linspace(0, total_duration_minutes, len(true_emissions))
        ax.plot(emission_times, true_emissions, 'k-', linewidth=3,
                label='True Emissions', alpha=0.9)

    # Plot EKI iteration estimates if available
    if eki_iterations:
        num_iters = len(eki_iterations)

        # Determine how many iterations to show (smart strategy)
        if num_iters == 1:
            iters_to_show = [0]
            show_labels = ['Iteration 1']
        elif num_iters <= 5:
            iters_to_show = list(range(num_iters))
            show_labels = [f'Iteration {i+1}' for i in range(num_iters)]
        else:
            iters_to_show = [0] + list(range(num_iters-3, num_iters))
            show_labels = ['Iteration 1'] + [f'Iteration {i+1}' for i in range(num_iters-3, num_iters)]

        # Generate colors dynamically
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(iters_to_show)))

        for color_idx, iter_idx in enumerate(iters_to_show):
            iteration = eki_iterations[iter_idx]
            if len(iteration) > 0 and iteration.ndim == 2:
                mean_est = iteration.mean(axis=1)
                std_est = iteration.std(axis=1)

                total_duration_minutes = num_timesteps * time_interval
                iter_times = np.linspace(0, total_duration_minutes, len(mean_est))

                linestyle = '-' if iter_idx == num_iters - 1 else '--'
                linewidth = 3 if iter_idx == num_iters - 1 else 2
                alpha = 0.9 if iter_idx == num_iters - 1 else 0.6

                ax.plot(iter_times, mean_est, linestyle, color=colors[color_idx],
                        linewidth=linewidth, label=show_labels[color_idx], alpha=alpha)

                # Show uncertainty only for last iteration
                if iter_idx == num_iters - 1:
                    ax.fill_between(iter_times,
                                   np.maximum(mean_est - std_est, 0),  # Use 0 as lower bound for linear scale
                                   mean_est + std_est,
                                   color=colors[color_idx], alpha=0.15)
    else:
        ax.text(0.5, 0.5, 'No EKI iteration data available',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='red')

    ax.axhline(y=1.5e8, color='blue', linestyle=':', linewidth=2,
               alpha=0.5, label='Prior Estimate')

    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Emission Rate (Bq)', fontsize=12)
    ax.set_title('EMISSION ESTIMATES (Linear Scale)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

def create_receptor_comparison():
    """Create comparison showing all receptors separately - generates multiple pages for > 3 receptors"""

    print("\n" + "="*70)
    print("ALL RECEPTORS COMPARISON - REAL DATA")
    print("="*70)

    # Load EKI settings
    eki_settings = load_eki_settings()
    num_receptors = eki_settings['num_receptors']
    num_timesteps = eki_settings['num_timesteps']
    time_interval = eki_settings['time_interval']

    num_pages = (num_receptors + 2) // 3  # Ceiling division
    print(f"\n📊 Generating plots for {num_receptors} receptors (from receptor.conf)")
    print(f"   Will create {num_pages} page(s) (3 receptors per page)")

    # Load single mode particle data - use dynamic receptor count from config
    single_data = parse_single_particle_counts(num_receptors=num_receptors)
    print(f"[DEBUG] Single data loaded: {len(single_data)} observations")

    # Load ensemble particle data - use dynamic receptor count from config
    ensemble_particle_data = parse_ensemble_particle_counts(num_receptors=num_receptors)

    # Load ensemble dose data
    ensemble_doses = load_ensemble_doses_from_shm()

    # Load EKI iterations
    eki_iterations = load_eki_iterations()

    # Load true emissions
    true_emissions = load_true_emissions()

    # Prepare data arrays
    times = np.arange(1, num_timesteps + 1) * time_interval

    # Initialize arrays dynamically based on actual receptor count
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

    # Process ensemble dose data FIRST (needed for plots)
    if ensemble_doses is not None:
        # Handle case where actual receptors in data < configured receptors
        actual_receptors_in_data = ensemble_doses.shape[1]
        if actual_receptors_in_data < num_receptors:
            print(f"⚠️  Warning: Config has {num_receptors} receptors but data only has {actual_receptors_in_data}")
            print(f"   Using {actual_receptors_in_data} receptors for plotting")
            num_receptors = actual_receptors_in_data
            # Recalculate number of pages
            num_pages = (num_receptors + 2) // 3

        ens_dose_mean = ensemble_doses.mean(axis=0)  # [num_receptors, num_timesteps]
        ens_dose_std = ensemble_doses.std(axis=0)
    else:
        ens_dose_mean = np.zeros((num_receptors, num_timesteps))
        ens_dose_std = np.zeros((num_receptors, num_timesteps))

    # Process ensemble particle data
    print(f"Ensemble particle data points from log: {len(ensemble_particle_data)}")

    # Initialize arrays with zeros
    ens_counts_mean = [np.zeros(num_timesteps) for _ in range(num_receptors)]
    ens_counts_std = [np.zeros(num_timesteps) for _ in range(num_receptors)]

    # Override with actual logged data where available
    logged_obs = []
    for obs_idx, data in ensemble_particle_data.items():
        if obs_idx < num_timesteps:
            logged_obs.append(obs_idx)
            for r in range(num_receptors):
                receptor_key = f'R{r+1}'
                if f'{receptor_key}_count' in data and data[f'{receptor_key}_count']:
                    ens_counts_mean[r][obs_idx] = np.mean(data[f'{receptor_key}_count'])
                    ens_counts_std[r][obs_idx] = np.std(data[f'{receptor_key}_count'])

    print(f"Logged observation indices: {sorted(logged_obs)}")
    print(f"Total logged observations: {len(logged_obs)} out of {num_timesteps}")

    # Colors
    single_color = '#2E86C1'
    ensemble_color = '#E74C3C'

    # Prepare receptor titles
    receptor_titles = []
    for i in range(num_receptors):
        if i < len(eki_settings['receptor_locations']):
            lat, lon = eki_settings['receptor_locations'][i]
            receptor_titles.append(f'R{i+1} ({lat:.1f}, {lon:.1f})')
        else:
            receptor_titles.append(f'R{i+1}')

    # ========== GENERATE MULTIPLE PAGES ==========
    output_paths = []

    for page_idx in range(num_pages):
        # Determine which receptors to show on this page
        start_receptor = page_idx * 3
        end_receptor = min(start_receptor + 3, num_receptors)
        receptors_on_page = list(range(start_receptor, end_receptor))
        num_cols = len(receptors_on_page)

        print(f"\n🖼️  Generating page {page_idx + 1}/{num_pages}: R{start_receptor+1}-R{end_receptor}")

        # Create figure with 3 rows (particles, doses, emissions)
        fig = plt.figure(figsize=(6*num_cols, 14))
        gs = GridSpec(3, num_cols, figure=fig, hspace=0.3, wspace=0.3)

        # ========== ROW 1: PARTICLE COUNTS ==========
        for col, r in enumerate(receptors_on_page):
            ax = fig.add_subplot(gs[0, col])
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

        # ========== ROW 2: OBSERVATION DOSES ==========
        for col, r in enumerate(receptors_on_page):
            ax = fig.add_subplot(gs[1, col])
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

        # ========== ROW 3: EMISSION ESTIMATES (full width) ==========
        ax_emission = fig.add_subplot(gs[2, :])
        plot_emission_estimates(ax_emission, eki_iterations, true_emissions, num_timesteps, time_interval)

        # Main title
        page_suffix = f" (Page {page_idx + 1}/{num_pages})" if num_pages > 1 else ""
        fig.suptitle(f'ALL RECEPTORS COMPARISON - REAL DATA{page_suffix}\nParticle Counts | Doses | Emission Estimates',
                    fontsize=16, fontweight='bold', y=0.995)

        # Save figure - NEW STRUCTURE
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if num_pages == 1:
            output_path = 'output/results/all_receptors_comparison.png'
        else:
            output_path = f'output/results/all_receptors_comparison_page{page_idx+1}.png'

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        output_paths.append(output_path)
        print(f"   ✅ Saved: {output_path}")

        plt.close(fig)

    # Print summary
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)

    # Single mode particle totals
    single_totals_str = ", ".join([f"R{i+1}={single_counts[i].sum():.0f}" for i in range(num_receptors)])
    print(f"Single Mode Particle Totals: {single_totals_str}")

    # Ensemble mean particle totals
    ensemble_totals_str = ", ".join([f"R{i+1}={ens_counts_mean[i].sum():.0f}" for i in range(num_receptors)])
    print(f"Ensemble Mean Particle Totals: {ensemble_totals_str}")

    # Single mode peak doses
    single_peaks_str = ", ".join([f"R{i+1}={single_doses[i].max():.2e}" for i in range(num_receptors)])
    print(f"Single Mode Peak Doses: {single_peaks_str} Sv")

    # Ensemble mean peak doses
    ensemble_peaks_str = ", ".join([f"R{i+1}={ens_dose_mean[i].max():.2e}" for i in range(num_receptors)])
    print(f"Ensemble Mean Peak Doses: {ensemble_peaks_str} Sv")

    if eki_iterations:
        print(f"EKI Iterations Available: {len(eki_iterations)}")

    print(f"\n📄 Total pages generated: {num_pages}")
    for path in output_paths:
        print(f"   • {path}")
    print("="*70)

if __name__ == "__main__":
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    create_receptor_comparison()
