#!/usr/bin/env python3
"""
Diagnose EKI convergence issue - check if observations are being updated
"""
import numpy as np
import os

# Change to project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

print("="*70)
print("DIAGNOSING EKI CONVERGENCE ISSUE")
print("="*70)

# Load iteration files
iteration_dir = 'logs/eki_iterations'
if not os.path.exists(iteration_dir):
    print(f"❌ {iteration_dir} not found")
    exit(1)

files = sorted([f for f in os.listdir(iteration_dir) if f.startswith('iteration_') and f.endswith('.npy')])

if len(files) == 0:
    print(f"❌ No iteration files found")
    exit(1)

print(f"\nFound {len(files)} iteration files")

# Load first few iterations
iterations_to_check = min(5, len(files))
print(f"Checking first {iterations_to_check} and last iteration...\n")

data_list = []
for i in range(iterations_to_check):
    data = np.load(os.path.join(iteration_dir, files[i]))
    data_list.append(data)
    mean_state = data.mean(axis=1)
    print(f"Iteration {i+1}:")
    print(f"  Shape: {data.shape}")
    print(f"  Mean emission: {mean_state.mean():.3e}")
    print(f"  Min: {mean_state.min():.3e}, Max: {mean_state.max():.3e}")

# Load last iteration
if len(files) > iterations_to_check:
    last_data = np.load(os.path.join(iteration_dir, files[-1]))
    mean_last = last_data.mean(axis=1)
    print(f"\nIteration {len(files)} (last):")
    print(f"  Shape: {last_data.shape}")
    print(f"  Mean emission: {mean_last.mean():.3e}")
    print(f"  Min: {mean_last.min():.3e}, Max: {mean_last.max():.3e}")

# Check convergence pattern
print(f"\n" + "="*70)
print("CONVERGENCE PATTERN ANALYSIS")
print("="*70)

# Compare first and last iterations
first_mean = data_list[0].mean(axis=1)
if len(files) > 1:
    if len(files) > iterations_to_check:
        last_mean = last_data.mean(axis=1)
    else:
        last_mean = data_list[-1].mean(axis=1)

    diff = np.abs(last_mean - first_mean)
    relative_change = diff / (first_mean + 1e-10)

    print(f"\nChange from Iteration 1 to {len(files)}:")
    print(f"  Mean absolute change: {diff.mean():.3e}")
    print(f"  Mean relative change: {relative_change.mean()*100:.2f}%")
    print(f"  Max relative change: {relative_change.max()*100:.2f}%")

    # Check if it's actually changing
    if relative_change.mean() < 0.01:  # Less than 1% average change
        print(f"\n⚠️  WARNING: Very small changes detected (<1%)")
        print(f"   This suggests the algorithm is converging too quickly or")
        print(f"   observations are not being properly updated.")
    else:
        print(f"\n✅ Significant changes detected, algorithm is actively updating")

# Check iteration-to-iteration changes
print(f"\n" + "="*70)
print("ITERATION-TO-ITERATION CHANGE ANALYSIS")
print("="*70)

for i in range(1, min(iterations_to_check, len(data_list))):
    prev_mean = data_list[i-1].mean(axis=1)
    curr_mean = data_list[i].mean(axis=1)
    diff = np.abs(curr_mean - prev_mean)
    relative = diff / (prev_mean + 1e-10)

    print(f"\nIteration {i} → {i+1}:")
    print(f"  Mean change: {diff.mean():.3e} ({relative.mean()*100:.2f}%)")
    print(f"  Max change: {diff.max():.3e} ({relative.max()*100:.2f}%)")

    if relative.mean() < 0.001:  # Less than 0.1%
        print(f"  ⚠️  Very small change - possible premature convergence")

# Load true emissions for comparison
print(f"\n" + "="*70)
print("COMPARISON WITH TRUE EMISSIONS")
print("="*70)

try:
    with open('input/eki_settings.txt', 'r') as f:
        true_emissions = []
        in_section = False
        for line in f:
            if 'TRUE_EMISSION_SERIES=' in line:
                in_section = True
                continue
            if in_section:
                line = line.strip()
                if not line or '=' in line:
                    if true_emissions:
                        break
                else:
                    try:
                        true_emissions.append(float(line))
                    except ValueError:
                        pass

    if true_emissions:
        true_emissions = np.array(true_emissions)
        print(f"\nTrue emissions loaded: {len(true_emissions)} timesteps")
        print(f"  Mean: {true_emissions.mean():.3e}")
        print(f"  Min: {true_emissions.min():.3e}, Max: {true_emissions.max():.3e}")

        # Compare with final estimate
        if len(files) > iterations_to_check:
            final_mean = last_mean
        else:
            final_mean = data_list[-1].mean(axis=1)

        if len(final_mean) == len(true_emissions):
            error = np.abs(final_mean - true_emissions)
            relative_error = error / (true_emissions + 1e-10)

            print(f"\nFinal estimate error:")
            print(f"  Mean absolute error: {error.mean():.3e}")
            print(f"  Mean relative error: {relative_error.mean()*100:.2f}%")
            print(f"  Max relative error: {relative_error.max()*100:.2f}%")
        else:
            print(f"  ⚠️  Shape mismatch: estimate has {len(final_mean)} values, true has {len(true_emissions)}")
except Exception as e:
    print(f"  Could not load true emissions: {e}")

print(f"\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
