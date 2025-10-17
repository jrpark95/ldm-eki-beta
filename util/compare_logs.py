#!/usr/bin/env python3
"""
Comparison Script: Development Code vs Reference Code
Identifies divergence points through step-by-step data comparison
"""

import numpy as np
import os
import sys

def compare_arrays(dev_data, ref_data, name):
    """Compare two numpy arrays and print detailed statistics"""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")

    # Shape comparison
    print(f"Dev shape: {dev_data.shape}")
    print(f"Ref shape: {ref_data.shape}")

    if dev_data.shape != ref_data.shape:
        print(f"⚠️  SHAPE MISMATCH!")
        return False

    # Statistics
    print(f"\nDev stats: min={dev_data.min():.6e}, max={dev_data.max():.6e}, mean={dev_data.mean():.6e}")
    print(f"Ref stats: min={ref_data.min():.6e}, max={ref_data.max():.6e}, mean={ref_data.mean():.6e}")

    # Difference
    diff = np.abs(dev_data - ref_data)
    max_diff = diff.max()
    mean_diff = diff.mean()

    # Relative difference (avoid division by zero)
    ref_nonzero = ref_data != 0
    if ref_nonzero.any():
        rel_diff = np.abs((dev_data[ref_nonzero] - ref_data[ref_nonzero]) / ref_data[ref_nonzero])
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()
    else:
        max_rel_diff = 0
        mean_rel_diff = 0

    print(f"\nAbsolute difference:")
    print(f"  Max: {max_diff:.6e}")
    print(f"  Mean: {mean_diff:.6e}")

    print(f"\nRelative difference (non-zero elements):")
    print(f"  Max: {max_rel_diff*100:.2f}%")
    print(f"  Mean: {mean_rel_diff*100:.2f}%")

    # Matching criteria
    if mean_rel_diff < 0.01:  # < 1% mean difference
        print(f"✅ MATCH (mean rel diff < 1%)")
        return True
    elif mean_rel_diff < 0.05:  # < 5% mean difference
        print(f"⚠️  CLOSE (mean rel diff < 5%)")
        return True
    else:
        print(f"❌ MISMATCH (mean rel diff >= 5%)")
        return False

def compare_iteration_data(iteration):
    """Compare data for a specific iteration"""
    dev_states_file = f"../logs/dev_iter{iteration:03d}_ensemble_states_sent.npy"
    ref_states_file = f"/home/jrpark/ekitest2/ldm-eki-ref-eki/logs/ref_iter{iteration:03d}_ensemble_states_sent.npy"

    dev_obs_file = f"../logs/dev_iter{iteration:03d}_ensemble_observations_received.npy"
    ref_obs_file = f"/home/jrpark/ekitest2/ldm-eki-ref-eki/logs/ref_iter{iteration:03d}_ensemble_observations_received.npy"

    # Check if files exist
    if not os.path.exists(dev_states_file):
        print(f"Dev states file not found: {dev_states_file}")
        return False
    if not os.path.exists(ref_states_file):
        print(f"Ref states file not found: {ref_states_file}")
        return False

    # Load and compare ensemble states sent
    dev_states = np.load(dev_states_file)
    ref_states = np.load(ref_states_file)
    match_states = compare_arrays(dev_states, ref_states, f"Iteration {iteration} - Ensemble States Sent")

    # Check if observation files exist
    if not os.path.exists(dev_obs_file):
        print(f"\nDev observations file not found: {dev_obs_file}")
        return match_states
    if not os.path.exists(ref_obs_file):
        print(f"\nRef observations file not found: {ref_obs_file}")
        return match_states

    # Load and compare ensemble observations received
    dev_obs = np.load(dev_obs_file)
    ref_obs = np.load(ref_obs_file)
    match_obs = compare_arrays(dev_obs, ref_obs, f"Iteration {iteration} - Ensemble Observations Received")

    return match_states and match_obs

def main():
    print("=" * 80)
    print("LDM-EKI Development Code vs Reference Code Comparison")
    print("=" * 80)

    # Compare prior state
    if os.path.exists("../logs/dev_prior_state.npy") and os.path.exists("../logs/ref_prior_state.npy"):
        dev_prior_state = np.load("../logs/dev_prior_state.npy")
        # Reference prior state is in the old location
        if os.path.exists("/home/jrpark/ekitest2/logs/reference/prior_state.npy"):
            ref_prior_state = np.load("/home/jrpark/ekitest2/logs/reference/prior_state.npy")
            compare_arrays(dev_prior_state, ref_prior_state, "Prior State")
    else:
        print("Prior state files not found, skipping...")

    # Compare prior ensemble
    dev_prior_ens_path = "../logs/dev_prior_ensemble.npy"
    ref_prior_ens_path = "/home/jrpark/ekitest2/ldm-eki-ref-eki/logs/ref_prior_ensemble.npy"
    if os.path.exists(dev_prior_ens_path) and os.path.exists(ref_prior_ens_path):
        dev_prior_ensemble = np.load(dev_prior_ens_path)
        ref_prior_ensemble = np.load(ref_prior_ens_path)
        compare_arrays(dev_prior_ensemble, ref_prior_ensemble, "Prior Ensemble")
    else:
        print(f"\nPrior ensemble files not found:")
        print(f"  Dev: {dev_prior_ens_path} exists={os.path.exists(dev_prior_ens_path)}")
        print(f"  Ref: {ref_prior_ens_path} exists={os.path.exists(ref_prior_ens_path)}")

    # Compare initial observation
    dev_init_obs_path = "../logs/dev_initial_observation.npy"
    ref_init_obs_path = "/home/jrpark/ekitest2/ldm-eki-ref-eki/logs/ref_initial_observation.npy"
    if os.path.exists(dev_init_obs_path) and os.path.exists(ref_init_obs_path):
        dev_init_obs = np.load(dev_init_obs_path)
        ref_init_obs = np.load(ref_init_obs_path)
        compare_arrays(dev_init_obs, ref_init_obs, "Initial Observation")
    else:
        print(f"\nInitial observation files not found:")
        print(f"  Dev: {dev_init_obs_path} exists={os.path.exists(dev_init_obs_path)}")
        print(f"  Ref: {ref_init_obs_path} exists={os.path.exists(ref_init_obs_path)}")

    # Compare iteration data
    print("\n" + "=" * 80)
    print("Iteration-by-Iteration Comparison")
    print("=" * 80)

    for iteration in range(1, 14):  # 13 iterations
        print(f"\n{'#'*80}")
        print(f"# Iteration {iteration}")
        print(f"{'#'*80}")

        if not compare_iteration_data(iteration):
            print(f"\n⚠️  First mismatch detected at iteration {iteration}")
            break

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)

if __name__ == "__main__":
    # Get the script directory and change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    main()
