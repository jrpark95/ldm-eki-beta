"""
Debug Logging Utilities for LDM-EKI

This module provides debug logging functions to save intermediate
data arrays during EKI optimization. All data is saved to a single
compressed binary file: logs/debug/eki_debug_data.npz
"""

import numpy as np
import os

# Debug logging configuration
DEBUG_LOGGING = True  # Always enabled

# Determine project root (assuming this file is in src/eki/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEBUG_DIR = os.path.join(PROJECT_ROOT, 'logs', 'debug')
DEBUG_FILE = os.path.join(DEBUG_DIR, 'eki_debug_data.npz')

# In-memory storage for all debug data
_debug_data = {}


def ensure_debug_dir():
    """Create debug directory if it doesn't exist."""
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR, exist_ok=True)


def _save_to_disk():
    """Save all accumulated debug data to a single npz file."""
    ensure_debug_dir()
    np.savez_compressed(DEBUG_FILE, **_debug_data)


def save_prior_state(state_init):
    """
    Save prior state data to debug archive.

    Args:
        state_init (numpy.ndarray): Prior state vector
    """
    _debug_data['prior_state'] = state_init
    _save_to_disk()


def save_initial_observation(obs):
    """
    Save initial observation data to debug archive.

    Args:
        obs (numpy.ndarray): Initial observation vector
    """
    _debug_data['initial_observation'] = obs
    _save_to_disk()


def save_prior_ensemble(state):
    """
    Save prior ensemble data to debug archive.

    Args:
        state (numpy.ndarray): Prior ensemble array (num_states x num_ensemble)
    """
    _debug_data['prior_ensemble'] = state
    _save_to_disk()


def save_ensemble_states_sent(iteration, tmp_states):
    """
    Save ensemble states being sent to C++ via shared memory.

    Args:
        iteration (int): Current iteration number
        tmp_states (numpy.ndarray): Ensemble states (num_timesteps x num_ensemble)
    """
    key = f'iter{iteration:03d}_states_sent'
    _debug_data[key] = tmp_states
    _save_to_disk()


def save_ensemble_observations_received(iteration, tmp_results):
    """
    Save ensemble observations received from C++ via shared memory.

    Args:
        iteration (int): Current iteration number
        tmp_results (numpy.ndarray): Ensemble observations (num_ensemble x num_timesteps x num_receptors)
    """
    key = f'iter{iteration:03d}_obs_received'
    _debug_data[key] = tmp_results
    _save_to_disk()


def load_debug_data():
    """
    Load all debug data from the npz archive.

    Returns:
        dict: Dictionary containing all saved arrays

    Example:
        data = load_debug_data()
        prior_state = data['prior_state']
        iter1_states = data['iter001_states_sent']
    """
    if os.path.exists(DEBUG_FILE):
        return dict(np.load(DEBUG_FILE))
    return {}


def clear_debug_data():
    """Clear all debug data from memory and disk."""
    global _debug_data
    _debug_data = {}
    if os.path.exists(DEBUG_FILE):
        os.remove(DEBUG_FILE)
