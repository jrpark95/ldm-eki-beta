#!/usr/bin/env python3
"""
Automated file splitter for ldm_func.cuh
Splits based on function boundaries into simulation/particle/output modules
"""

import os
import re
from pathlib import Path

# Function boundaries identified from grep analysis
FUNCTION_SPLITS = {
    'simulation': {
        'functions': [
            'runSimulation',
            'runSimulation_eki',
            'runSimulation_eki_dump'
        ],
        'descriptions': [
            'Main simulation loop',
            'EKI simulation loop',
            'EKI simulation with dump'
        ],
        'line_ranges': [(112, 261), (263, 552), (1260, 1549)]
    },
    'particle': {
        'functions': [
            'allocateGPUMemory',
            'checkParticleNaN',
            'checkMeteoDataNaN'
        ],
        'descriptions': [
            'GPU memory allocation for particles',
            'Debug particle NaN checking',
            'Debug meteorological data NaN checking'
        ],
        'line_ranges': [(54, 110), (554, 591), (593, 660)]
    },
    'output': {
        'functions': [
            'startTimer',
            'stopTimer',
            'initializeEKIObservationSystem',
            'computeReceptorObservations',
            'saveEKIObservationResults',
            'writeEKIObservationsToSharedMemory',
            'computeReceptorObservations_AllEnsembles',
            'cleanupEKIObservationSystem',
            'resetEKIObservationSystemForNewIteration',
            'computeGridReceptorObservations',
            'saveGridReceptorData',
            'cleanupGridReceptorSystem'
        ],
        'descriptions': [
            'Start performance timer',
            'Stop performance timer',
            'Initialize EKI observation system',
            'Compute single-mode receptor observations',
            'Save EKI observation results to file',
            'Write observations to shared memory',
            'Compute ensemble receptor observations',
            'Cleanup EKI observation system',
            'Reset observation system for new iteration',
            'Compute grid receptor observations',
            'Save grid receptor data',
            'Cleanup grid receptor system'
        ],
        'line_ranges': [
            (42, 45), (47, 52),
            (664, 750), (752, 844), (846, 900),
            (902, 907), (909, 1021), (1023, 1056),
            (1058, 1088), (1092, 1167), (1169, 1225),
            (1227, 1258)
        ]
    }
}

def read_file_lines(filepath):
    """Read file and return lines with 1-based indexing"""
    with open(filepath, 'r') as f:
        return [''] + f.readlines()  # Index 0 is dummy for 1-based indexing

def extract_function_block(lines, start_line, end_line):
    """Extract function block including complete braces"""
    return ''.join(lines[start_line:end_line+1])

def generate_function_declaration(func_name):
    """Generate function declaration based on function name patterns"""
    # Timer functions
    if func_name in ['startTimer', 'stopTimer']:
        return f"void LDM::{func_name}();\n"

    # GPU/Memory functions
    if 'allocate' in func_name.lower():
        return f"void LDM::{func_name}();\n"

    # Check/Debug functions
    if 'check' in func_name.lower():
        if 'Particle' in func_name:
            return f"void LDM::{func_name}(const std::string& location, int max_check = 10);\n"
        else:
            return f"void LDM::{func_name}(const std::string& location);\n"

    # Simulation functions
    if 'runSimulation' in func_name:
        return f"void LDM::{func_name}();\n"

    # Observation system functions
    if 'initialize' in func_name.lower() or 'cleanup' in func_name.lower() or 'reset' in func_name.lower():
        return f"void LDM::{func_name}();\n"

    if 'compute' in func_name.lower():
        if 'AllEnsembles' in func_name:
            return f"void LDM::{func_name}(int timestep, float currentTime, int num_ensembles, int num_timesteps);\n"
        else:
            return f"void LDM::{func_name}(int timestep, float currentTime);\n"

    if 'save' in func_name.lower():
        return f"void LDM::{func_name}();\n"

    if 'write' in func_name.lower() and 'SharedMemory' in func_name:
        return f"bool LDM::{func_name}(void* writer_ptr);\n"

    # Default
    return f"void LDM::{func_name}();\n"

def create_header_file(module_name, functions, descriptions):
    """Create header file with function declarations"""

    module_descriptions = {
        'simulation': 'Main simulation loop and execution control',
        'particle': 'Particle management and GPU memory operations',
        'output': 'Output handling, observation system, and logging'
    }

    header = f"""#pragma once

/**
 * @file ldm_func_{module_name}.cuh
 * @brief {module_descriptions[module_name]}
 *
 * This file contains {module_name}-related functions for the LDM simulation.
 * Part of the LDM-EKI simulation framework.
 */

#include "ldm_config.cuh"
#include "ldm_struct.cuh"
#include <string>
#include <vector>
#include <chrono>

// Forward declarations
class LDM;

"""

    # Add function declarations with documentation
    for func_name, desc in zip(functions, descriptions):
        header += f"/**\n * @brief {desc}\n */\n"
        header += generate_function_declaration(func_name)
        header += "\n"

    return header

def create_implementation_file(module_name, content):
    """Create implementation file with function bodies"""
    impl = f"""/**
 * @file ldm_func_{module_name}.cu
 * @brief {module_name.capitalize()} module implementation
 */

#include "ldm.cuh"
#include "ldm_func_{module_name}.cuh"
#include "colors.h"

"""
    impl += content
    return impl

def split_func_file(input_path, output_dir):
    """Main splitting logic"""
    print(f"Reading source file: {input_path}")
    lines = read_file_lines(input_path)
    total_lines = len(lines) - 1
    print(f"Total lines: {total_lines}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each module
    for module_name, info in FUNCTION_SPLITS.items():
        print(f"\n{'='*60}")
        print(f"Processing module: {module_name}")
        print(f"{'='*60}")

        # Create header file
        header_path = os.path.join(output_dir, f"ldm_func_{module_name}.cuh")
        header_content = create_header_file(module_name, info['functions'], info['descriptions'])

        with open(header_path, 'w') as f:
            f.write(header_content)
        print(f"✓ Created header: {header_path}")

        # Extract implementation content
        impl_content = ""
        for func_name, (start, end), desc in zip(info['functions'], info['line_ranges'], info['descriptions']):
            print(f"  Extracting {func_name} (lines {start}-{end})")
            impl_content += extract_function_block(lines, start, end)
            impl_content += "\n"

        # Create implementation file
        impl_path = os.path.join(output_dir, f"ldm_func_{module_name}.cu")
        impl_file_content = create_implementation_file(module_name, impl_content)

        with open(impl_path, 'w') as f:
            f.write(impl_file_content)
        print(f"✓ Created implementation: {impl_path}")
        print(f"  Lines: {impl_content.count(chr(10))}")

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "src/include/ldm_func.cuh"
    output_dir = project_root / "src/simulation"

    print("LDM Function File Splitter")
    print("="*60)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1

    split_func_file(str(input_file), str(output_dir))

    print("\n" + "="*60)
    print("✓ File splitting completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - src/simulation/ldm_func_simulation.cuh/cu")
    print("  - src/simulation/ldm_func_particle.cuh/cu")
    print("  - src/simulation/ldm_func_output.cuh/cu")

    return 0

if __name__ == "__main__":
    exit(main())
