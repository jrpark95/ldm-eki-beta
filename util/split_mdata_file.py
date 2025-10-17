#!/usr/bin/env python3
"""
Automated file splitter for ldm_mdata.cuh
Splits based on function boundaries into loading/processing/cache modules
"""

import os
import re
from pathlib import Path

# Function boundaries identified from analysis
FUNCTION_SPLITS = {
    'loading': {
        'functions': [
            'initializeFlexGFSData',
            'loadFlexGFSData',
            'loadFlexHeightData'
        ],
        'line_ranges': [(4, 1089), (1091, 1376), (1378, 1435)]
    },
    'processing': {
        'functions': [],  # DRHO calculations are embedded
        'line_ranges': []  # Will extract DRHO calculation logic
    },
    'cache': {
        'functions': [
            'calculateRequiredMeteoFiles',
            'loadSingleMeteoFile',
            'preloadAllEKIMeteorologicalData',
            'cleanupEKIMeteorologicalData'
        ],
        'line_ranges': [(1441, 1454), (1456, 1721), (1723, 1973), (1975, 1979)]
    }
}

def read_file_lines(filepath):
    """Read file and return lines with 1-based indexing"""
    with open(filepath, 'r') as f:
        return [''] + f.readlines()  # Index 0 is dummy for 1-based indexing

def extract_function_block(lines, start_line, end_line):
    """Extract function block including complete braces"""
    return ''.join(lines[start_line:end_line+1])

def create_header_file(module_name, functions):
    """Create header file with function declarations"""
    header = f"""#pragma once

/**
 * @file ldm_mdata_{module_name}.cuh
 * @brief Meteorological data {module_name} module
 *
 * This file contains {module_name}-related functions for meteorological data handling.
 * Part of the LDM-EKI meteorological data management system.
 */

#include "ldm_config.cuh"
#include "ldm_struct.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

// Forward declarations
class LDM;
struct FlexPres;
struct FlexUnis;

"""

    # Add function declarations
    for func_name in functions:
        if 'initialize' in func_name.lower() or 'load' in func_name.lower():
            header += f"void LDM::{func_name}();\n"
        elif 'calculate' in func_name.lower():
            header += f"int LDM::{func_name}();\n"
        elif func_name == 'loadSingleMeteoFile':
            header += f"bool LDM::{func_name}(int file_index, FlexPres*& pres_data, FlexUnis*& unis_data, std::vector<float>& hgt_data);\n"
        elif func_name == 'preloadAllEKIMeteorologicalData':
            header += f"bool LDM::{func_name}();\n"
        elif 'cleanup' in func_name.lower():
            header += f"void LDM::{func_name}();\n"

    return header

def create_implementation_file(module_name, content):
    """Create implementation file with function bodies"""
    impl = f"""/**
 * @file ldm_mdata_{module_name}.cu
 * @brief Meteorological data {module_name} implementation
 */

#include "ldm.cuh"
#include "ldm_mdata_{module_name}.cuh"
#include "colors.h"

"""
    impl += content
    return impl

def split_mdata_file(input_path, output_dir):
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
        header_path = os.path.join(output_dir, f"ldm_mdata_{module_name}.cuh")
        header_content = create_header_file(module_name, info['functions'])

        with open(header_path, 'w') as f:
            f.write(header_content)
        print(f"✓ Created header: {header_path}")

        # Extract implementation content
        impl_content = ""
        for func_name, (start, end) in zip(info['functions'], info['line_ranges']):
            print(f"  Extracting {func_name} (lines {start}-{end})")
            impl_content += extract_function_block(lines, start, end)
            impl_content += "\n"

        # Create implementation file
        impl_path = os.path.join(output_dir, f"ldm_mdata_{module_name}.cu")
        impl_file_content = create_implementation_file(module_name, impl_content)

        with open(impl_path, 'w') as f:
            f.write(impl_file_content)
        print(f"✓ Created implementation: {impl_path}")
        print(f"  Lines: {impl_content.count(chr(10))}")

def main():
    project_root = Path(__file__).parent.parent
    input_file = project_root / "src/include/ldm_mdata.cuh"
    output_dir = project_root / "src/data/meteo"

    print("LDM Meteorological Data File Splitter")
    print("="*60)

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        return 1

    split_mdata_file(str(input_file), str(output_dir))

    print("\n" + "="*60)
    print("✓ File splitting completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - src/data/meteo/ldm_mdata_loading.cuh/cu")
    print("  - src/data/meteo/ldm_mdata_processing.cuh/cu")
    print("  - src/data/meteo/ldm_mdata_cache.cuh/cu")

    return 0

if __name__ == "__main__":
    exit(main())
