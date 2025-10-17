#pragma once

/**
 * @file ldm_func_output.cuh
 * @brief Output handling, observation system, and logging
 *
 * This file contains output-related functions for the LDM simulation.
 * Part of the LDM-EKI simulation framework.
 *
 * @note This header only provides forward declarations.
 *       Actual method declarations are in ldm.cuh
 *       Implementations are in ldm_func_output.cu
 */

// Forward declaration only - method declarations are in ldm.cuh
class LDM;

// Test function to verify g_logonly cross-compilation-unit access
void test_g_logonly_from_output_module();
