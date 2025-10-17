/******************************************************************************
 * @file ldm_func_particle.cuh
 * @brief Particle management and GPU memory operations
 *
 * This module handles particle data structures and their transfer between
 * CPU and GPU memory. Key responsibilities include:
 *
 * - GPU memory allocation and deallocation for particle arrays
 * - Host-to-device and device-to-host data transfers
 * - Memory verification and integrity checking
 * - Diagnostic tools for detecting NaN and numerical instabilities
 *
 * Particle Structure Management:
 * - Each particle contains: position (x,y,z), velocity, concentration, nuclide info
 * - Particles are stored in structure-of-arrays (SoA) layout for coalesced GPU access
 * - Memory alignment optimized for GPU cache performance
 *
 * Debug Capabilities:
 * - NaN detection in particle position, velocity, and meteorological fields
 * - Memory transfer verification for GPU consistency
 * - Particle state logging at key simulation stages
 *
 * @note This header only provides forward declarations.
 *       Actual method declarations are in ldm.cuh
 *       Implementations are in ldm_func_particle.cu
 *
 * @see ldm.cuh for LDM class method declarations
 * @see ldm_func_particle.cu for implementation details
 * @see ldm_struct.cuh for LDMpart structure definition
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// Forward declaration only - method declarations are in ldm.cuh
class LDM;
