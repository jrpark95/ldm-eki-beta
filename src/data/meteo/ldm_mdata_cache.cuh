/******************************************************************************
 * @file ldm_mdata_cache.cuh
 * @brief Meteorological data caching module header
 *
 * @details Forward declaration header for EKI meteorological data preloading
 *          and caching functions. This module enables high-performance ensemble
 *          simulations by eliminating file I/O overhead.
 *
 *          **Module Functions (declared in ldm.cuh):**
 *          - LDM::calculateRequiredMeteoFiles() - Compute file count from settings
 *          - LDM::loadSingleMeteoFile() - Thread-safe single file loader
 *          - LDM::preloadAllEKIMeteorologicalData() - Parallel preload master
 *          - LDM::cleanupEKIMeteorologicalData() - Memory cleanup
 *
 *          **Global Cache:**
 *          - g_eki_meteo (EKIMeteoCache) - Defined in ldm.cu
 *          - Stores all timesteps in GPU memory
 *          - Accessed by ensemble kernels for fast retrieval
 *
 *          **Implementation:**
 *          - See ldm_mdata_cache.cu for full implementations
 *
 * @note Header-only file with forward declaration
 * @note Critical for EKI performance (20+ iterations per inversion)
 * @note Part of modularized meteorological data management system
 * @note Include ldm.cuh for actual method declarations
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#pragma once

// Forward declaration only - method declarations are in ldm.cuh
class LDM;
