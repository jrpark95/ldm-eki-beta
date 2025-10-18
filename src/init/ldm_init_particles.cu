/******************************************************************************
 * @file ldm_init_particles.cu
 * @brief Implementation of particle initialization functions for LDM simulation
 *
 * @details Provides three particle initialization modes:
 *          1. Standard mode: Concentration-based particle distribution
 *          2. EKI single mode: True emission time series for observations
 *          3. EKI ensemble mode: Multiple ensemble member initialization
 *
 *          All modes properly initialize multi-nuclide concentration arrays
 *          and particle physical properties (size, density, decay constants).
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"
#include <chrono>

/******************************************************************************
 * @brief Initialize particles for standard simulation mode
 *
 * @details Creates particles based on source locations and concentration values
 *          defined in source.txt [RELEASE_CASES] section. Particles are
 *          distributed evenly across release cases with randomized physical
 *          properties (size distribution).
 *
 * @pre Configuration must be loaded:
 *      - concentrations: Release case emission values (Bq/s)
 *      - sources: Source coordinates (lon, lat, height)
 *      - nop: Total number of particles
 *      - g_mpi.particleSizes[PROCESS_INDEX]: Mean particle radius (μm)
 *      - g_mpi.sizeStandardDeviations[PROCESS_INDEX]: Size distribution spread
 *
 * @post part vector populated with initialized particles
 * @post Particles sorted by timeidx for proper activation sequencing
 * @post Multi-nuclide concentrations initialized for each particle
 *
 * @algorithm
 *   1. Calculate particles per concentration case (partPerConc)
 *   2. Initialize random number generator with time-based seed
 *   3. For each concentration case:
 *      - Retrieve source location (lon, lat, height)
 *      - Convert to grid coordinates: x = (lon+179)/0.5, y = (lat+90)/0.5
 *      - Generate partPerConc particles:
 *        * Sample radius from normal distribution
 *        * Create LDMpart with physical properties
 *        * Initialize multi-nuclide concentrations using config ratios
 *   4. Sort particles by timeidx (activation order)
 *
 * @note Coordinate system:
 *       - Grid origin: 179°W, 90°S
 *       - Grid spacing: 0.5°
 *       - Height: meters above ground
 * @note Particle size: Sampled from normal(mean, std) in micrometers
 * @note Multi-nuclide initialization: Uses initial_ratio from NuclideConfig
 *
 * @see initializeParticlesEKI() for EKI single mode
 * @see initializeParticlesEKI_AllEnsembles() for EKI ensemble mode
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeParticles(){
    if (h_concentrations.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No concentrations data loaded for particle initialization" << std::endl;
        return;
    }

    if (h_sources.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No sources data loaded for particle initialization" << std::endl;
        return;
    }

    int partPerConc = nop / h_concentrations.size();

    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    int particle_count = 0;
    for (const auto& conc : h_concentrations) {
        if (conc.location - 1 >= h_sources.size()) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Invalid source location index: " << conc.location << " (max: " << h_sources.size() << ")" << std::endl;
            continue;
        }

        for (int i = 0; i < partPerConc; ++i) {
            float x = (h_sources[conc.location - 1].lon + 179.0) / 0.5;
            float y = (h_sources[conc.location - 1].lat + 90) / 0.5;
            float z = h_sources[conc.location - 1].height;

            float random_radius = dist(gen);

            h_part.push_back(LDMpart(x, y, z,
                                   g_mpi.decayConstants[PROCESS_INDEX],
                                   conc.value,
                                   g_mpi.depositionVelocities[PROCESS_INDEX],
                                   random_radius,
                                   g_mpi.particleDensities[PROCESS_INDEX],
                                   i + 1));

            // Initialize multi-nuclide concentrations for the newly created particle
            LDMpart& current_particle = h_part.back();

            // Get nuclide configuration
            NuclideConfig* nucConfig = NuclideConfig::getInstance();
            int num_nuclides = nucConfig->getNumNuclides();


            // Set up multi-nuclide concentrations - use individual initial ratios from config
            for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                if (nuc < num_nuclides) {
                    // Use individual initial ratio for each nuclide from config file
                    float initial_ratio = nucConfig->getInitialRatio(nuc);
                    current_particle.concentrations[nuc] = conc.value * initial_ratio;

                } else {
                    current_particle.concentrations[nuc] = 0.0f;
                }
            }

            particle_count++;
        }
    }

    std::sort(h_part.begin(), h_part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

/******************************************************************************
 * @brief Initialize particles for EKI single-mode simulation
 *
 * @details Creates particles using true_emissions time series from EKI config.
 *          Generates "truth" observations for Ensemble Kalman Inversion by
 *          simulating particles with known emission rates. Particles are
 *          distributed across time intervals to represent time-varying emissions.
 *
 * @pre Configuration must be loaded:
 *      - g_eki.true_emissions: Time series of emission rates (Bq/s)
 *      - sources: At least one source location (first source used)
 *      - nop: Total number of particles (should be divisible by num_timesteps)
 *      - g_mpi particle properties: size, density, decay constants
 *
 * @post part vector cleared and repopulated with new particles
 * @post Particles sorted by timeidx
 * @post All particles activate immediately (timeidx = i+1)
 * @post Multi-nuclide concentrations initialized for each particle
 *
 * @algorithm
 *   1. Calculate particles_per_interval = nop / true_emissions.size()
 *   2. Initialize random number generator
 *   3. For each particle i in [0, nop):
 *      - Determine time interval index
 *      - Get emission_value from true_emissions[interval_idx]
 *      - Generate particle with:
 *        * Source location from sources[0]
 *        * Random radius from size distribution
 *        * Emission value from true_emissions
 *        * timeidx = i+1 for sequential activation
 *      - Initialize multi-nuclide concentrations
 *      - Advance interval when particles_per_interval reached
 *   4. Sort particles by timeidx
 *
 * @note Difference from ensemble mode: All particles activate immediately
 *       (timeidx increments for each particle, no delayed activation)
 * @note Used for generating "truth" observations in EKI workflow
 * @note Only first source location (sources[0]) is used
 *
 * @output Console: "[ENSEMBLE] Initializing N particles (M time steps)"
 * @output Console: "✓ Initialized N particles"
 *
 * @see initializeParticlesEKI_AllEnsembles() for ensemble mode
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeParticlesEKI(){
    // EKI mode particle initialization using true_emissions from g_eki
    if (g_eki.true_emissions.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No true_emissions data loaded for EKI particle initialization" << std::endl;
        return;
    }

    if (h_sources.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No sources data loaded for EKI particle initialization" << std::endl;
        return;
    }

    std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
              << "Initializing " << nop << " particles ("
              << g_eki.true_emissions.size() << " time steps)" << std::endl;

    int particles_per_interval = nop / g_eki.true_emissions.size();

    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    int interval_idx = 0;
    int count_in_interval = 0;
    int particle_count = 0;

    // Clear existing particles
    h_part.clear();

    for (int i = 0; i < nop; ++i) {
        // Use first source location (same as original)
        float x = (h_sources[0].lon + 179.0) / 0.5;
        float y = (h_sources[0].lat + 90) / 0.5;
        float z = h_sources[0].height;

        float random_radius = dist(gen);

        // Use true_emissions value for this time interval
        float emission_value = g_eki.true_emissions[interval_idx];

        // For single mode EKI: set all particles to timeidx=1 so they activate immediately
        // (different from ensemble mode where particles activate at different times)
        h_part.push_back(LDMpart(x, y, z,
                               g_mpi.decayConstants[PROCESS_INDEX],
                               emission_value,
                               g_mpi.depositionVelocities[PROCESS_INDEX],
                               random_radius,
                               g_mpi.particleDensities[PROCESS_INDEX],
                               i+1));  // Changed from (i+1) to (1) for immediate activation

        // Initialize multi-nuclide concentrations for the newly created particle
        LDMpart& current_particle = h_part.back();

        // Get nuclide configuration
        NuclideConfig* nucConfig = NuclideConfig::getInstance();
        int num_nuclides = nucConfig->getNumNuclides();

        // Initialize multi-nuclide concentrations array
        // IMPORTANT: This is required because move_part_by_wind_mpi kernel updates p.conc as sum of concentrations[]
        for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
            if (nuc < num_nuclides) {
                float initial_ratio = nucConfig->getInitialRatio(nuc);
                current_particle.concentrations[nuc] = emission_value * initial_ratio;
            } else {
                current_particle.concentrations[nuc] = 0.0f;
            }
        }

        particle_count++;
        count_in_interval++;

        // Move to next time interval when we've allocated enough particles
        if (count_in_interval == particles_per_interval) {
            interval_idx++;
            count_in_interval = 0;

            // Don't exceed the available time steps
            if (interval_idx >= g_eki.true_emissions.size()) {
                interval_idx = g_eki.true_emissions.size() - 1;
            }
        }
    }

    std::cout << Color::GREEN << "  ✓ " << Color::RESET
              << "Initialized " << Color::BOLD << h_part.size() << Color::RESET
              << " particles" << std::endl;

    std::sort(h_part.begin(), h_part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

/******************************************************************************
 * @brief Initialize particles for EKI ensemble simulation mode
 *
 * @details Creates particles for multiple ensemble members simultaneously.
 *          Each ensemble member has its own emission time series, and particles
 *          are tagged with ensemble_id for independent parallel simulation.
 *          Critical for ensemble Kalman inversion workflow.
 *
 * @param[in] ensemble_states Emission rates [num_ensembles × num_timesteps]
 *                            - Row-major layout: states[ens*num_timesteps + t]
 *                            - Units: Bq/s (Becquerels per second)
 *                            - Typical size: 50-100 ensembles × 24-48 timesteps
 * @param[in] num_ensembles Number of ensemble members
 *                          - Typical range: 50-100
 *                          - Determines statistical quality of inversion
 * @param[in] num_timesteps Number of time intervals in emission series
 *                          - Matches temporal resolution of observations
 *                          - Typical: 24-48 (hourly/half-hourly for 1-2 days)
 *
 * @pre Configuration must be loaded:
 *      - sources: At least one source location (first source used)
 *      - nop: Particles per ensemble (total = nop * num_ensembles)
 *      - g_mpi particle properties: size distribution, density, decay
 *
 * @post part vector contains (nop * num_ensembles) particles
 * @post Particles sorted by (ensemble_id, timeidx)
 * @post Each particle has valid ensemble_id ∈ [0, num_ensembles-1]
 * @post timeidx ranges from 1 to (nop) for each ensemble
 *
 * @algorithm
 *   1. Calculate dimensions:
 *      - particles_per_ensemble = nop
 *      - particles_per_timestep = nop / num_timesteps
 *      - total_particles = nop * num_ensembles
 *   2. Reserve memory: part.reserve(total_particles)
 *   3. For each ensemble ∈ [0, num_ensembles):
 *        For each timestep ∈ [0, num_timesteps):
 *          - Fetch emission_value = ensemble_states[ens * num_timesteps + t]
 *          - Generate particles_per_timestep particles:
 *            * Source location from sources[0]
 *            * Random radius from size distribution
 *            * timeidx = t * particles_per_timestep + p + 1
 *            * ensemble_id = ens
 *            * Initialize multi-nuclide concentrations
 *   4. Sort particles by (ensemble_id, timeidx)
 *
 * @note CRITICAL: Particles MUST be sorted by ensemble_id for GPU kernel efficiency
 * @note Each ensemble simulates independently with its own emission series
 * @note timeidx is local to each ensemble (1 to particles_per_ensemble)
 *
 * @performance
 *   - Memory: O(num_ensembles * nop) particles
 *   - Time: O(num_ensembles * nop) particle generation
 *   - Sorting: O(N log N) where N = total particles
 *   - Example: 100 ensembles × 10,000 particles = 1 million particles
 *
 * @output Console progress updates:
 *   - Initial: "[ENSEMBLE] Initializing N ensembles × M timesteps (P particles)"
 *   - Progress: "Progress: X/N (Y%)" every 20%
 *   - Final: "✓ Created P particles (sorted by ensemble)"
 *
 * @warning Large ensemble sizes (>100) with large nop (>100k) can exceed GPU memory
 * @warning Particles must be sorted for correct GPU kernel execution
 *
 * @see initializeParticlesEKI() for single-mode truth simulation
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::initializeParticlesEKI_AllEnsembles(float* ensemble_states, int num_ensembles, int num_timesteps) {
    // EKI ensemble mode: Initialize particles for all ensemble members in parallel
    // ensemble_states: [num_ensembles × num_timesteps] matrix (e.g., 100×24)

    if (h_sources.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No sources data loaded for EKI ensemble initialization" << std::endl;
        return;
    }

    int particles_per_ensemble = nop;  // Particle count specified in setting.txt
    int particles_per_timestep = particles_per_ensemble / num_timesteps;
    int total_particles = particles_per_ensemble * num_ensembles;

    std::cout << Color::YELLOW << "[ENSEMBLE] " << Color::RESET
              << "Initializing " << Color::BOLD << num_ensembles << Color::RESET
              << " ensembles × " << Color::BOLD << num_timesteps << Color::RESET
              << " timesteps (" << Color::BOLD << total_particles << Color::RESET << " particles)" << std::endl;


    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    // Clear and reserve memory
    h_part.clear();
    h_part.reserve(total_particles);

    // Generate particles for each ensemble
    for (int ens = 0; ens < num_ensembles; ens++) {
        for (int t = 0; t < num_timesteps; t++) {
            // Current timestep's emission rate (ensemble_states is row-major: [ens * num_timesteps + t])
            float emission_value = ensemble_states[ens * num_timesteps + t];

            // Generate particles for this timestep
            for (int p = 0; p < particles_per_timestep; p++) {
                // Source location
                float x = (h_sources[0].lon + 179.0) / 0.5;
                float y = (h_sources[0].lat + 90) / 0.5;
                float z = h_sources[0].height;
                float random_radius = dist(gen);

                // timeidx: Local particle ID within each ensemble (1~particles_per_ensemble)
                // All ensembles have identical timeidx range for independent activation
                int timeidx = t * particles_per_timestep + p + 1;

                // Generate particle
                LDMpart particle(x, y, z,
                                g_mpi.decayConstants[PROCESS_INDEX],
                                emission_value,
                                g_mpi.depositionVelocities[PROCESS_INDEX],
                                random_radius,
                                g_mpi.particleDensities[PROCESS_INDEX],
                                timeidx);

                // Set ensemble ID
                particle.ensemble_id = ens;

                // Initialize multi-nuclide concentrations array
                // IMPORTANT: This is required because move_part_by_wind_mpi kernel updates p.conc as sum of concentrations[]
                NuclideConfig* nucConfig = NuclideConfig::getInstance();
                int num_nuclides = nucConfig->getNumNuclides();
                for(int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                    if (nuc < num_nuclides) {
                        float initial_ratio = nucConfig->getInitialRatio(nuc);
                        particle.concentrations[nuc] = emission_value * initial_ratio;
                    } else {
                        particle.concentrations[nuc] = 0.0f;
                    }
                }

                h_part.push_back(particle);
            }
        }

        // Progress indicator (only show every 20%)
        if ((ens + 1) % (num_ensembles / 5) == 0 || (ens + 1) == num_ensembles) {
            std::cout << "\r  Progress: " << (ens + 1) << "/" << num_ensembles
                      << " (" << ((ens + 1) * 100 / num_ensembles) << "%)" << std::flush;
        }
    }

    std::cout << "\r" << Color::GREEN << "  ✓ " << Color::RESET
              << "Created " << Color::BOLD << h_part.size() << Color::RESET << " particles";

    // CRITICAL: Sort by ensemble_id (independent simulations)
    std::sort(h_part.begin(), h_part.end(), [](const LDMpart& a, const LDMpart& b) {
        if (a.ensemble_id != b.ensemble_id)
            return a.ensemble_id < b.ensemble_id;
        else
            return a.timeidx < b.timeidx;
    });

    std::cout << " (sorted by ensemble)" << std::endl;
}

/******************************************************************************
 * @brief Calculate average settling velocity using hardcoded particle properties
 *
 * @details Legacy function for computing settling velocity (vsetaver) and
 *          Cunningham correction factor using hardcoded particle properties.
 *          Integrates over log-normal size distribution using NI=11 intervals.
 *
 * @pre None (uses hardcoded constants)
 *
 * @post vsetaver: Average settling velocity (m/s, negative for downward)
 * @post cunningham: Average Cunningham correction factor (dimensionless)
 * @post Values passed to GPU kernels via KernelScalars struct
 *
 * @algorithm
 *   1. Define particle properties:
 *      - prho = 2500.0 kg/m³ (particle density)
 *      - radi = 0.6 μm (mean radius)
 *      - dsig = 0.3 (geometric standard deviation)
 *   2. Generate NI=11 size fractions using log-normal distribution
 *   3. For each size fraction:
 *      - Calculate Knudsen number: kn = 2*λ/d
 *      - Compute Cunningham factor: cun = 1 + alpha*kn
 *      - Compute settling velocity: vsh = g*ρ*d²*cun/(18*μ) (Stokes law)
 *   4. Average over all size fractions weighted by fract[i]
 *   5. Store results in vsetaver and cunningham members
 *
 * @note Hardcoded particle properties:
 *       - Density: 2500 kg/m³ (typical mineral dust)
 *       - Radius: 0.6 μm (mean)
 *       - Size std: 0.3 (geometric)
 * @note Cunningham correction accounts for slip flow in small particles
 * @note vsetaver is negative (downward settling)
 *
 * @deprecated Use calculateSettlingVelocity() instead for configurable properties
 *
 * @see calculateSettlingVelocity() for configurable version
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::calculateAverageSettlingVelocity(){

            // settling rho, temp
            float prho = 2500.0;
            float radi = 6.0e-7*1.0e+6; // m to um
            float dsig = 3.0e-1;

            float xdummy = sqrt(2.0f)*logf(dsig);
            float delta = 6.0/static_cast<float>(NI);
            float d01 = radi*pow(dsig,-3.0);
            float d02, x01, x02, dmean, kn, alpha, cun, dc;

            float fract[NI] = {0, };
            float schmi[NI] = {0, };
            float vsh[NI] = {0, };


            for(int i=1; i<NI+1; i++){
                d02 = d01;
                d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
                x01 = logf(d01/radi)/xdummy;
                x02 = logf(d02/radi)/xdummy;


                fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
                dmean = 1.0e-6*exp(0.5*log(d01*d02));
                kn = 2.0f*_lam/dmean;


                if(-1.1/kn <= log10(_eps)*log(10.0))
                    alpha = 1.257;
                else
                    alpha = 1.257+0.4*exp(-1.1/kn);

                cun = 1.0 + alpha*kn;
                dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
                //schmi = pow(_nyl/dc,-2.0/3.0);
                vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

            }

            for(int i=1; i<NI+1; i++){
                vsetaver -= fract[i-1]*vsh[i-1];
                cunningham += fract[i-1]*cun;
            }

            // Note: vsetaver and cunningham are now passed via KernelScalars, no need for cudaMemcpyToSymbol


}

/******************************************************************************
 * @brief Calculate average settling velocity using configuration-based particle properties
 *
 * @details Computes settling velocity (vsetaver) and Cunningham correction factor
 *          for particles with properties loaded from configuration file. Integrates
 *          over log-normal size distribution using NI=11 size intervals. Uses
 *          Stokes law with slip correction for small particles.
 *
 * @pre Configuration must be loaded:
 *      - g_mpi.particleDensities[PROCESS_INDEX]: Particle density (kg/m³)
 *      - g_mpi.particleSizes[PROCESS_INDEX]: Mean radius (μm)
 *      - g_mpi.sizeStandardDeviations[PROCESS_INDEX]: Size distribution width
 *
 * @post vsetaver: Average settling velocity (m/s, negative for downward)
 * @post cunningham: Average Cunningham correction factor (dimensionless)
 * @post Values passed to GPU kernels via KernelScalars struct
 *
 * @algorithm
 *   1. Load particle properties from g_mpi configuration
 *   2. Handle special case: radi==0 (gas-phase species)
 *      - Set vsetaver=1.0, cunningham=-1.0 (flag values)
 *   3. Generate NI=11 size fractions using log-normal distribution:
 *      - fract[i] = fraction of particles in size interval i
 *   4. For each size fraction i:
 *      - Calculate mean diameter: dmean
 *      - Compute Knudsen number: kn = 2*λ/d
 *        (λ = mean free path = _lam)
 *      - Compute Cunningham correction factor:
 *        alpha = 1.257 + 0.4*exp(-1.1/kn)
 *        cun = 1 + alpha*kn
 *      - Compute settling velocity (Stokes law with slip correction):
 *        vsh[i] = g*ρ*d²*cun/(18*μ)
 *        (g = _ga, μ = _myl)
 *   5. Average over all size fractions:
 *      vsetaver = Σ fract[i] * vsh[i]
 *      cunningham = Σ fract[i] * cun[i]
 *
 * @note Physical constants used:
 *       - _ga: Gravitational acceleration (9.81 m/s²)
 *       - _myl: Dynamic viscosity of air (~1.8×10⁻⁵ kg/(m·s))
 *       - _lam: Mean free path of air (~6.6×10⁻⁸ m)
 *       - _kb: Boltzmann constant (1.38×10⁻²³ J/K)
 *       - _Tr: Reference temperature (K)
 * @note Cunningham correction: Accounts for slip flow when particle size
 *       approaches mean free path of air (important for particles <1 μm)
 * @note vsetaver is negative (downward settling by convention)
 *
 * @see calculateAverageSettlingVelocity() for legacy hardcoded version
 *
 * @author Juryong Park
 * @date 2025
 *****************************************************************************/
void LDM::calculateSettlingVelocity(){
    float prho = g_mpi.particleDensities[PROCESS_INDEX];
    float radi = g_mpi.particleSizes[PROCESS_INDEX];
    float dsig = g_mpi.sizeStandardDeviations[PROCESS_INDEX];

    if (radi == 0.0f) {
        vsetaver = 1.0f;
        cunningham = -1.0f;
        // Note: vsetaver and cunningham are now passed via KernelScalars, no need for cudaMemcpyToSymbol
        return;
    }

    float xdummy = sqrt(2.0f)*logf(dsig);
    float delta = 6.0/static_cast<float>(NI);
    float d01 = radi*pow(dsig,-3.0);
    float d02, x01, x02, dmean, kn, alpha, cun, dc;

    float fract[NI] = {0, };
    float schmi[NI] = {0, };
    float vsh[NI] = {0, };

    vsetaver = 0.0f;
    cunningham = 0.0f;

    for(int i=1; i<NI+1; i++){
        d02 = d01;
        d01 = radi*pow(dsig, -3.0 + delta*static_cast<float>(i));
        x01 = logf(d01/radi)/xdummy;
        x02 = logf(d02/radi)/xdummy;


        fract[i-1] = 0.5*(std::erf(x01)-std::erf(x02));
        dmean = 1.0e-6*exp(0.5*log(d01*d02));
        kn = 2.0f*_lam/dmean;


        if(-1.1/kn <= log10(_eps)*log(10.0))
            alpha = 1.257;
        else
            alpha = 1.257+0.4*exp(-1.1/kn);

        cun = 1.0 + alpha*kn;
        dc = _kb*_Tr*cun/(3.0*PI*_myl*dmean);
        //schmi = pow(_nyl/dc,-2.0/3.0);
        vsh[i-1] = _ga*prho*dmean*dmean*cun/(18.0*_myl);

    }

    for(int i=1; i<NI+1; i++){
        vsetaver -= fract[i-1]*vsh[i-1];
        cunningham += fract[i-1]*cun;
    }

    // Note: vsetaver and cunningham are now passed via KernelScalars, no need for cudaMemcpyToSymbol
}
