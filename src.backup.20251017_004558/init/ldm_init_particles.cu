/**
 * @file ldm_init_particles.cu
 * @brief Implementation of particle initialization functions
 */

#include "../core/ldm.cuh"
#include "../physics/ldm_nuclides.cuh"
#include "colors.h"
#include <chrono>

void LDM::initializeParticles(){
    if (concentrations.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No concentrations data loaded for particle initialization" << std::endl;
        return;
    }
    
    if (sources.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No sources data loaded for particle initialization" << std::endl;
        return;
    }
    
    int partPerConc = nop / concentrations.size();

    std::random_device rd;
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count() + PROCESS_INDEX * 1000;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(g_mpi.particleSizes[PROCESS_INDEX], g_mpi.sizeStandardDeviations[PROCESS_INDEX]);

    int particle_count = 0;
    for (const auto& conc : concentrations) {
        if (conc.location - 1 >= sources.size()) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Invalid source location index: " << conc.location << " (max: " << sources.size() << ")" << std::endl;
            continue;
        }

        for (int i = 0; i < partPerConc; ++i) {
            float x = (sources[conc.location - 1].lon + 179.0) / 0.5;
            float y = (sources[conc.location - 1].lat + 90) / 0.5;
            float z = sources[conc.location - 1].height;

            float random_radius = dist(gen);

            part.push_back(LDMpart(x, y, z,
                                   g_mpi.decayConstants[PROCESS_INDEX],
                                   conc.value,
                                   g_mpi.depositionVelocities[PROCESS_INDEX],
                                   random_radius,
                                   g_mpi.particleDensities[PROCESS_INDEX],
                                   i + 1));
            
            // Initialize multi-nuclide concentrations for the newly created particle
            LDMpart& current_particle = part.back();
            
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

    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

void LDM::initializeParticlesEKI(){
    // EKI mode particle initialization using true_emissions from g_eki
    if (g_eki.true_emissions.empty()) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "No true_emissions data loaded for EKI particle initialization" << std::endl;
        return;
    }
    
    if (sources.empty()) {
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
    part.clear();

    for (int i = 0; i < nop; ++i) {
        // Use first source location (same as original)
        float x = (sources[0].lon + 179.0) / 0.5;
        float y = (sources[0].lat + 90) / 0.5;
        float z = sources[0].height;

        float random_radius = dist(gen);

        // Use true_emissions value for this time interval
        float emission_value = g_eki.true_emissions[interval_idx];

        // For single mode EKI: set all particles to timeidx=1 so they activate immediately
        // (different from ensemble mode where particles activate at different times)
        part.push_back(LDMpart(x, y, z,
                               g_mpi.decayConstants[PROCESS_INDEX],
                               emission_value,
                               g_mpi.depositionVelocities[PROCESS_INDEX],
                               random_radius,
                               g_mpi.particleDensities[PROCESS_INDEX],
                               i+1));  // Changed from (i+1) to (1) for immediate activation
        
        // Initialize multi-nuclide concentrations for the newly created particle
        LDMpart& current_particle = part.back();
        
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

    std::cout << Color::GREEN << "  " << Color::RESET
              << "Initialized " << Color::BOLD << part.size() << Color::RESET
              << " particles" << std::endl;

    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        return a.timeidx < b.timeidx;
    });
}

void LDM::initializeParticlesEKI_AllEnsembles(float* ensemble_states, int num_ensembles, int num_timesteps) {
    // EKI ensemble mode: Initialize particles for all ensemble members in parallel
    // ensemble_states: [num_ensembles × num_timesteps] matrix (e.g., 100×24)

    if (sources.empty()) {
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
    part.clear();
    part.reserve(total_particles);

    // Generate particles for each ensemble
    for (int ens = 0; ens < num_ensembles; ens++) {
        for (int t = 0; t < num_timesteps; t++) {
            // Current timestep's emission rate (ensemble_states is row-major: [ens * num_timesteps + t])
            float emission_value = ensemble_states[ens * num_timesteps + t];

            // Generate particles for this timestep
            for (int p = 0; p < particles_per_timestep; p++) {
                // Source location
                float x = (sources[0].lon + 179.0) / 0.5;
                float y = (sources[0].lat + 90) / 0.5;
                float z = sources[0].height;
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

                part.push_back(particle);
            }
        }

        // Progress indicator (only show every 20%)
        if ((ens + 1) % (num_ensembles / 5) == 0 || (ens + 1) == num_ensembles) {
            std::cout << "\r  Progress: " << (ens + 1) << "/" << num_ensembles
                      << " (" << ((ens + 1) * 100 / num_ensembles) << "%)" << std::flush;
        }
    }

    std::cout << "\r" << Color::GREEN << "  " << Color::RESET
              << "Created " << Color::BOLD << part.size() << Color::RESET << " particles";

    // CRITICAL: Sort by ensemble_id (independent simulations)
    std::sort(part.begin(), part.end(), [](const LDMpart& a, const LDMpart& b) {
        if (a.ensemble_id != b.ensemble_id)
            return a.ensemble_id < b.ensemble_id;
        else
            return a.timeidx < b.timeidx;
    });

    std::cout << " (sorted by ensemble)" << std::endl;
}

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
