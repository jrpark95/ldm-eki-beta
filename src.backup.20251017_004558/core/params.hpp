#pragma once

// Kernel scalar parameters structure
// Replaces __constant__ scalar variables to avoid invalid device symbol errors in non-RDC mode
// NOTE: Field names avoid macro conflicts (nop, isRural, isPG, etc. are macros in ldm.cuh)
struct alignas(16) KernelScalars {
    // Simulation switches
    int turb_switch;
    int drydep;
    int wetdep;
    int raddecay;

    // Particle and model parameters
    int num_particles;    // nop
    int is_rural;         // isRural
    int is_pg;            // isPG
    int is_gfs;           // isGFS

    // Time and grid parameters
    float delta_time;     // dt
    float grid_start_lat; // start_lat
    float grid_start_lon; // start_lon
    float grid_lat_step;  // lat_step
    float grid_lon_step;  // lon_step

    // Physics parameters
    float settling_vel;   // vsetaver
    float cunningham_fac; // cunningham

    // CRAM decay matrix pointer
    const float* T_matrix;  // Decay transition matrix (N_NUCLIDES × N_NUCLIDES)

    // Height data pointer
    const float* flex_hgt;  // Vertical height levels (dimZ_GFS elements)
};
