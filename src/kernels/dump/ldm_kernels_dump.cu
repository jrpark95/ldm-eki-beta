/**
 * @file ldm_kernels_dump.cu
 * @brief Implementation of particle advection with VTK output (single mode)
 */

#include "ldm_kernels_dump.cuh"

__global__ void advectParticlesWithVTK(
    LDM::LDMpart* d_part, float t0, int rank, float* d_dryDep, float* d_wetDep, int mesh_nx, int mesh_ny,
    FlexUnis* d_surface_past,
    FlexPres* d_pressure_past,
    FlexUnis* d_surface_future,
    FlexPres* d_pressure_future,
    const KernelScalars ks){

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= ks.num_particles) return;
        //if(idx != 0) return;  // Process all particles

        // Debug output disabled for performance

        LDM::LDMpart& p = d_part[idx];
        if(!p.flag) {
            return;
        }
        
        // Debug disabled for performance

        // Direct use of T_const instead of shared memory copy

        unsigned long long seed = static_cast<unsigned long long>(t0 * ULLONG_MAX);  // Time-dependent seed like CRAM
        curandState ss;
        curand_init(seed, idx, 0, &ss);


        int xidx, yidx;
        if(p.x*0.5 -179.0 >= 180.0) {
            xidx = 1;
            p.flag=false;
        }
        else xidx = int(p.x);
        yidx = int(p.y);

        int zidx = 0;
        int index;

        float fdump = 0;

        float hmix = 0;
        for(int i=0; i<2; i++){
            for(int j=0; j<2; j++){
                index = (xidx+i) * dimY_GFS + (yidx+j);
                hmix = max(hmix, d_surface_past[index].HMIX);
                hmix = max(hmix, d_surface_future[index].HMIX);
            }
        }

        // Debug disabled for performance

        float zeta = p.z/hmix;
        
        for(int i=0; i<dimZ_GFS; i++){
            if(ks.height_levels[i] > p.z){
                zidx = i-1;  // Fixed: use lower level index like CRAM
                break;
            }
        }
        if(zidx < 0) zidx = 0;  // Ensure non-negative index

        float x0 = p.x-xidx;
        float y0 = p.y-yidx;
        
        // CRITICAL FIX: Handle near-zero height differences to avoid division by zero
        // If height difference is close to zero, use safe value
        float height_diff = ks.height_levels[zidx+1] - ks.height_levels[zidx];
        float z0;
        if (abs(height_diff) < 1e-6f) {
            z0 = 0.0f;
        } else {
            z0 = (p.z - ks.height_levels[zidx]) / height_diff;
        }
        
        float x1 = 1-x0;
        float y1 = 1-y0;
        float z1 = 1-z0;
        float t1 = 1-t0;
        
        // Debug disabled for performance

        float ustr = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].USTR
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].USTR
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].USTR
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].USTR
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].USTR
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].USTR
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].USTR
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].USTR;

        float wstr = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].WSTR
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].WSTR
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].WSTR
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].WSTR
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].WSTR
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].WSTR
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].WSTR
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].WSTR;

        float obkl = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].OBKL
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].OBKL
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].OBKL
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].OBKL
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].OBKL
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].OBKL
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].OBKL
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].OBKL;

        obkl = 1/obkl;

        float vdep = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].VDEP
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].VDEP
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].VDEP
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].VDEP
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].VDEP
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].VDEP
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].VDEP
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].VDEP;

        float lsp = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].LPREC
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].LPREC
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].LPREC
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].LPREC
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].LPREC
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].LPREC
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].LPREC
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].LPREC;

        float convp = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].CPREC
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].CPREC
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].CPREC
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].CPREC
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].CPREC
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].CPREC
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].CPREC
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].CPREC;

        float cc = x1*y1*t1*d_surface_past[(xidx) * dimY_GFS + (yidx)].TCC
                    +x0*y1*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx)].TCC
                    +x1*y0*t1*d_surface_past[(xidx) * dimY_GFS + (yidx+1)].TCC
                    +x0*y0*t1*d_surface_past[(xidx+1) * dimY_GFS + (yidx+1)].TCC
                    +x1*y1*t0*d_surface_future[(xidx) * dimY_GFS + (yidx)].TCC
                    +x0*y1*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx)].TCC
                    +x1*y0*t0*d_surface_future[(xidx) * dimY_GFS + (yidx+1)].TCC
                    +x0*y0*t0*d_surface_future[(xidx+1) * dimY_GFS + (yidx+1)].TCC;


        // Debug: Check individual DRHO values before interpolation
        float drho_000 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO;
        float drho_100 = d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO;
        
        if (idx == 0 && isnan(drho_000)) {
// printf("[DRHO_DEBUG] DRHO_000 is NaN at indices [%d,%d,%d,%d]\n", xidx, yidx, zidx, 0);
        }
        if (idx == 0 && isnan(drho_100)) {
// printf("[DRHO_DEBUG] DRHO_100 is NaN at indices [%d,%d,%d,%d]\n", xidx+1, yidx, zidx, 0);
        }
        
        float drho_raw = x1*y1*z1*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO
                    +x0*y1*z1*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO
                    +x1*y0*z1*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].DRHO
                    +x0*y0*z1*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].DRHO
                    +x1*y1*z0*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].DRHO
                    +x0*y1*z0*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].DRHO
                    +x1*y0*z0*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].DRHO
                    +x0*y0*z0*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].DRHO
                    +x1*y1*z1*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO
                    +x0*y1*z1*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].DRHO
                    +x1*y0*z1*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].DRHO
                    +x0*y0*z1*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].DRHO
                    +x1*y1*z0*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].DRHO
                    +x0*y1*z0*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].DRHO
                    +x1*y0*z0*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].DRHO
                    +x0*y0*z0*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].DRHO;
        
        // Fix NaN issue: replace NaN with 0
        float drho = isnan(drho_raw) ? 0.0f : drho_raw;

        float rho = x1*y1*z1*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].RHO
                   +x0*y1*z1*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].RHO
                   +x1*y0*z1*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].RHO
                   +x0*y0*z1*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].RHO
                   +x1*y1*z0*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].RHO
                   +x0*y1*z0*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].RHO
                   +x1*y0*z0*t1*d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].RHO
                   +x0*y0*z0*t1*d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].RHO
                   +x1*y1*z1*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].RHO
                   +x0*y1*z1*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].RHO
                   +x1*y0*z1*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].RHO
                   +x0*y0*z1*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].RHO
                   +x1*y1*z0*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].RHO
                   +x0*y1*z0*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].RHO
                   +x1*y0*z0*t0*d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].RHO
                   +x0*y0*z0*t0*d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].RHO;

        // Optimize memory access by caching meteorological data points
        FlexPres met_p0[8], met_p1[8];
        
        // Cache meteorological data points with boundary checks
        // Ensure array indices are within bounds to prevent memory access violations
        int safe_xidx = min(xidx, dimX_GFS - 2);    // Ensure xidx+1 is valid
        int safe_yidx = min(yidx, dimY_GFS - 2);    // Ensure yidx+1 is valid
        int safe_zidx = min(zidx, dimZ_GFS - 2);    // Ensure zidx+1 is valid
        
        met_p0[0] = d_pressure_past[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx)];
        met_p0[1] = d_pressure_past[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx)];
        met_p0[2] = d_pressure_past[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx)];
        met_p0[3] = d_pressure_past[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx)];
        met_p0[4] = d_pressure_past[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx+1)];
        met_p0[5] = d_pressure_past[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx+1)];
        met_p0[6] = d_pressure_past[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx+1)];
        met_p0[7] = d_pressure_past[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx+1)];
        
        met_p1[0] = d_pressure_future[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx)];
        met_p1[1] = d_pressure_future[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx)];
        met_p1[2] = d_pressure_future[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx)];
        met_p1[3] = d_pressure_future[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx)];
        met_p1[4] = d_pressure_future[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx+1)];
        met_p1[5] = d_pressure_future[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx) * dimZ_GFS + (safe_zidx+1)];
        met_p1[6] = d_pressure_future[(safe_xidx) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx+1)];
        met_p1[7] = d_pressure_future[(safe_xidx+1) * dimY_GFS * dimZ_GFS + (safe_yidx+1) * dimZ_GFS + (safe_zidx+1)];
        
        // Debug disabled for performance

        float temp = x1*y1*z1*t1*met_p0[0].TT + x0*y1*z1*t1*met_p0[1].TT + x1*y0*z1*t1*met_p0[2].TT + x0*y0*z1*t1*met_p0[3].TT
                    +x1*y1*z0*t1*met_p0[4].TT + x0*y1*z0*t1*met_p0[5].TT + x1*y0*z0*t1*met_p0[6].TT + x0*y0*z0*t1*met_p0[7].TT
                    +x1*y1*z1*t0*met_p1[0].TT + x0*y1*z1*t0*met_p1[1].TT + x1*y0*z1*t0*met_p1[2].TT + x0*y0*z1*t0*met_p1[3].TT
                    +x1*y1*z0*t0*met_p1[4].TT + x0*y1*z0*t0*met_p1[5].TT + x1*y0*z0*t0*met_p1[6].TT + x0*y0*z0*t0*met_p1[7].TT;

        float xwind = 1.0;

        float ywind = 0.0;

        float zwind = 0.0;

        // Debug wind and critical checks (disabled)



        float usig, vsig, wsig, dsw2;
        float uusig = 0, vvsig = 0, wwsig = 0;
        float Tu, Tv, Tw;
        float s1, s2;

        float dx = 0, dy = 0;
        float dxt = 0, dyt = 0;

        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].UU;

        uusig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);

        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].UU;

        uusig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);


        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].VV;

        vvsig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);

        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].VV;

        vvsig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);


        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx)].WW;

        wwsig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);

        s1 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            *d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW;

        s2 = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_past[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW
            +d_pressure_future[(xidx+1) * dimY_GFS * dimZ_GFS + (yidx+1) * dimZ_GFS + (zidx+1)].WW;

        wwsig += 0.5*sqrt((s1-s2*s2/8.0)/7.0);


        // p.radi = 6.0e-1;
        // p.prho = 2500.0;

        float vis = Dynamic_viscosity(temp)/rho;
        float Re = p.radi/1.0e6*fabsf(ks.settling_vel)/vis;
        float settold = ks.settling_vel;
        float settling;
        float c_d;

        if(p.radi > 1.0e-10){
            for(int i=0; i<20; i++){
                if(Re<1.917) c_d = 24.0/Re;
                else if(Re<500.0) c_d = 18.5/pow(Re, 0.6);
                else c_d = 0.44;
    
                settling = -1.0*sqrt(4.0*_ga*p.radi/1.0e6*p.prho*ks.cunningham_fac/(3.0*c_d*rho));

                if(fabsf((settling-settold)/settling)<0.01) break;
    
                Re = p.radi/1.0e6*fabsf(settling)/vis;
                settold = settling;
            }
            zwind += settling;
        }


        p.w_wind = zwind;

        if(zeta <= 1.0) {

            if(hmix/abs(obkl) < 1.0){ // Neutral condition
                if(ustr<1.0e-4) ustr=1.0e-4;
                usig = 2.0*ustr*exp(-3.0e-4*p.z/ustr);
                if(usig<1.0e-5) usig=1.0e-5;
                vsig = 1.3*ustr*exp(-2.0e-4*p.z/ustr);
                if(vsig<1.0e-5) vsig=1.0e-5;
                wsig=vsig;

                dsw2 = -6.76e-4*ustr*exp(-4.0e-4*p.z/ustr);

                Tu=0.5*p.z/wsig/(1.0+1.5e-3*p.z/ustr);
                Tv=Tu;
                Tw=Tu;

            }

            else if(obkl < 0.0){ // Unstable condition
                usig = ustr*pow(12-0.5*hmix/obkl,1.0/3.0);
                if(usig<1.0e-6) usig=1.0e-6;
                vsig = usig;

                
                if(zeta < 0.03){
                    wsig = 0.9600*wstr*pow(3*zeta-obkl/hmix,1.0/3.0);
                    dsw2 = 1.8432*wstr*wstr/hmix*pow(3*zeta-obkl/hmix,-1.0/3.0);
                }
                else if(zeta < 0.40){
                    s1 = 0.9600*pow(3*zeta-obkl/hmix,1.0/3.0);
                    s2 = 0.7630*pow(zeta,0.175);
                    if(s1 < s2){
                        wsig = wstr*s1;
                        dsw2 = 1.8432*wstr*wstr/hmix*pow(3*zeta-obkl/hmix,-1.0/3.0);
                    }
                    else{
                        wsig = wstr*s2;
                        dsw2 = 0.203759*wstr*wstr/hmix*pow(zeta,-0.65);
                    }
                }
                else if(zeta < 0.96){
                    wsig = 0.722*wstr*pow(1-zeta,0.207);
                    dsw2 = -0.215812*wstr*wstr/hmix*pow(1-zeta,-0.586);
                }
                else if(zeta < 1.00){
                    wsig = 0.37*wstr;
                    dsw2 = 0.00;
                }

                if(wsig<1.0e-6) wsig=1.0e-6;

                Tu = 0.15*hmix/usig;
                Tv = Tu;

                if(p.z < abs(obkl)){
                    Tw = 0.1*p.z/(wsig*(0.55-0.38*abs(p.z/obkl)));
                } 
                else if(zeta < 0.1){
                    Tw = 0.59*p.z/wsig;
                }
                else{
                    Tw = 0.15*hmix/wsig*(1.0-exp(-5*zeta));
                }
            }

            else{ // Stable condition

                usig = 2.0*ustr*(1.0-zeta);
                vsig = 1.3*ustr*(1.0-zeta);
                if(usig<1.0e-6) usig=1.0e-6;
                if(vsig<1.0e-6) vsig=1.0e-6;
                wsig = vsig;

                dsw2 = 3.38*ustr*ustr*(zeta-1.0)/hmix;

                Tu = 0.15*hmix/usig*sqrt(zeta);
                Tv = 0.467*Tu;
                Tw = 0.1*hmix/wsig*pow(zeta,0.8);

            }

            if(Tu<10.0) Tu=10.0;
            if(Tv<10.0) Tv=10.0;
            if(Tw<30.0) Tw=30.0;

            float ux, uy, uz, rw;
            
            if(ks.delta_time/Tu < 0.5) p.up = (1.0-ks.delta_time/Tu)*p.up + curand_normal_double(&ss)*usig*sqrt(2.0*ks.delta_time/Tu);
            else p.up = exp(-ks.delta_time/Tu)*p.up + curand_normal_double(&ss)*usig*sqrt(1.0-exp(-ks.delta_time/Tu)*exp(-ks.delta_time/Tu));
                    
            if(ks.delta_time/Tv < 0.5) p.vp = (1.0-ks.delta_time/Tv)*p.vp + curand_normal_double(&ss)*vsig*sqrt(2.0*ks.delta_time/Tv);
            else p.vp = exp(-ks.delta_time/Tv)*p.vp + curand_normal_double(&ss)*vsig*sqrt(1.0-exp(-ks.delta_time/Tv)*exp(-ks.delta_time/Tv));    
        
            if(ks.turb_switch){}
            else{
                rw = exp(-ks.delta_time/Tw);
                float old_wp = p.wp;
                p.wp = (rw*p.wp + curand_normal_double(&ss)*sqrt(1.0-rw*rw)*wsig + Tw*(1.0-rw)*(dsw2+drho/rho*wsig*wsig))*p.dir;
                // Debug wp calculation (disabled)
            }
            

            // Debug disabled for performance
            
            // if (p.wp*ks.delta_time < -p.z){
            //     p.dir = -1;
            //     float old_z = p.z;
            //     p.z = -p.z - p.wp*ks.delta_time;
            //     if (idx == 0) {
            //         static int reflect_debug1 = 0;
            //         if (reflect_debug1 < 3) {
            //             printf("[Z_REFLECT1] Particle 0: old_z=%.6f, wp=%.6f, dt=%.6f, new_z=%.6f (NaN=%d)\n", 
            //                    old_z, p.wp, ks.delta_time, p.z, isnan(p.z));
            //             reflect_debug1++;
            //         }
            //     }
            // }
            // else if (p.wp*ks.delta_time > (hmix-p.z)){
            //     p.dir = -1;
            //     float old_z = p.z;
            //     p.z = -p.z - p.wp*ks.delta_time + 2.*hmix;
            //     if (idx == 0) {
            //         static int reflect_debug2 = 0;
            //         if (reflect_debug2 < 3) {
            //             printf("[Z_REFLECT2] Particle 0: old_z=%.6f, wp=%.6f, dt=%.6f, hmix=%.6f, new_z=%.6f (NaN=%d)\n", 
            //                    old_z, p.wp, ks.delta_time, hmix, p.z, isnan(p.z));
            //             reflect_debug2++;
            //         }
            //     }
            // }
            // else{
            //     p.dir = 1;
            //     p.z = p.z + p.wp*ks.delta_time;
            //     if (idx == 0) {
            //         static int reflect_debug3 = 0;
            //         if (reflect_debug3 < 3) {
            //             printf("[Z_NORMAL] Particle 0: old_z=%.6f, wp=%.6f, dt=%.6f, new_z=%.6f (NaN=%d)\n", 
            //                    p.z - p.wp*ks.delta_time, p.wp, ks.delta_time, p.z, isnan(p.z));
            //             reflect_debug3++;
            //         }
            //     }
            // }



            //p.z += p.wp*ks.delta_time;

            dx += xwind*ks.delta_time;
            dy += ywind*ks.delta_time;
            // dxt += p.up*ks.delta_time;
            // dyt += p.vp*ks.delta_time;
            float old_z_zwind = p.z;
            // p.z += zwind*ks.delta_time;
            
            // Debug first particle z update for NaN tracking (disabled)

        }
        else{

            float ux, uy, uz;

            
            // if(p.z < trop){
            //     ux = 0.0;
            //     uy = 0.0;
            //     uz = 0.0;
            // }
            // else if(p.z < trop+1000.0){
            //     ux = 0.0;
            //     uy = 0.0;
            //     uz = 0.0;
            // }
            // else{
            //     ux = 0.0;
            //     uy = 0.0;
            //     uz = 0.0;
            // }

            ux = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time);
            uy = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time);
            uz = 0.0;

            // if(p.z < trop){
            //     ux = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time);
            //     uy = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time);
            //     uz = 0.0;
            // }
            // else if(p.z < trop+1000.0){
            //     ux = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time*(1-(p.z-trop)/1000.0));
            //     uy = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.0*d_trop/ks.delta_time*(1-(p.z-trop)/1000.0));
            //     uz = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.*d_strat/ks.delta_time*(p.z-trop)/1000.0)+d_strat/1000.0;
            // }
            // else{
            //     ux = 0.0;
            //     uy = 0.0;
            //     uz = GaussianRand(&ss, 0.0f, 1.0f)*sqrt(2.*d_strat/ks.delta_time);
            // }
            

            dx += (xwind+ux)*ks.delta_time;
            dy += (ywind+uy)*ks.delta_time;
            float old_z_strat = p.z;
            p.z += (zwind+uz)*ks.delta_time;
            
            // Debug second particle z update for NaN tracking (disabled)
            
            if(p.z<0.0) {
                float old_z_neg = p.z;
                p.z=-p.z;
                // Debug negative z correction (disabled)
            }

        }

        float r = exp(-2.0*ks.delta_time/static_cast<float>(time_interval));
        float rs = sqrt(1.0-r*r);

        if(p.z<0.0) {
            float old_z_final_neg = p.z;
            p.z=-p.z;
        }

        
        float wind = sqrt(xwind*xwind+ywind*ywind);

        dx += xwind/wind*dxt-ywind/wind*dyt;
        dy += ywind/wind*dxt+xwind/wind*dyt;

        //printf("s1= %f, s2= %f\n", xwind/wind*dxt-ywind/wind*dyt, ywind/wind*dxt+xwind/wind*dyt);

        s1 = 180.0/(0.5*r_earth*PI);// dxconst, dyconst
        s2 = s1/cos((p.y*0.5-90.0)*PI180);// cosfact

        //printf("dx= %f, dy= %f, s1= %f, s2= %f\n", dx, dy, s1, s2);
        //printf("dx= %f, dy= %f\n", dx*s2, dy*s1);
        p.x += dx*s2; //!!
        p.y += dy*s1; //!!
        

        if(p.z > ks.height_levels[dimZ_GFS-1]) {
            float old_z_clamp = p.z;
            p.z = ks.height_levels[dimZ_GFS-1]*0.999999;
        }

        float prob = 0.0;
        float decfact = 1.0;
        float prob_dry = 0.0f;

        if (ks.drydep && p.z < 2.0f * _href) {
            // if (idx == 0 && tstep <= 3) {
            //     printf("[GPU] DRYDEP enabled: z=%.2f, href=%.2f, vdep=%.6f\n", p.z, _href, vdep);
            // }
            float arg = -vdep * ks.delta_time / (2.0f * _href);
            prob_dry = clamp01(1.0f - __expf(arg));
            // if (idx == 0 && tstep <= 3) {
            //     printf("[GPU] DRYDEP calculation: arg=%.6f, exp(arg)=%.6f, prob_dry=%.6f\n", arg, __expf(arg), prob_dry);
            // }
        } 
        //else if (idx == 0 && tstep <= 3) {
        //     printf("[GPU] DRYDEP disabled or z too high: ks.drydep=%d, z=%.2f\n", ks.drydep, p.z);
        // }

        float clouds_v, clouds_h;

        if(t0<=0.5) {
            clouds_v = d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].CLDS;
            clouds_h = d_surface_past[(xidx) * dimY_GFS + (yidx)].CLDH;
        }
        else{
            clouds_v = d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].CLDS;
            clouds_h = d_surface_future[(xidx) * dimY_GFS + (yidx)].CLDH;
        }

            float wet_removal = 0.0f;

            if (ks.wetdep && (lsp >= 0.01f || convp >= 0.01f) && clouds_v > 1.0f) {
                // if (idx == 0 && tstep <= 3) {
                //     printf("[GPU] WETDEP enabled: lsp=%.3f, convp=%.3f, clouds=%.1f\n", lsp, convp, clouds_v);
                // }
                const float lfr[5] = {0.5f, 0.65f, 0.8f, 0.9f, 0.95f};
                const float cfr[5] = {0.4f, 0.55f, 0.7f, 0.8f, 0.9f};

                int weti = (lsp > 20.0f) ? 5 : (lsp > 8.0f) ? 4 : (lsp > 3.0f) ? 3 : (lsp > 1.0f) ? 2 : 1;
                int wetj = (convp > 20.0f) ? 5 : (convp > 8.0f) ? 4 : (convp > 3.0f) ? 3 : (convp > 1.0f) ? 2 : 1;

                float grfraction = 0.05f;
                if (lsp + convp > 0.0f) {
                    grfraction = fmaxf(0.05f, cc * (lsp * lfr[weti - 1] + convp * cfr[wetj - 1]) / (lsp + convp));
                }

                float prec = (lsp + convp) / grfraction;

                // 스캐빈징
                float wetscav = 0.0f;
                const float weta = 9.99999975e-5f;
                const float wetb = 0.800000012f;
                const float henry = p.drydep_vel;

                if (weta > 0.0f) {
                    if (clouds_v >= 4.0f) {
                        wetscav = weta * powf(prec, wetb);
                    } else {
                        float act_temp = (t0 <= 0.5f)
                            ? d_pressure_past[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].TT
                            : d_pressure_future[(xidx) * dimY_GFS * dimZ_GFS + (yidx) * dimZ_GFS + (zidx)].TT;
                        float cl = 2.0e-7f * powf(prec, 0.36f);
                        float S_i = (p.radi > 1.0e-10f)
                            ? (0.9f / cl)
                            : 1.0f / ((1.0f - cl) / (henry * (_rair / 3500.0f) * act_temp) + cl);
                        wetscav = S_i * prec / 3.6e6f / fmaxf(1.0f, clouds_h);
                    }
                }

                wet_removal = clamp01((1.0f - __expf(-wetscav * ks.delta_time)) * grfraction);
            } 
            //else if (idx == 0 && tstep <= 3) {
            //     printf("[GPU] WETDEP disabled or no precipitation: ks.wetdep=%d, lsp=%.3f, convp=%.3f\n", ks.wetdep, lsp, convp);
            // }

            if (ks.raddecay) {
                // if (idx == 0 && tstep <= 5) {  // Only first particle, first few timesteps
                //     printf("[GPU] RADDECAY enabled: applying T matrix\n");
                // }
                //apply_T_once_rowmajor_60(ks.T_matrix, p.concentrations);
                cram_decay_calculation(ks.T_matrix, p.concentrations);
            } 
            // else if (idx == 0 && tstep <= 5) {
            //     printf("[GPU] RADDECAY disabled: skipping T matrix\n");
            // }


            if (ks.wetdep && wet_removal > 0.0f) {
                #pragma unroll
                for (int i = 0; i < N_NUCLIDES; ++i) {
                    float c = p.concentrations[i];
                    if (c > 0.0f) p.concentrations[i] = c * (1.0f - wet_removal);
                }
            }


        if (ks.drydep && prob_dry > 0.0f) {
            // if (idx == 0 && tstep <= 3) {
            //     printf("[GPU] DRYDEP applying: prob_dry=%.6f\n", prob_dry);
            // }
            
            #pragma unroll
            for (int i = 0; i < N_NUCLIDES; ++i) {
                float c = p.concentrations[i];
                if (c > 0.0f) p.concentrations[i] = c * (1.0f - prob_dry);
            }
            
        } 
        //else if (ks.drydep && idx == 0 && tstep <= 3) {
        //     printf("[GPU] DRYDEP not applying: prob_dry=%.6f\n", prob_dry);
        // }


    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_NUCLIDES; ++i) {
        float c = p.concentrations[i];
        c = isfinite(c) ? c : 0.0f;
        c = fminf(c, 1e20f);
        // ALLOW NEGATIVE CONCENTRATIONS for EKI algorithm
        // c = fmaxf(c, 0.0f);  // REMOVED: Don't clamp to zero
        p.concentrations[i] = c;
        total += c;
    }
    // ALLOW NEGATIVE TOTAL for EKI algorithm
    // p.conc = fminf(fmaxf(total, 0.0f), 1e20f);  // REMOVED: Don't clamp to zero
    p.conc = isfinite(total) ? fminf(total, 1e20f) : 0.0f;
    

        // Safety checks for wind components
        p.u_wind = isnan(xwind) ? 0.0f : xwind;
        p.v_wind = isnan(ywind) ? 0.0f : ywind;
        p.w_wind = isnan(zwind) ? 0.0f : zwind;

        // Final debug only for critical check
        if (idx == 0) {
            static int final_debug_count = 0;
            if (final_debug_count < 5) {
// printf("[FINAL] Particle 0: z=%.3f (NaN=%d)\n", p.z, isnan(p.z));
                final_debug_count++;
            }
        }

}

