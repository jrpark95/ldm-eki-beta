/**
 * @file particle_advection_optimized.cu
 * @brief Implementation of optimized particle advection kernels
 * @date 2025-01-17
 * @version 2.0
 */

#include "particle_advection.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace ldm {
namespace kernels {

namespace cg = cooperative_groups;

// ============================================================================
// Time Interpolation Kernel Implementation
// ============================================================================

template<typename Real>
__global__ void interpolateMeteoFields(
    const FlexUnis* __restrict__ unis0,
    const FlexUnis* __restrict__ unis1,
    const FlexPres* __restrict__ pres0,
    const FlexPres* __restrict__ pres1,
    FlexUnis* __restrict__ unis_current,
    FlexPres* __restrict__ pres_current,
    const Real alpha,
    PerformanceMetrics* metrics
) {
    // Start timing
    clock_t start_clock = clock();

    // Grid-stride loop for better occupancy
    const int gx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gy = blockIdx.y * blockDim.y + threadIdx.y;
    const int gz = blockIdx.z * blockDim.z + threadIdx.z;

    const Real alpha_inv = Real(1) - alpha;

    // ========================================================================
    // 2D Fields Interpolation (7 variables)
    // ========================================================================
    if (gx < dimX_GFS && gy < dimY_GFS && gz == 0) {
        const int idx2d = gx * dimY_GFS + gy;

        // Use fused multiply-add for better performance
        unis_current[idx2d].USTR = fmaf(alpha, unis1[idx2d].USTR,
                                        alpha_inv * unis0[idx2d].USTR);
        unis_current[idx2d].WSTR = fmaf(alpha, unis1[idx2d].WSTR,
                                        alpha_inv * unis0[idx2d].WSTR);
        unis_current[idx2d].OBKL = fmaf(alpha, unis1[idx2d].OBKL,
                                        alpha_inv * unis0[idx2d].OBKL);
        unis_current[idx2d].VDEP = fmaf(alpha, unis1[idx2d].VDEP,
                                        alpha_inv * unis0[idx2d].VDEP);
        unis_current[idx2d].LPREC = fmaf(alpha, unis1[idx2d].LPREC,
                                         alpha_inv * unis0[idx2d].LPREC);
        unis_current[idx2d].CPREC = fmaf(alpha, unis1[idx2d].CPREC,
                                         alpha_inv * unis0[idx2d].CPREC);
        unis_current[idx2d].TCC = fmaf(alpha, unis1[idx2d].TCC,
                                       alpha_inv * unis0[idx2d].TCC);
        unis_current[idx2d].HMIX = fmaf(alpha, unis1[idx2d].HMIX,
                                        alpha_inv * unis0[idx2d].HMIX);
        unis_current[idx2d].TROP = fmaf(alpha, unis1[idx2d].TROP,
                                        alpha_inv * unis0[idx2d].TROP);
        unis_current[idx2d].CLDH = fmaf(alpha, unis1[idx2d].CLDH,
                                        alpha_inv * unis0[idx2d].CLDH);
    }

    // ========================================================================
    // 3D Fields Interpolation (6 main variables + auxiliary)
    // ========================================================================
    if (gx < dimX_GFS && gy < dimY_GFS && gz < dimZ_GFS) {
        const int idx3d = gx * dimY_GFS * dimZ_GFS + gy * dimZ_GFS + gz;

        // Wind components
        pres_current[idx3d].UU = fmaf(alpha, pres1[idx3d].UU,
                                      alpha_inv * pres0[idx3d].UU);
        pres_current[idx3d].VV = fmaf(alpha, pres1[idx3d].VV,
                                      alpha_inv * pres0[idx3d].VV);
        pres_current[idx3d].WW = fmaf(alpha, pres1[idx3d].WW,
                                      alpha_inv * pres0[idx3d].WW);

        // Thermodynamic variables
        pres_current[idx3d].TT = fmaf(alpha, pres1[idx3d].TT,
                                      alpha_inv * pres0[idx3d].TT);
        pres_current[idx3d].RHO = fmaf(alpha, pres1[idx3d].RHO,
                                       alpha_inv * pres0[idx3d].RHO);
        pres_current[idx3d].DRHO = fmaf(alpha, pres1[idx3d].DRHO,
                                        alpha_inv * pres0[idx3d].DRHO);

        // Cloud variables
        pres_current[idx3d].CLDS = fmaf(alpha, pres1[idx3d].CLDS,
                                        alpha_inv * pres0[idx3d].CLDS);
    }

    // Record timing
    if (metrics != nullptr && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        clock_t end_clock = clock();
        float elapsed_ms = (end_clock - start_clock) / (float)CLOCKS_PER_SEC * 1000.0f;
        atomicAdd(&metrics->time_interpolation, elapsed_ms);
    }
}

// ============================================================================
// Optimized Particle Advection V1: Time Separation
// ============================================================================

template<int BLOCK_SIZE>
__global__ void advanceParticlesOptimizedV1(
    LDM::LDMpart* __restrict__ particles,
    const FlexUnis* __restrict__ unis_current,
    const FlexPres* __restrict__ pres_current,
    const int total_particles,
    const KernelScalars ks,
    PerformanceMetrics* __restrict__ metrics
) {
    // Thread index
    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total_particles) return;

    // Start total timing
    clock_t start_total = clock();

    // Load particle
    LDM::LDMpart& p = particles[idx];
    if (!p.flag) return;

    // ========================================================================
    // Position Calculation
    // ========================================================================

    // Grid indices
    int xidx = int(p.x);
    int yidx = int(p.y);

    // Boundary check
    if (xidx >= dimX_GFS - 1 || yidx >= dimY_GFS - 1) {
        p.flag = false;
        return;
    }

    // Find vertical level
    int zidx = 0;
    for (int i = 0; i < dimZ_GFS - 1; i++) {
        if (ks.flex_hgt[i+1] > p.z) {
            zidx = i;
            break;
        }
    }

    // ========================================================================
    // OPTIMIZED: 3D Spatial Interpolation Only (No Time!)
    // ========================================================================

    clock_t start_interp = clock();

    // Calculate interpolation weights
    const float x0 = p.x - xidx;
    const float y0 = p.y - yidx;

    float z0 = 0.0f;
    const float height_diff = ks.flex_hgt[zidx+1] - ks.flex_hgt[zidx];
    if (fabsf(height_diff) > 1e-6f) {
        z0 = (p.z - ks.flex_hgt[zidx]) / height_diff;
    }

    const float x1 = 1.0f - x0;
    const float y1 = 1.0f - y0;
    const float z1 = 1.0f - z0;

    // ========================================================================
    // 2D Variable Interpolation (8 -> 4 memory accesses!)
    // ========================================================================

    float ustr = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].USTR
                +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].USTR
                +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].USTR
                +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].USTR;

    float wstr = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].WSTR
                +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].WSTR
                +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].WSTR
                +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].WSTR;

    float obkl = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].OBKL
                +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].OBKL
                +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].OBKL
                +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].OBKL;
    obkl = 1.0f / obkl;

    float vdep = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].VDEP
                +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].VDEP
                +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].VDEP
                +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].VDEP;

    float lsp = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].LPREC
               +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].LPREC
               +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].LPREC
               +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].LPREC;

    float convp = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].CPREC
                 +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].CPREC
                 +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].CPREC
                 +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].CPREC;

    float cc = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].TCC
              +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].TCC
              +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].TCC
              +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].TCC;

    float hmix = x1*y1*unis_current[(xidx) * dimY_GFS + (yidx)].HMIX
                +x0*y1*unis_current[(xidx+1) * dimY_GFS + (yidx)].HMIX
                +x1*y0*unis_current[(xidx) * dimY_GFS + (yidx+1)].HMIX
                +x0*y0*unis_current[(xidx+1) * dimY_GFS + (yidx+1)].HMIX;

    // ========================================================================
    // 3D Variable Interpolation (16 -> 8 memory accesses!)
    // ========================================================================

    #define IDX3D(x,y,z) ((x) * dimY_GFS * dimZ_GFS + (y) * dimZ_GFS + (z))

    float drho = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].DRHO
                +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].DRHO
                +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].DRHO
                +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].DRHO
                +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].DRHO
                +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].DRHO
                +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].DRHO
                +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].DRHO;

    float rho = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].RHO
               +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].RHO
               +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].RHO
               +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].RHO
               +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].RHO
               +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].RHO
               +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].RHO
               +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].RHO;

    float temp = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].TT
                +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].TT
                +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].TT
                +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].TT
                +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].TT
                +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].TT
                +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].TT
                +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].TT;

    float xwind = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].UU
                 +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].UU
                 +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].UU
                 +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].UU
                 +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].UU
                 +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].UU
                 +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].UU
                 +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].UU;

    float ywind = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].VV
                 +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].VV
                 +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].VV
                 +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].VV
                 +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].VV
                 +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].VV
                 +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].VV
                 +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].VV;

    float zwind = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].WW
                 +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].WW
                 +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].WW
                 +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].WW
                 +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].WW
                 +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].WW
                 +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].WW
                 +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].WW;

    float clouds_v = x1*y1*z1*pres_current[IDX3D(xidx, yidx, zidx)].CLDS
                    +x0*y1*z1*pres_current[IDX3D(xidx+1, yidx, zidx)].CLDS
                    +x1*y0*z1*pres_current[IDX3D(xidx, yidx+1, zidx)].CLDS
                    +x0*y0*z1*pres_current[IDX3D(xidx+1, yidx+1, zidx)].CLDS
                    +x1*y1*z0*pres_current[IDX3D(xidx, yidx, zidx+1)].CLDS
                    +x0*y1*z0*pres_current[IDX3D(xidx+1, yidx, zidx+1)].CLDS
                    +x1*y0*z0*pres_current[IDX3D(xidx, yidx+1, zidx+1)].CLDS
                    +x0*y0*z0*pres_current[IDX3D(xidx+1, yidx+1, zidx+1)].CLDS;

    #undef IDX3D

    clock_t end_interp = clock();

    // ========================================================================
    // Physics Calculations (Same as original)
    // ========================================================================

    clock_t start_physics = clock();

    // Initialize random state
    curandState rng_state;
    unsigned long long seed = idx + blockIdx.x * total_particles;
    curand_init(seed, idx, 0, &rng_state);

    // Turbulence calculation
    float zeta = p.z / hmix;
    float usig = 0.0f, vsig = 0.0f, wsig = 0.0f;
    float dsw2 = 0.0f;
    float Tu = 10.0f, Tv = 10.0f, Tw = 30.0f;

    if (zeta <= 1.0f) {
        if (hmix / fabsf(obkl) < 1.0f) {
            // Neutral condition
            if (ustr < 1.0e-4f) ustr = 1.0e-4f;
            usig = 2.0f * ustr * expf(-3.0e-4f * p.z / ustr);
            vsig = 1.3f * ustr * expf(-2.0e-4f * p.z / ustr);
            wsig = vsig;
            dsw2 = -6.76e-4f * ustr * expf(-4.0e-4f * p.z / ustr);
            Tu = 0.5f * p.z / wsig / (1.0f + 1.5e-3f * p.z / ustr);
            Tv = Tu;
            Tw = Tu;
        } else if (obkl < 0.0f) {
            // Unstable condition
            usig = ustr * powf(12.0f - 0.5f * hmix / obkl, 1.0f/3.0f);
            vsig = usig;

            if (zeta < 0.03f) {
                wsig = 0.96f * wstr * powf(3.0f * zeta - obkl / hmix, 1.0f/3.0f);
                dsw2 = 1.8432f * wstr * wstr / hmix * powf(3.0f * zeta - obkl / hmix, -1.0f/3.0f);
            } else if (zeta < 0.40f) {
                float s1 = 0.96f * powf(3.0f * zeta - obkl / hmix, 1.0f/3.0f);
                float s2 = 0.763f * powf(zeta, 0.175f);
                if (s1 < s2) {
                    wsig = wstr * s1;
                    dsw2 = 1.8432f * wstr * wstr / hmix * powf(3.0f * zeta - obkl / hmix, -1.0f/3.0f);
                } else {
                    wsig = wstr * s2;
                    dsw2 = 0.203759f * wstr * wstr / hmix * powf(zeta, -0.65f);
                }
            } else if (zeta < 0.96f) {
                wsig = 0.722f * wstr * powf(1.0f - zeta, 0.207f);
                dsw2 = -0.215812f * wstr * wstr / hmix * powf(1.0f - zeta, -0.586f);
            } else {
                wsig = 0.37f * wstr;
                dsw2 = 0.0f;
            }

            Tu = 0.15f * hmix / usig;
            Tv = Tu;
            if (p.z < fabsf(obkl)) {
                Tw = 0.1f * p.z / (wsig * (0.55f - 0.38f * fabsf(p.z / obkl)));
            } else if (zeta < 0.1f) {
                Tw = 0.59f * p.z / wsig;
            } else {
                Tw = 0.15f * hmix / wsig * (1.0f - expf(-5.0f * zeta));
            }
        } else {
            // Stable condition
            usig = 2.0f * ustr * (1.0f - zeta);
            vsig = 1.3f * ustr * (1.0f - zeta);
            wsig = vsig;
            dsw2 = 3.38f * ustr * ustr * (zeta - 1.0f) / hmix;
            Tu = 0.15f * hmix / usig * sqrtf(zeta);
            Tv = 0.467f * Tu;
            Tw = 0.1f * hmix / wsig * powf(zeta, 0.8f);
        }

        // Apply minimum values
        if (usig < 1.0e-6f) usig = 1.0e-6f;
        if (vsig < 1.0e-6f) vsig = 1.0e-6f;
        if (wsig < 1.0e-6f) wsig = 1.0e-6f;
        if (Tu < 10.0f) Tu = 10.0f;
        if (Tv < 10.0f) Tv = 10.0f;
        if (Tw < 30.0f) Tw = 30.0f;

        // Turbulent velocity updates
        if (ks.delta_time / Tu < 0.5f) {
            p.up = (1.0f - ks.delta_time / Tu) * p.up +
                   curand_normal(&rng_state) * usig * sqrtf(2.0f * ks.delta_time / Tu);
        } else {
            p.up = expf(-ks.delta_time / Tu) * p.up +
                   curand_normal(&rng_state) * usig * sqrtf(1.0f - expf(-2.0f * ks.delta_time / Tu));
        }

        if (ks.delta_time / Tv < 0.5f) {
            p.vp = (1.0f - ks.delta_time / Tv) * p.vp +
                   curand_normal(&rng_state) * vsig * sqrtf(2.0f * ks.delta_time / Tv);
        } else {
            p.vp = expf(-ks.delta_time / Tv) * p.vp +
                   curand_normal(&rng_state) * vsig * sqrtf(1.0f - expf(-2.0f * ks.delta_time / Tv));
        }

        // Vertical turbulence with reflection
        if (!ks.turb_switch) {
            float rw = expf(-ks.delta_time / Tw);
            p.wp = (rw * p.wp + curand_normal(&rng_state) * sqrtf(1.0f - rw * rw) * wsig +
                    Tw * (1.0f - rw) * (dsw2 + drho / rho * wsig * wsig)) * p.dir;
        }

        // Boundary reflection
        if (p.wp * ks.delta_time < -p.z) {
            p.dir = -1;
            p.z = -p.z - p.wp * ks.delta_time;
        } else if (p.wp * ks.delta_time > (hmix - p.z)) {
            p.dir = -1;
            p.z = -p.z - p.wp * ks.delta_time + 2.0f * hmix;
        } else {
            p.dir = 1;
            p.z = p.z + p.wp * ks.delta_time;
        }
    }

    // Settling velocity
    if (p.radi > 1.0e-10f) {
        float vis = Dynamic_viscosity(temp) / rho;
        float Re = p.radi / 1.0e6f * fabsf(ks.settling_vel) / vis;
        float settling = ks.settling_vel;
        float c_d;

        for (int i = 0; i < 20; i++) {
            if (Re < 1.917f) c_d = 24.0f / Re;
            else if (Re < 500.0f) c_d = 18.5f / powf(Re, 0.6f);
            else c_d = 0.44f;

            float settling_new = -sqrtf(4.0f * _ga * p.radi / 1.0e6f * p.prho *
                                        ks.cunningham_fac / (3.0f * c_d * rho));

            if (fabsf((settling_new - settling) / settling_new) < 0.01f) break;

            Re = p.radi / 1.0e6f * fabsf(settling_new) / vis;
            settling = settling_new;
        }
        zwind += settling;
    }

    // Update particle position
    float dx = xwind * ks.delta_time;
    float dy = ywind * ks.delta_time;
    float dxt = p.up * ks.delta_time;
    float dyt = p.vp * ks.delta_time;

    // Coordinate transformation
    float wind_mag = sqrtf(xwind * xwind + ywind * ywind);
    if (wind_mag > 0.0f) {
        dx += xwind / wind_mag * dxt - ywind / wind_mag * dyt;
        dy += ywind / wind_mag * dxt + xwind / wind_mag * dyt;
    }

    float s1 = 180.0f / (0.5f * r_earth * PI);
    float s2 = s1 / cosf((p.y * 0.5f - 90.0f) * PI180);

    p.x += dx * s2;
    p.y += dy * s1;
    p.z += zwind * ks.delta_time;

    // Height clamping
    if (p.z < 0.0f) p.z = -p.z;
    if (p.z > ks.flex_hgt[dimZ_GFS-1]) {
        p.z = ks.flex_hgt[dimZ_GFS-1] * 0.999999f;
    }

    // Dry deposition
    float prob_dry = 0.0f;
    if (ks.drydep && p.z < 2.0f * _href) {
        float arg = -vdep * ks.delta_time / (2.0f * _href);
        prob_dry = fmaxf(0.0f, fminf(1.0f, 1.0f - expf(arg)));
    }

    // Wet deposition
    float wet_removal = 0.0f;
    if (ks.wetdep && (lsp >= 0.01f || convp >= 0.01f) && clouds_v > 1.0f) {
        // Simplified wet scavenging calculation
        const float weta = 9.99999975e-5f;
        const float wetb = 0.800000012f;
        float prec = lsp + convp;
        float wetscav = weta * powf(prec, wetb);
        wet_removal = fminf(1.0f, (1.0f - expf(-wetscav * ks.delta_time)) * cc);
    }

    // Radioactive decay
    if (ks.raddecay) {
        cram_decay_calculation(ks.T_matrix, p.concentrations);
    }

    // Apply deposition and decay
    #pragma unroll
    for (int i = 0; i < N_NUCLIDES; i++) {
        float c = p.concentrations[i];
        if (ks.wetdep && wet_removal > 0.0f) {
            c *= (1.0f - wet_removal);
        }
        if (ks.drydep && prob_dry > 0.0f) {
            c *= (1.0f - prob_dry);
        }
        p.concentrations[i] = c;
    }

    // Update total concentration
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < N_NUCLIDES; i++) {
        total += p.concentrations[i];
    }
    p.conc = total;

    // Store wind components
    p.u_wind = isnan(xwind) ? 0.0f : xwind;
    p.v_wind = isnan(ywind) ? 0.0f : ywind;
    p.w_wind = isnan(zwind) ? 0.0f : zwind;

    clock_t end_physics = clock();
    clock_t end_total = clock();

    // Record metrics
    if (metrics != nullptr && idx == 0) {
        float interp_ms = (end_interp - start_interp) / (float)CLOCKS_PER_SEC * 1000.0f;
        float physics_ms = (end_physics - start_physics) / (float)CLOCKS_PER_SEC * 1000.0f;
        float total_ms = (end_total - start_total) / (float)CLOCKS_PER_SEC * 1000.0f;

        atomicAdd(&metrics->time_interpolation, interp_ms);
        atomicAdd(&metrics->time_physics, physics_ms);
        atomicAdd(&metrics->time_total, total_ms);
    }
}

// ============================================================================
// Explicit Template Instantiation
// ============================================================================

template __global__ void interpolateMeteoFields<float>(
    const FlexUnis*, const FlexUnis*, const FlexPres*, const FlexPres*,
    FlexUnis*, FlexPres*, const float, PerformanceMetrics*);

template __global__ void advanceParticlesOptimizedV1<256>(
    LDM::LDMpart*, const FlexUnis*, const FlexPres*,
    const int, const KernelScalars, PerformanceMetrics*);

} // namespace kernels
} // namespace ldm