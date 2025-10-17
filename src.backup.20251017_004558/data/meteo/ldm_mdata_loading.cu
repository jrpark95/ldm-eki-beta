/**
 * @file ldm_mdata_loading.cu
 * @brief Meteorological data loading implementation
 */

#include "../../core/ldm.cuh"
#include "ldm_mdata_loading.cuh"
#include "colors.h"


void LDM::initializeFlexGFSData(){

#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Starting meteorological data initialization\n";
#endif

    flex_hgt.resize(dimZ_GFS);

    const char* filename = "../gfsdata/0p5/2.txt";
    int recordMarker;

    size_t pres_data_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_data_size = (dimX_GFS + 1) * dimY_GFS;
#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Allocating memory: FlexPres=" << pres_data_size
              << ", FlexUnis=" << unis_data_size << " elements\n";
#endif

    FlexPres* flexpresdata = new FlexPres[pres_data_size];
    FlexUnis* flexunisdata = new FlexUnis[unis_data_size];

    if (!flexpresdata || !flexunisdata) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to allocate memory for meteorological data\n";
        if (flexpresdata) delete[] flexpresdata;
        if (flexunisdata) delete[] flexunisdata;
        return;
    }

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Cannot open file: " << Color::BOLD << filename << Color::RESET << "\n";
        delete[] flexpresdata;
        delete[] flexunisdata;
        return;
    }
#ifdef DEBUG
    std::cout << Color::GREEN << "✓ " << Color::RESET
              << "Opened file: " << filename << "\n";
#endif

    // Read HMIX data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // Sample debug output for verification
            if ((i == 100 && j == 50) || (i == 200 && j == 150) || (i == 360 && j == 180)) {
                std::cout << "[VERIFY] HMIX[" << i << "," << j << "] = " << flexunisdata[index].HMIX << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            //std::cout << "OBKL = " << flexunisdata[index].OBKL << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0f)std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "CPREC = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) 
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
            //if(i<10&&j<10) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }



    // for (int k = 0; k < 1; ++k) {
    //     for (int i = 0; i < dimX_GFS+1; ++i) {
    //         for (int j = 0; j < dimY_GFS; ++j) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }

    // Debug height data before DRHO calculation
    std::cout << "[DRHO_DEBUG] Height check: flex_hgt[0]=" << flex_hgt[0] << ", flex_hgt[1]=" << flex_hgt[1] 
              << ", diff=" << (flex_hgt[1] - flex_hgt[0]) << std::endl;
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            float rho0 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO;
            float rho1 = flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO;
            float hgt_diff = flex_hgt[1] - flex_hgt[0];
            
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = (rho1 - rho0) / hgt_diff;
            
            // Debug first few points
            if (i < 2 && j < 2) {
                std::cout << "[DRHO_CALC] [" << i << "," << j << "]: rho0=" << rho0 << ", rho1=" << rho1 
                          << ", hgt_diff=" << hgt_diff << ", DRHO=" << flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO << std::endl;
            }

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }
    
#ifdef DEBUG
    std::cout << Color::YELLOW << "[DEBUG] " << Color::RESET
              << "Wind data sample: UU[0,0,0]=" << flexpresdata[0].UU
              << ", VV[0,0,0]=" << flexpresdata[0].VV
              << ", WW[0,0,0]=" << flexpresdata[0].WW << "\n";
#endif

    // Check file read status
    if (file.fail() || file.bad()) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "File read error detected\n";
    }
#ifdef DEBUG
    else {
        std::cout << Color::GREEN << "✓ " << Color::RESET
                  << "File read completed successfully\n";
    }
#endif



    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres0, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis0, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaError_t copy_err = cudaMemcpy(device_meteorological_flex_pres0, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to copy pres data to GPU: " << cudaGetErrorString(copy_err) << "\n";
    }
#ifdef DEBUG
    else {
        std::cout << Color::GREEN << "✓ " << Color::RESET << "Pres data copied to GPU\n";
    }
#endif
    
    copy_err = cudaMemcpy(device_meteorological_flex_unis0, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);
    if (copy_err != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to copy unis data to GPU: " << cudaGetErrorString(copy_err) << "\n";
    }
#ifdef DEBUG
    else {
        std::cout << Color::GREEN << "✓ " << Color::RESET << "Unis data copied to GPU\n";
    }
#endif


    // int size2D = (dimX_GFS + 1) * dimY_GFS;
    // int size3D = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;

    // this->host_unisA0 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB0 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA0 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB0 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA0[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB0[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA0[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB0[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // size_t width  = dimX_GFS + 1; 
    // size_t height = dimY_GFS;

    // cudaMallocArray(&d_unisArrayA0, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB0, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA0,
    //     0, 0,
    //     host_unisA0,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB0,
    //     0, 0,
    //     host_unisB0,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaExtent extent = make_cudaExtent(dimX_GFS+1, dimY_GFS, dimZ_GFS);
    // cudaMalloc3DArray(&d_presArrayA0, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB0, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA0 = {0};
    // copyParamsA0.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA0, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA0.dstArray = d_presArrayA0;
    // copyParamsA0.extent   = extent;
    // copyParamsA0.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA0);

    // cudaMemcpy3DParms copyParamsB0 = {0};
    // copyParamsB0.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB0, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB0.dstArray = d_presArrayB0;
    // copyParamsB0.extent   = extent;
    // copyParamsB0.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB0);

    
    filename = "../gfsdata/0p5/3.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //if(flexunisdata[index].LPREC>0.0)printf("lsp = %e\n", flexunisdata[index].LPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //if(flexunisdata[index].CPREC>0.0)printf("convprec = %e\n", flexunisdata[index].CPREC);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                // file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    // for (int j = 0; j < dimY_GFS; ++j) {
    //     int index = 647 * dimY_GFS + j; 
    //     std::cout << "CLDH(" << 647 << ", " << j << ") = " << flexunisdata[index].CLDH << std::endl;
    //     index = 647 * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 5; 
    //     std::cout << "CLDS(" << 647 << ", " << j << ") = " << flexpresdata[index].CLDS << std::endl;
    // }


    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres1, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis1, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);

    
    // this->host_unisA1 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB1 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA1 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB1 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA1[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB1[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA1[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB1[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // cudaMallocArray(&d_unisArrayA1, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB1, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA1,
    //     0, 0,
    //     host_unisA1,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB1,
    //     0, 0,
    //     host_unisB1,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaMalloc3DArray(&d_presArrayA1, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB1, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA1 = {0};
    // copyParamsA1.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA1, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA1.dstArray = d_presArrayA1;
    // copyParamsA1.extent   = extent;
    // copyParamsA1.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA1);

    // cudaMemcpy3DParms copyParamsB1 = {0};
    // copyParamsB1.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB1, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB1.dstArray = d_presArrayB1;
    // copyParamsB1.extent   = extent;
    // copyParamsB1.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB1);


    
    
    filename = "../gfsdata/0p5/4.txt";

    file.open(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.UU[" << k << "] = " << flexpresdata[index].UU << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.VV[" << k << "] = " << flexpresdata[index].VV << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    // for (int i = 0; i < dimX_GFS+1; ++i) {
    //     for (int j = 0; j < dimY_GFS; ++j) {
    //         for (int k = 0; k < dimZ_GFS; ++k) {
    //             int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             file.read(reinterpret_cast<char*>(&flexpresdata[index].VDEP), sizeof(float));
    //             file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    //             if(i==50&&j==100) std::cout << "flexpresdata.VDEP[" << k << "] = " << flexpresdata[index].VDEP << std::endl;
    //         }
    //     }
    // }


    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();

    cudaMalloc((void**)&device_meteorological_flex_pres2, 
    (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres));
    cudaMalloc((void**)&device_meteorological_flex_unis2, 
    (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis));
    
    cudaMemcpy(device_meteorological_flex_pres2, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis2, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    // this->host_unisA2 = new float4[size2D]; // HMIX, USTR, WSTR, OBKL
    // this->host_unisB2 = new float4[size2D]; // VDEP, LPREC, CPREC, TCC
    // this->host_presA2 = new float4[size3D]; // DRHO, RHO, TT, QV
    // this->host_presB2 = new float4[size3D]; // UU, VV, WW, 0.0 

    // for(int i = 0; i < size2D; i++){
    //     host_unisA2[i] = make_float4(
    //         flexunisdata[i].HMIX,
    //         flexunisdata[i].USTR,
    //         flexunisdata[i].WSTR,
    //         flexunisdata[i].OBKL
    //     );
    //     host_unisB2[i] = make_float4(
    //         flexunisdata[i].VDEP,
    //         flexunisdata[i].LPREC,
    //         flexunisdata[i].CPREC,
    //         flexunisdata[i].TCC
    //     );
    // }

    // for(int i = 0; i < size3D; i++){
    //     host_presA2[i] = make_float4(
    //         flexpresdata[i].DRHO,
    //         flexpresdata[i].RHO,
    //         flexpresdata[i].TT,
    //         flexpresdata[i].QV
    //     );
    //     host_presB2[i] = make_float4(
    //         flexpresdata[i].UU,
    //         flexpresdata[i].VV,
    //         flexpresdata[i].WW,
    //         0.0f        
    //     );
    // }

    // cudaMallocArray(&d_unisArrayA2, &channelDesc2D, width, dimY_GFS);
    // cudaMallocArray(&d_unisArrayB2, &channelDesc2D, dimX_GFS + 1, dimY_GFS);

    // cudaMemcpy2DToArray(
    //     d_unisArrayA2,
    //     0, 0,
    //     host_unisA2,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );
    
    // cudaMemcpy2DToArray(
    //     d_unisArrayB2,
    //     0, 0,
    //     host_unisB2,
    //     width*sizeof(float4),
    //     width*sizeof(float4),
    //     height,
    //     cudaMemcpyHostToDevice
    // );

    // cudaMalloc3DArray(&d_presArrayA2, &channelDesc3D, extent);
    // cudaMalloc3DArray(&d_presArrayB2, &channelDesc3D, extent);

    // cudaMemcpy3DParms copyParamsA2 = {0};
    // copyParamsA2.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presA2, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsA2.dstArray = d_presArrayA2;
    // copyParamsA2.extent   = extent;
    // copyParamsA2.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsA2);

    // cudaMemcpy3DParms copyParamsB2 = {0};
    // copyParamsB2.srcPtr = make_cudaPitchedPtr(
    //     (void*)host_presB2, 
    //     (dimX_GFS+1)*sizeof(float4), // pitch in bytes
    //     (dimX_GFS+1),                // width in elements
    //     dimY_GFS                     // height
    // );
    // copyParamsB2.dstArray = d_presArrayB2;
    // copyParamsB2.extent   = extent;
    // copyParamsB2.kind     = cudaMemcpyHostToDevice;
    // cudaMemcpy3D(&copyParamsB2);

        
    delete[] flexunisdata;
    delete[] flexpresdata;

}

void LDM::loadFlexGFSData(){

    gfs_idx ++;

    char filename[256];
    sprintf(filename, "../gfsdata/0p5/%d.txt", gfs_idx+3);
    std::cout << "[DEBUG] Loading next gfs file: " << filename << std::endl;
    int recordMarker;

    // FlexUnis flexunisdata;
    // FlexPres flexpresdata;

    FlexPres* flexpresdata = new FlexPres[(dimX_GFS + 1) * dimY_GFS * dimZ_GFS];
    FlexUnis* flexunisdata = new FlexUnis[(dimX_GFS + 1) * dimY_GFS];

    cudaMemcpy(device_meteorological_flex_pres0, 
        device_meteorological_flex_pres1, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyDeviceToDevice);
    cudaMemcpy(device_meteorological_flex_unis0, 
        device_meteorological_flex_unis1, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyDeviceToDevice);

    size_t width  = dimX_GFS + 1; 
    size_t height = dimY_GFS;

    // flexunisdata.HMIX.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.TROP.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.USTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.WSTR.resize((dimX_GFS + 1) * dimY_GFS);
    // flexunisdata.OBKL.resize((dimX_GFS + 1) * dimY_GFS);

    // flexpresdata.RHO.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.TT.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.UU.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.VV.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);
    // flexpresdata.WW.resize((dimX_GFS + 1) * dimY_GFS * dimZ_GFS);

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Cannot open file: " << Color::BOLD << filename << Color::RESET << "\n";
        return;
    }
    else{
        //std::cout << "open file: " << filename << std::endl;
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            // if(i==50) std::cout << "flexunisdata.USTR[" << j << "] = " << flexunisdata.USTR[index] << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].LPREC), sizeof(float));
            //std::cout << "LPREC = " << flexunisdata[index].LPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].CPREC), sizeof(float));
            //std::cout << "cprec = " << flexunisdata[index].CPREC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].TCC), sizeof(float));
            //std::cout << "TCC = " << flexunisdata[index].TCC << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            flexunisdata[index].CLDH = static_cast<float>(intBuffer);
            //file.read(reinterpret_cast<char*>(&flexunisdata[index].CLDH), sizeof(int));
            //std::cout << "CLDH = " << flexunisdata[index].CLDH << std::endl;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.RHO[" << k << "] = " << flexpresdata[index].RHO << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                //if(i==50&&j==100) std::cout << "flexpresdata.TT[" << k << "] = " << flexpresdata[index].TT << std::endl;
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&flexpresdata[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // Debug disabled
            }
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j; 
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&flexunisdata[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            //if(i==50&&j==100) std::cout << "VDEP[" << i << "," << j << "] = " << flexunisdata[index].VDEP << std::endl;
        }
    }

    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k; 
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                flexpresdata[index].CLDS = static_cast<float>(intBuffer);
                //file.read(reinterpret_cast<char*>(&flexpresdata[index].CLDS), sizeof(int));
                //std::cout << "CLDS = " << static_cast<int>(flexpresdata[index].CLDS) << std::endl;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                // if(i==50&&j==100) std::cout << "flexpresdata.WW[" << k << "] = " << flexpresdata[index].WW << std::endl;
            }
        }
    }



    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = 
            (flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO - flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO) /
            (flex_hgt[1]-flex_hgt[0]);

            //if(i==100 && j==50) printf("drho(%d) = %f\n", 0, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO);
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                flexpresdata[index].DRHO = 
                (flexpresdata[index+1].RHO - flexpresdata[index-1].RHO) / (flex_hgt[k+1]-flex_hgt[k-1]);
                //if(i==100 && j==50) printf("drho(%d) = %f\n", k, flexpresdata[index].DRHO);
            }

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;

            //if(i==100 && j==50) printf("drho(%d) = %f\n", dimZ_GFS-2, flexpresdata[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO);

        }
    }

    file.close();
    
    cudaMemcpy(device_meteorological_flex_pres1, 
        flexpresdata, (dimX_GFS+1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_flex_unis1, 
        flexunisdata, (dimX_GFS+1) * dimY_GFS * sizeof(FlexUnis), 
        cudaMemcpyHostToDevice);


    char filename_hgt[256];
    sprintf(filename_hgt, "../gfsdata/0p5/hgt_%d.txt", gfs_idx+2);
    //std::cout << filename_hgt << std::endl;

    int recordMarker_hgt;

    std::ifstream file_hgt(filename_hgt, std::ios::binary);
    if (!file_hgt) {
        std::cerr << "Cannot open file: " << filename_hgt << std::endl;
        return;
    }

    for (int index = 0; index < dimZ_GFS; ++index) {
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        file_hgt.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        file_hgt.read(reinterpret_cast<char*>(&recordMarker_hgt), sizeof(int));
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file_hgt.close();

    // Allocate GPU memory for height data if not already allocated
    if (d_flex_hgt == nullptr) {
        cudaError_t alloc_err = cudaMalloc(&d_flex_hgt, sizeof(float) * dimZ_GFS);
        if (alloc_err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for flex_hgt: " << cudaGetErrorString(alloc_err) << std::endl;
            return;
        }
    }

    // Copy height data to GPU memory
    cudaError_t err = cudaMemcpy(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy flex_hgt to GPU memory: " << cudaGetErrorString(err) << std::endl;
    }


}

void LDM::loadFlexHeightData(){ 

    std::cout << "[DEBUG] Starting read_meteorological_flex_hgt function..." << std::endl;
    
    flex_hgt.resize(dimZ_GFS);
    std::cout << "[DEBUG] flex_hgt vector resized to " << dimZ_GFS << " elements" << std::endl;

    const char* filename = "../gfsdata/0p5/hgt_2.txt";
    int recordMarker;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "[ERROR] Cannot open file: " << filename << std::endl;
        return;
    }
    std::cout << "[DEBUG] Successfully opened file: " << filename << std::endl;

    for (int index = 0; index < dimZ_GFS; ++index) {
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read record marker at index " << index << std::endl;
            return;
        }
        
        file.read(reinterpret_cast<char*>(&flex_hgt[index]), sizeof(float));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read flex_hgt data at index " << index << std::endl;
            return;
        }
        
        // Check for NaN in height data
        if (std::isnan(flex_hgt[index])) {
            std::cout << "[HGT_NAN] flex_hgt[" << index << "] is NaN!" << std::endl;
        }
        
        file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        if (file.fail()) {
            std::cerr << "[ERROR] Failed to read closing record marker at index " << index << std::endl;
            return;
        }
        
        // std::cout << "flex_hgt[" << index << "] = " << flex_hgt[index] << std::endl;
    }

    file.close();
    std::cout << "[DEBUG] Successfully read " << dimZ_GFS << " height levels from file" << std::endl;

    // Allocate GPU memory for height data if not already allocated
    if (d_flex_hgt == nullptr) {
        std::cout << "[DEBUG] Allocating GPU memory for " << dimZ_GFS << " height levels" << std::endl;
        cudaError_t alloc_err = cudaMalloc(&d_flex_hgt, sizeof(float) * dimZ_GFS);
        if (alloc_err != cudaSuccess) {
            std::cerr << "[ERROR] Failed to allocate GPU memory for flex_hgt: " << cudaGetErrorString(alloc_err) << std::endl;
            return;
        }
    }

    // Copy height data to GPU memory
    std::cout << "[DEBUG] Copying " << dimZ_GFS << " float values to d_flex_hgt GPU memory" << std::endl;
    cudaError_t err = cudaMemcpy(d_flex_hgt, flex_hgt.data(), sizeof(float) * dimZ_GFS, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] Failed to copy flex_hgt to GPU memory: " << cudaGetErrorString(err) << std::endl;
        std::cerr << "[ERROR] Size attempted: " << sizeof(float) * dimZ_GFS << " bytes (" << dimZ_GFS << " floats)" << std::endl;
    } else {
        std::cout << "[DEBUG] Successfully copied flex_hgt to GPU memory" << std::endl;
    }


}

