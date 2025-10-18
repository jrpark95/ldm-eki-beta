/**
 * @file ldm_mdata_cache.cu
 * @brief Meteorological data cache implementation
 */

#include "../../core/ldm.cuh"
#include "ldm_mdata_cache.cuh"
#include "colors.h"

int LDM::calculateRequiredMeteoFiles() {
    // Calculate required number of files from total simulation time and meteorological data interval in settings.txt
    float total_simulation_time = time_end;  // Total simulation time (seconds)
    int meteo_time_interval = Constants::time_interval;  // Meteorological data time interval (seconds)

    // Number of files needed = (total simulation time / meteorological interval) + 1 (including file 0)
    int num_files = static_cast<int>(std::ceil(total_simulation_time / meteo_time_interval)) + 1;

    std::cout << "Total simulation time: " << total_simulation_time << " seconds" << std::endl;
    std::cout << "Meteorological data time interval: " << meteo_time_interval << " seconds" << std::endl;
    std::cout << "Required meteorological data files: " << num_files << " files (0~" << (num_files-1) << ")" << std::endl;
    
    return num_files;
}

bool LDM::loadSingleMeteoFile(int file_index, FlexPres*& pres_data, FlexUnis*& unis_data, std::vector<float>& hgt_data) {
    // Generate file paths
    char filename[256];
    sprintf(filename, "../gfsdata/0p5/%d.txt", file_index);
    
    char hgt_filename[256];
    sprintf(hgt_filename, "../gfsdata/0p5/hgt_%d.txt", file_index);
    
    
    // Memory allocation
    size_t pres_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS;
    size_t unis_size = (dimX_GFS + 1) * dimY_GFS;
    
    pres_data = new FlexPres[pres_size];
    unis_data = new FlexUnis[unis_size];
    hgt_data.resize(dimZ_GFS);
    
    if (!pres_data || !unis_data) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Memory allocation failed: " << filename << std::endl;
        if (pres_data) delete[] pres_data;
        if (unis_data) delete[] unis_data;
        return false;
    }
    
    // Read meteorological data file
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to open file: " << filename << std::endl;
        delete[] pres_data;
        delete[] unis_data;
        return false;
    }
    
    int recordMarker;
    
    // Read HMIX data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].HMIX), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read TROP data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].TROP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read USTR data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].USTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read WSTR data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].WSTR), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read OBKL data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].OBKL), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read LPREC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].LPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CPREC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].CPREC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read TCC data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].TCC), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CLDH data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            int intBuffer;
            file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
            unis_data[index].CLDH = static_cast<float>(intBuffer);
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read 3D pressure data (RHO, TT, UU, VV, WW)
    // RHO
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].RHO), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // TT
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].TT), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // UU
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].UU), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // VV
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].VV), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // WW
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                file.read(reinterpret_cast<char*>(&pres_data[index].WW), sizeof(float));
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    // Read VDEP data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            int index = i * dimY_GFS + j;
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            file.read(reinterpret_cast<char*>(&unis_data[index].VDEP), sizeof(float));
            file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        }
    }
    
    // Read CLDS data
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            for (int k = 0; k < dimZ_GFS; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
                int intBuffer;
                file.read(reinterpret_cast<char*>(&intBuffer), sizeof(int));
                pres_data[index].CLDS = static_cast<float>(intBuffer);
                file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
            }
        }
    }
    
    file.close();
    
    // Read height data
    std::ifstream hgt_file(hgt_filename, std::ios::binary);
    if (!hgt_file) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to open height file: " << hgt_filename << std::endl;
        delete[] pres_data;
        delete[] unis_data;
        return false;
    }
    
    std::cout << "[DEBUG_HGT] Loading: " << hgt_filename << std::endl;
    
    for (int index = 0; index < dimZ_GFS; ++index) {
        hgt_file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
        hgt_file.read(reinterpret_cast<char*>(&hgt_data[index]), sizeof(float));
        hgt_file.read(reinterpret_cast<char*>(&recordMarker), sizeof(int));
    }
    hgt_file.close();
    
    // Verify height data values for first and last files
    if (file_index == 0 || file_index == 4) {
        std::cout << "[DEBUG_HGT] File " << file_index << " height data sample: ";
        for (int i = 0; i < std::min(5, (int)dimZ_GFS); i++) {
            std::cout << hgt_data[i] << " ";
        }
        std::cout << "... " << hgt_data[dimZ_GFS-1] << std::endl;
    }
    
    // DRHO 계산 (기존 코드와 동일)
    for (int i = 0; i < dimX_GFS+1; ++i) {
        for (int j = 0; j < dimY_GFS; ++j) {
            float rho0 = pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].RHO;
            float rho1 = pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + 1].RHO;
            float hgt_diff = hgt_data[1] - hgt_data[0];
            
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS].DRHO = (rho1 - rho0) / hgt_diff;
            
            for (int k = 1; k < dimZ_GFS-2; ++k) {
                int index = i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + k;
                pres_data[index].DRHO = 
                (pres_data[index+1].RHO - pres_data[index-1].RHO) / (hgt_data[k+1]-hgt_data[k-1]);
            }

            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO = 
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-3].DRHO;

            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-1].DRHO = 
            pres_data[i * dimY_GFS * dimZ_GFS + j * dimZ_GFS + dimZ_GFS-2].DRHO;
        }
    }
    
    return true;
}

bool LDM::preloadAllEKIMeteorologicalData() {
    std::cout << Color::CYAN << Color::BOLD << "\n[METEO] " << Color::RESET
              << "Starting meteorological data preloading for EKI\n";

    // Clean up existing data
    g_eki_meteo.cleanup();

    // Calculate required number of files
    int num_files = calculateRequiredMeteoFiles();
    if (num_files <= 0) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Invalid file count: " << num_files << std::endl;
        return false;
    }

    // Set metadata
    g_eki_meteo.num_time_steps = num_files;
    g_eki_meteo.pressure_size = (dimX_GFS + 1) * dimY_GFS * dimZ_GFS * sizeof(FlexPres);
    g_eki_meteo.surface_size = (dimX_GFS + 1) * dimY_GFS * sizeof(FlexUnis);
    g_eki_meteo.height_size = dimZ_GFS * sizeof(float);

    // Initialize host memory vectors
    g_eki_meteo.h_pressure.resize(num_files);
    g_eki_meteo.h_surface.resize(num_files);
    g_eki_meteo.h_height.resize(num_files);

    std::cout << "\n" << Color::CYAN << "[METEO] " << Color::RESET
              << "Parallel CPU loading: " << Color::BOLD << num_files << Color::RESET << " files\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Use thread pool for true parallel execution
    std::vector<std::thread> threads;
    std::vector<bool> results(num_files, false);
    std::mutex completion_mutex;
    std::atomic<int> completed_files(0);
    
    for (int i = 0; i < num_files; i++) {
        threads.emplace_back([this, i, &results, &completion_mutex, &completed_files]() {
            FlexPres* pres_data = nullptr;
            FlexUnis* unis_data = nullptr;
            std::vector<float> hgt_data;

            std::cout << "Thread " << std::this_thread::get_id()
                      << " loading file " << i << ".txt..." << std::endl;

            bool success = this->loadSingleMeteoFile(i, pres_data, unis_data, hgt_data);
            if (success) {
                std::lock_guard<std::mutex> lock(completion_mutex);
                g_eki_meteo.h_pressure[i] = pres_data;
                g_eki_meteo.h_surface[i] = unis_data;
                g_eki_meteo.h_height[i] = hgt_data;
                results[i] = true;
                completed_files++;
                std::cout << "Meteorological data file " << i << ".txt loaded successfully ("
                          << completed_files.load() << "/" << g_eki_meteo.num_time_steps << ")" << std::endl;
            } else {
                std::lock_guard<std::mutex> lock(completion_mutex);
                std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to load file " << i << ".txt!" << std::endl;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 결과 확인
    bool all_success = true;
    for (int i = 0; i < num_files; i++) {
        if (!results[i]) {
            all_success = false;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (!all_success) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Some meteorological files failed: "
                  << completed_files.load() << "/" << num_files << "\n";
        g_eki_meteo.cleanup();
        return false;
    }

    std::cout << "CPU loading completed: " << Color::BOLD << num_files << Color::RESET
              << " files (" << duration.count() << "ms)\n";

    // GPU memory allocation and copying
    std::cout << "\n" << Color::CYAN << "[METEO] " << Color::RESET
              << "Transferring to GPU memory\n";
    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory pointer arrays
    cudaError_t err1 = cudaMalloc((void**)&g_eki_meteo.d_pressure_array, num_files * sizeof(FlexPres*));
    cudaError_t err2 = cudaMalloc((void**)&g_eki_meteo.d_surface_array, num_files * sizeof(FlexUnis*));
    cudaError_t err3 = cudaMalloc((void**)&g_eki_meteo.d_height_array, num_files * sizeof(float*));
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to allocate GPU pointer array\n";
        g_eki_meteo.cleanup();
        return false;
    }
    
    // Allocate GPU memory and transfer data for each timestep
    std::vector<FlexPres*> temp_pres_ptrs(num_files, nullptr);
    std::vector<FlexUnis*> temp_unis_ptrs(num_files, nullptr);
    std::vector<float*> temp_hgt_ptrs(num_files, nullptr);
    
    bool gpu_allocation_success = true;
    
    for (int i = 0; i < num_files; i++) {
        // Pres data
        if (cudaMalloc((void**)&temp_pres_ptrs[i], g_eki_meteo.pressure_size) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Pres memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_pres_ptrs[i], g_eki_meteo.h_pressure[i], 
                       g_eki_meteo.pressure_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Pres data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        // Unis data
        if (cudaMalloc((void**)&temp_unis_ptrs[i], g_eki_meteo.surface_size) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Unis memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_unis_ptrs[i], g_eki_meteo.h_surface[i], 
                       g_eki_meteo.surface_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Unis data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        // Height data
        if (cudaMalloc((void**)&temp_hgt_ptrs[i], g_eki_meteo.height_size) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Height memory allocation failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        if (cudaMemcpy(temp_hgt_ptrs[i], g_eki_meteo.h_height[i].data(), 
                       g_eki_meteo.height_size, cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << Color::RED << "[ERROR] " << Color::RESET << "GPU Height data transfer failed (file " << i << ")" << std::endl;
            gpu_allocation_success = false;
            break;
        }
        
        // Silent progress - individual file transfers not reported
    }

    if (!gpu_allocation_success) {
        // Clean up partially allocated memory
        for (int i = 0; i < num_files; i++) {
            if (temp_pres_ptrs[i]) cudaFree(temp_pres_ptrs[i]);
            if (temp_unis_ptrs[i]) cudaFree(temp_unis_ptrs[i]);
            if (temp_hgt_ptrs[i]) cudaFree(temp_hgt_ptrs[i]);
        }
        g_eki_meteo.cleanup();
        return false;
    }

    // Copy pointer arrays to GPU
    err1 = cudaMemcpy(g_eki_meteo.d_pressure_array, temp_pres_ptrs.data(),
                      num_files * sizeof(FlexPres*), cudaMemcpyHostToDevice);
    err2 = cudaMemcpy(g_eki_meteo.d_surface_array, temp_unis_ptrs.data(),
                      num_files * sizeof(FlexUnis*), cudaMemcpyHostToDevice);
    err3 = cudaMemcpy(g_eki_meteo.d_height_array, temp_hgt_ptrs.data(),
                      num_files * sizeof(float*), cudaMemcpyHostToDevice);

    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        std::cerr << Color::RED << Color::BOLD << "[ERROR] " << Color::RESET
                  << "Failed to copy GPU pointer array\n";
        g_eki_meteo.cleanup();
        return false;
    }

    auto gpu_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end_time - gpu_start_time);
    
    // Slot 0 (for past data)
    cudaError_t err_alloc1 = cudaMalloc((void**)&g_eki_meteo.d_pressure_past, g_eki_meteo.pressure_size);
    cudaError_t err_alloc2 = cudaMalloc((void**)&g_eki_meteo.d_surface_past, g_eki_meteo.surface_size);
    
    // Slot 1 (for future data)  
    cudaError_t err_alloc3 = cudaMalloc((void**)&g_eki_meteo.d_pressure_future, g_eki_meteo.pressure_size);
    cudaError_t err_alloc4 = cudaMalloc((void**)&g_eki_meteo.d_surface_future, g_eki_meteo.surface_size);
    
    // Copy pointers to global variables
    d_pressure_past = g_eki_meteo.d_pressure_past;
    d_surface_past = g_eki_meteo.d_surface_past;
    d_pressure_future = g_eki_meteo.d_pressure_future;
    d_surface_future = g_eki_meteo.d_surface_future;
    
    if (err_alloc1 != cudaSuccess || err_alloc2 != cudaSuccess || 
        err_alloc3 != cudaSuccess || err_alloc4 != cudaSuccess) {
        std::cerr << Color::RED << "[ERROR] " << Color::RESET << "Failed to allocate existing LDM GPU memory slots" << std::endl;
        std::cerr << "  d_pressure_past: " << cudaGetErrorString(err_alloc1) << std::endl;
        std::cerr << "  d_surface_past: " << cudaGetErrorString(err_alloc2) << std::endl;
        std::cerr << "  d_pressure_future: " << cudaGetErrorString(err_alloc3) << std::endl;
        std::cerr << "  d_surface_future: " << cudaGetErrorString(err_alloc4) << std::endl;
        g_eki_meteo.cleanup();
        return false;
    }
    
    // 초기 상태로 첫 번째 기상자료 로드 (과거/미래 동일하게)
    if (g_eki_meteo.num_time_steps > 0) {
        FlexPres* first_pres_ptr;
        FlexUnis* first_unis_ptr;
        
        cudaMemcpy(&first_pres_ptr, &g_eki_meteo.d_pressure_array[0], 
                   sizeof(FlexPres*), cudaMemcpyDeviceToHost);
        cudaMemcpy(&first_unis_ptr, &g_eki_meteo.d_surface_array[0], 
                   sizeof(FlexUnis*), cudaMemcpyDeviceToHost);
        
        // Past slot
        cudaMemcpy(g_eki_meteo.d_pressure_past, first_pres_ptr, 
                   g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_eki_meteo.d_surface_past, first_unis_ptr, 
                   g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
        
        // Future slot (initially same data)
        cudaMemcpy(g_eki_meteo.d_pressure_future, first_pres_ptr, 
                   g_eki_meteo.pressure_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(g_eki_meteo.d_surface_future, first_unis_ptr, 
                   g_eki_meteo.surface_size, cudaMemcpyDeviceToDevice);
        
        std::cout << "Initial meteorological data loaded (index 0)" << std::endl;

        // Allocate and initialize d_height_levels for kernel usage
        if (d_height_levels == nullptr) {
            std::cout << "Allocating d_height_levels for kernel usage..." << std::endl;
            cudaError_t hgt_alloc_err = cudaMalloc(&d_height_levels, g_eki_meteo.height_size);
            if (hgt_alloc_err != cudaSuccess) {
                std::cerr << Color::RED << "[ERROR] " << Color::RESET
                          << "Failed to allocate d_height_levels: "
                          << cudaGetErrorString(hgt_alloc_err) << std::endl;
                g_eki_meteo.cleanup();
                return false;
            }

            // Initialize with first height data
            float* first_hgt_ptr;
            cudaMemcpy(&first_hgt_ptr, &g_eki_meteo.d_height_array[0],
                       sizeof(float*), cudaMemcpyDeviceToHost);
            cudaError_t hgt_copy_err = cudaMemcpy(d_height_levels, first_hgt_ptr,
                                                  g_eki_meteo.height_size, cudaMemcpyDeviceToDevice);
            if (hgt_copy_err != cudaSuccess) {
                std::cerr << Color::RED << "[ERROR] " << Color::RESET
                          << "Failed to initialize d_height_levels: "
                          << cudaGetErrorString(hgt_copy_err) << std::endl;
                cudaFree(d_height_levels);
                d_height_levels = nullptr;
                g_eki_meteo.cleanup();
                return false;
            }
            std::cout << "d_height_levels allocated and initialized ("
                      << (g_eki_meteo.height_size / 1024.0) << " KB)" << std::endl;
        }
    }

    g_eki_meteo.is_initialized = true;

    std::cout << "GPU transfer completed (" << gpu_duration.count() << "ms)\n";
    std::cout << "\nMetorological data preloading completed (" << (duration.count() + gpu_duration.count()) << "ms)\n";
    std::cout << "  Memory usage:\n";
    std::cout << "    Pres data   : " << Color::BOLD << (g_eki_meteo.pressure_size * num_files / 1024 / 1024) << " MB" << Color::RESET << "\n";
    std::cout << "    Unis data   : " << Color::BOLD << (g_eki_meteo.surface_size * num_files / 1024 / 1024) << " MB" << Color::RESET << "\n";
    std::cout << "    Height data : " << Color::BOLD << (g_eki_meteo.height_size * num_files / 1024) << " KB" << Color::RESET << "\n";

    return true;
}

void LDM::cleanupEKIMeteorologicalData() {
    std::cout << "Starting EKI meteorological data memory cleanup..." << std::endl;
    g_eki_meteo.cleanup();
    std::cout << "EKI meteorological data memory cleanup completed" << std::endl;
}
