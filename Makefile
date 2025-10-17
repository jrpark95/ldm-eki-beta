NVCC = /usr/local/cuda/bin/nvcc
# Optimized flags for faster compilation:
# - Removed -DCRAM_DEBUG (CRAM not used)
# - Using -O2 instead of -O3 (faster compilation, minimal runtime difference)
# - Parallel build enabled (use: make -j4)
NVCCFLAGS = -w -O2 -arch=sm_61 -I./src -I./src/kernels -I./src/kernels/cram -I./src/data/config

# Enable parallel builds
MAKEFLAGS += -j$(shell nproc)

ifeq ($(OS),Windows_NT)
    OS_DETECTED := Windows
    PATH_SEP := /
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        OS_DETECTED := Linux
        # Removed -fPIC, -lcublas, -lmpi (not actively used)
        # Note: mpiRank/mpiSize variables exist in code but MPI functions are not called
        NVCCFLAGS += -Xcompiler -fopenmp -lgomp
        PATH_SEP := /
    endif
endif

OBJ_PATH = .$(PATH_SEP)objectfiles

TARGET = ldm
TARGET_EKI = ldm-eki
TARGET_RECEPTOR_DEBUG = ldm-receptor-debug

# ============================================================================
# REFACTORED SOURCE FILE ORGANIZATION
# ============================================================================
# All files organized by module following PARALLEL_REFACTORING_FINAL.md

# Core LDM class sources
CORE_SOURCES = \
    src/core/ldm.cu \
    src/core/device_storage.cu

# Kernel module sources (Agent 1)
KERNEL_SOURCES = \
    src/kernels/device/ldm_kernels_device.cu \
    src/kernels/particle/ldm_kernels_particle.cu \
    src/kernels/particle/ldm_kernels_particle_ens.cu \
    src/kernels/eki/ldm_kernels_eki.cu \
    src/kernels/dump/ldm_kernels_dump.cu \
    src/kernels/dump/ldm_kernels_dump_ens.cu

# Data module sources (Agent 2)
DATA_SOURCES = \
    src/data/meteo/ldm_mdata_loading.cu \
    src/data/meteo/ldm_mdata_processing.cu \
    src/data/meteo/ldm_mdata_cache.cu

# Simulation module sources (Agent 2)
SIMULATION_SOURCES = \
    src/simulation/ldm_func_simulation.cu \
    src/simulation/ldm_func_particle.cu \
    src/simulation/ldm_func_output.cu

# Visualization module sources (Agent 3)
VISUALIZATION_SOURCES = \
    src/visualization/ldm_plot_vtk.cu \
    src/visualization/ldm_plot_utils.cu

# Initialization module sources (Agent 3)
INIT_SOURCES = \
    src/init/ldm_init_particles.cu \
    src/init/ldm_init_config.cu

# IPC module sources (Agent 4)
IPC_SOURCES = \
    src/ipc/ldm_eki_writer.cu \
    src/ipc/ldm_eki_reader.cu

# Physics module sources (Agent 4)
PHYSICS_SOURCES = \
    src/physics/ldm_cram2.cu \
    src/physics/ldm_nuclides.cu

# Debug module sources (Agent 4)
DEBUG_SOURCES = \
    src/debug/memory_doctor.cu \
    src/debug/kernel_error_collector.cu

# All CUDA sources combined
ALL_CU_SOURCES = \
    $(CORE_SOURCES) \
    $(KERNEL_SOURCES) \
    $(DATA_SOURCES) \
    $(SIMULATION_SOURCES) \
    $(VISUALIZATION_SOURCES) \
    $(INIT_SOURCES) \
    $(IPC_SOURCES) \
    $(PHYSICS_SOURCES) \
    $(DEBUG_SOURCES)

# Main entry points for each executable
SRCS = src/main.cu $(ALL_CU_SOURCES)
SRCS_EKI = src/main_eki.cu $(ALL_CU_SOURCES)
SRCS_RECEPTOR_DEBUG = src/main_receptor_debug.cu $(ALL_CU_SOURCES)

# Convert source paths to object paths (handle subdirectories)
OBJS = $(patsubst %.cu,$(OBJ_PATH)/%.o,$(SRCS))
OBJS_EKI = $(patsubst %.cu,$(OBJ_PATH)/%.o,$(SRCS_EKI))
OBJS_RECEPTOR_DEBUG = $(patsubst %.cu,$(OBJ_PATH)/%.o,$(SRCS_RECEPTOR_DEBUG))

all: $(TARGET_EKI)

all-targets: $(TARGET) $(TARGET_EKI) $(TARGET_RECEPTOR_DEBUG)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS)

$(TARGET_EKI): $(OBJS_EKI)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS_EKI)

$(TARGET_RECEPTOR_DEBUG): $(OBJS_RECEPTOR_DEBUG)
	$(NVCC) $(NVCCFLAGS) -o $@ $(OBJS_RECEPTOR_DEBUG)

$(OBJ_PATH)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_PATH)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -x c++ -c $< -o $@

clean:
ifeq ($(OS_DETECTED),Windows)
	del /Q /F $(subst /,\,$(OBJS)) $(subst /,\,$(OBJS_EKI)) $(subst /,\,$(OBJS_RECEPTOR_DEBUG)) $(TARGET).exe $(TARGET_EKI).exe $(TARGET_RECEPTOR_DEBUG).exe $(TARGET).exp $(TARGET).lib $(TARGET_EKI).exp $(TARGET_EKI).lib $(TARGET_RECEPTOR_DEBUG).exp $(TARGET_RECEPTOR_DEBUG).lib
else
	rm -f $(OBJS) $(OBJS_EKI) $(OBJS_RECEPTOR_DEBUG) $(TARGET) $(TARGET_EKI) $(TARGET_RECEPTOR_DEBUG)
endif

$(OBJS) $(OBJS_EKI) $(OBJS_RECEPTOR_DEBUG): | $(OBJ_PATH)

$(OBJ_PATH):
	mkdir -p $(OBJ_PATH)
