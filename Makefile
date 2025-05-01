# Compiler
NVCC := nvcc

# CUDA/cuDNN installation path (override if needed)
CUDA_PATH ?= /usr/local/cuda

# Include and library paths
INCLUDES   := -Iinclude -I$(CUDA_PATH)/include
LIB_PATHS  := -L$(CUDA_PATH)/lib64
LIBS       := -lcudnn

# NVCC & linker flags
NVCC_FLAGS := -O2 $(INCLUDES)
LDFLAGS    := $(LIB_PATHS) $(LIBS)

# Directories
SRC_DIR := src
OBJ_DIR := obj

# Custom‐CUDA helper sources (no mains)
CUSTOM_SRCS := \
    cuda_convolution.cu \
    batch_norm.cu      \
    relu.cu            \
    pooling.cu

# cuDNN helper sources
CUDNN_SRCS := \
    cudnn_convolution.cu \
    cudnn_batch_norm.cu \
	cudnn_relu.cu \
	cudnn_pooling.cu

# Drivers
DRIVER_SRCS := cnn_pipeline.cu       # selects custom vs cudnn
STANDALONE_SRCS := cudnn_pipeline.cu # cudNN-only pipeline

# Map to full paths
CUSTOM_OBJS    := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CUSTOM_SRCS))
CUDNN_OBJS     := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(CUDNN_SRCS))
DRIVER_OBJ     := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(DRIVER_SRCS))
STANDALONE_OBJ := $(patsubst %.cu,$(OBJ_DIR)/%.o,$(STANDALONE_SRCS))

# Targets
all: cnn_pipeline cudnn_pipeline

# 1) Combined driver
cnn_pipeline: $(DRIVER_OBJ) $(CUSTOM_OBJS) $(CUDNN_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# 2) Standalone cuDNN pipeline
cudnn_pipeline: $(STANDALONE_OBJ) $(CUDNN_OBJS) $(CUSTOM_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(LDFLAGS)

# Generic rule for compiling .cu -> .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJ_DIR) cnn_pipeline cudnn_pipeline

.PHONY: all clean