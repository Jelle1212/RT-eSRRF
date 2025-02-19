# Compilers
CC = gcc
NVCC = nvcc

# Compiler flags
CFLAGS = -Wall -Iinclude
NVCCFLAGS = -Iinclude

# Directories
SRC_DIR = src
BUILD_DIR = build

# Source files
C_SRCS = $(wildcard $(SRC_DIR)/*.c)
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o) \
       $(CU_SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

# Executable name
TARGET = ESRRF_GPU

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ -ltiff -lm

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
