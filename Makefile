# Makefile for CUDA Audio Processor

# Compiler
NVCC = nvcc

# Compiler flags
# -std=c++14 enables modern C++ features
# -O3 is an optimization flag
CXXFLAGS = -std=c++14 -O3

# The target executable
TARGET = audio_processor

# The source files
SOURCES = main.cu

# Default rule to build the target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

# Rule to clean up the build directory
clean:
	rm -f $(TARGET) *.o output_*.wav

.PHONY: all clean