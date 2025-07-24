#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include "ppm_helper.h"

// Kernel to convert RGB image to grayscale
__global__ void rgb_to_grayscale_kernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int gray_idx = y * width + x;
        
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        
        output[gray_idx] = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

// DEBUGGING KERNEL: This version simply copies the grayscale image to the output.
__global__ void sobel_filter_kernel(const unsigned char* grayscale_input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        // This line now just copies the input to the output for testing purposes.
        output[idx] = grayscale_input[idx];
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file.ppm> <output_file.ppm>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];

    Image h_input_rgb;
    if (!read_ppm(input_filename, h_input_rgb)) { return 1; }
    
    int width = h_input_rgb.width;
    int height = h_input_rgb.height;
    std::cout << "Image loaded: " << width << "x" << height << std::endl;

    unsigned char *d_input_rgb, *d_grayscale, *d_output;
    cudaMalloc(&d_input_rgb, width * height * 3 * sizeof(unsigned char));
    cudaMalloc(&d_grayscale, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input_rgb, h_input_rgb.data.data(), width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    std::cout << "\n--- Running GPU Image Processing ---" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    rgb_to_grayscale_kernel<<<numBlocks, threadsPerBlock>>>(d_input_rgb, d_grayscale, width, height);
    
    // We are now calling the simplified kernel for debugging.
    sobel_filter_kernel<<<numBlocks, threadsPerBlock>>>(d_grayscale, d_output, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_duration_ms = 0;
    cudaEventElapsedTime(&gpu_duration_ms, start, stop);
    std::cout << "GPU kernel execution time: " << gpu_duration_ms << " ms" << std::endl;

    Image h_output_bw;
    h_output_bw.width = width;
    h_output_bw.height = height;
    h_output_bw.data.resize(width * height);
    cudaMemcpy(h_output_bw.data.data(), d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    Image h_output_rgb;
    h_output_rgb.width = width;
    h_output_rgb.height = height;
    h_output_rgb.data.resize(width * height * 3);
    for(int i = 0; i < width * height; ++i) {
        h_output_rgb.data[i * 3 + 0] = h_output_bw.data[i];
        h_output_rgb.data[i * 3 + 1] = h_output_bw.data[i];
        h_output_rgb.data[i * 3 + 2] = h_output_bw.data[i];
    }

    if (write_ppm(output_filename, h_output_rgb)) {
        std::cout << "Output image saved to " << output_filename << std::endl;
    }

    cudaFree(d_input_rgb);
    cudaFree(d_grayscale);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
