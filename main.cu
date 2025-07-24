#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "wav_helper.h"

// CUDA kernel for performing convolution
__global__ void convolution_kernel(const float* input, const float* ir, float* output, int input_len, int ir_len, int output_len) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < output_len) {
        float sum = 0.0f;
        for (int k = 0; k < ir_len; ++k) {
            int input_idx = n - k;
            if (input_idx >= 0 && input_idx < input_len) {
                sum += input[input_idx] * ir[k];
            }
        }
        output[n] = sum;
    }
}

// CPU version for comparison
void convolution_cpu(const std::vector<float>& input, const std::vector<float>& ir, std::vector<float>& output) {
    int output_len = input.size() + ir.size() - 1;
    output.assign(output_len, 0.0f);

    for (int n = 0; n < output_len; ++n) {
        for (int k = 0; k < ir.size(); ++k) {
            if (n - k >= 0 && n - k < input.size()) {
                output[n] += input[n - k] * ir[k];
            }
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file.wav> <ir_file.wav> <output_file.wav>" << std::endl;
        return 1;
    }

    std::string input_filename = argv[1];
    std::string ir_filename = argv[2];
    std::string output_filename = argv[3];

    // Load audio data
    std::vector<float> h_input, h_ir;
    int sample_rate;
    if (!read_wav(input_filename, h_input, sample_rate) || !read_wav(ir_filename, h_ir, sample_rate)) {
        return 1;
    }

    std::cout << "Input signal length: " << h_input.size() << " samples" << std::endl;
    std::cout << "IR signal length:    " << h_ir.size() << " samples" << std::endl;

    // --- CPU Execution ---
    std::cout << "\n--- Running CPU Convolution ---" << std::endl;
    std::vector<float> h_output_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    convolution_cpu(h_input, h_ir, h_output_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << cpu_duration.count() << " ms" << std::endl;
    write_wav("output_cpu.wav", h_output_cpu, sample_rate);
    std::cout << "CPU output saved to output_cpu.wav" << std::endl;


    // --- GPU Execution ---
    std::cout << "\n--- Running GPU Convolution ---" << std::endl;
    float *d_input, *d_ir, *d_output;
    int output_len = h_input.size() + h_ir.size() - 1;
    std::vector<float> h_output_gpu(output_len);

    // Allocate memory on the GPU
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_ir, h_ir.size() * sizeof(float));
    cudaMalloc(&d_output, output_len * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ir, h_ir.data(), h_ir.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Setup kernel launch configuration
    int threads_per_block = 256;
    int blocks_per_grid = (output_len + threads_per_block - 1) / threads_per_block;
    
    // Use CUDA events for accurate timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    convolution_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, d_ir, d_output, h_input.size(), h_ir.size(), output_len);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float gpu_duration_ms = 0;
    cudaEventElapsedTime(&gpu_duration_ms, start_gpu, stop_gpu);
    std::cout << "GPU kernel execution time: " << gpu_duration_ms << " ms" << std::endl;

    // Copy result back from device to host
    cudaMemcpy(h_output_gpu.data(), d_output, output_len * sizeof(float), cudaMemcpyDeviceToHost);
    write_wav(output_filename, h_output_gpu, sample_rate);
    std::cout << "GPU output saved to " << output_filename << std::endl;

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_ir);
    cudaFree(d_output);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // --- Final Analysis ---
    std::cout << "\n--- Analysis ---" << std::endl;
    double speedup = cpu_duration.count() / gpu_duration_ms;
    std::cout << "Speedup (CPU Time / GPU Time): " << speedup << "x" << std::endl;

    return 0;
}