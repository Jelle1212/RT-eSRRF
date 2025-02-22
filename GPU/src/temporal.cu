#include "temporal.hu"
#include "settings.hu"
#include <stdio.h>

#define THREADS_PER_BLOCK 256  // Adjust based on architecture

__global__ void average_kernel(const float* image_stack, float* image_out, int frames, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float sum = 0.0f;
        for (int f = 0; f < frames; f++) {
            sum += image_stack[f * total_pixels + idx];
        }
        image_out[idx] = sum / frames;
    }
}

__global__ void variance_kernel(const float* image_stack, float* image_out, const float* mean_image, int frames, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float var_sum = 0.0f;
        for (int f = 0; f < frames; f++) {
            float diff = image_stack[f * total_pixels + idx] - mean_image[idx];
            var_sum += diff * diff;
        }
        image_out[idx] = var_sum / frames;
    }
}

__global__ void temporal_auto_correlation_kernel(
    const float* image_stack, float* image_out, const float* mean_image, int frames, int total_pixels, int nlag) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float sum_corr = 0.0f;
        for (int f = 0; f < frames - nlag; f++) {
            float centered1 = image_stack[f * total_pixels + idx] - mean_image[idx];
            float centered2 = image_stack[(f + nlag) * total_pixels + idx] - mean_image[idx];
            sum_corr += centered1 * centered2;
        }
        image_out[idx] = sum_corr / (frames - nlag);
    }
}

extern "C" {
    void temporal(const float* d_image_stack, float* d_image_out, float* d_mean_image, int type, int frames, int rows, int cols) {
        int total_pixels = rows * cols;
        int nlag = 1;

        // Define block and grid size
        dim3 blockSize(THREADS_PER_BLOCK);
        dim3 gridSize((total_pixels + blockSize.x - 1) / blockSize.x);

        if (type == 0) {
            average_kernel<<<gridSize, blockSize>>>(d_image_stack, d_image_out, frames, total_pixels);
        } else if (type == 1) {
            average_kernel<<<gridSize, blockSize>>>(d_image_stack, d_mean_image, frames, total_pixels);
            cudaDeviceSynchronize();  // Ensure mean is computed before using it
            variance_kernel<<<gridSize, blockSize>>>(d_image_stack, d_image_out, d_mean_image, frames, total_pixels);
        } else {
            average_kernel<<<gridSize, blockSize>>>(d_image_stack, d_mean_image, frames, total_pixels);
            cudaDeviceSynchronize();
            temporal_auto_correlation_kernel<<<gridSize, blockSize>>>(
                d_image_stack, d_image_out, d_mean_image, frames, total_pixels, nlag);
        }
    }
}
