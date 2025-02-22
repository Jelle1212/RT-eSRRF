#include "temporal.hu"
#include <stdio.h>

#define THREADS_PER_BLOCK 256  // Adjust based on architecture

__global__ void incremental_average_kernel(const float* d_rgc_map, float* d_mean_image, int frame_idx, int frames, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        d_mean_image[idx] = d_mean_image[idx] + (d_rgc_map[idx] - d_mean_image[idx]) / (float)frames;
    }
}

__global__ void incremental_variance_kernel(const float* d_rgc_map, float* d_mean_image, float* d_var_image, int frame_idx, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float old_mean = d_mean_image[idx];
        float alpha = 1.0f / (frame_idx + 1);
        
        // Update mean
        float new_mean = (1 - alpha) * old_mean + alpha * d_rgc_map[idx];
        d_mean_image[idx] = new_mean;

        // Update variance using Welford’s method
        d_var_image[idx] += (d_rgc_map[idx] - old_mean) * (d_rgc_map[idx] - new_mean);
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
    void temporal(TemporalParams &params, float *d_rgc_map) {
        int total_pixels = params.rowsM * params.colsM;
        int nlag = 1;

        // Define block and grid size
        dim3 blockSize(THREADS_PER_BLOCK);
        dim3 gridSize((total_pixels + blockSize.x - 1) / blockSize.x);

        if (params.type == 0) {
            printf("%d\n",params.frame_idx);
            incremental_average_kernel<<<gridSize, blockSize>>>(d_rgc_map, params.d_sr_image, params.frame_idx, params.frames, total_pixels);
        } else if (params.type == 1) {
            incremental_variance_kernel<<<gridSize, blockSize>>>(d_rgc_map, params.d_sr_image, params.d_mean_image, params.frame_idx, total_pixels);
        } else if (params.type == 2){
            // average_kernel<<<gridSize, blockSize>>>(params.d_rgc_maps, params.d_mean_image, params.frames, total_pixels);
            // cudaDeviceSynchronize();
            // temporal_auto_correlation_kernel<<<gridSize, blockSize>>>(
            //     params.d_rgc_maps, params.d_sr_image, params.d_mean_image, params.frames, total_pixels, nlag);
        } else {
            printf("ERROR: Unsupported Temporal Type: %d\n", params.type);
            return;
        }
    }
}
