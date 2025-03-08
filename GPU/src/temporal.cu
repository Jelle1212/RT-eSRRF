#include "temporal.hu"
#include <stdio.h>

#define THREADS_PER_BLOCK 256  // Adjust based on architecture

__global__ void incremental_average_kernel(const float* d_rgc_map, float* d_mean_image, int frame_idx, int total_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don’t access out-of-bounds memory
    if (idx >= total_pixels) return;

    float new_value = d_rgc_map[idx];
    float old_mean = d_mean_image[idx];

    // Compute the incremental mean update
    float updated_mean = (frame_idx * old_mean + new_value) / (frame_idx + 1);

    // Update the mean directly
    d_mean_image[idx] = updated_mean;
}


__global__ void incremental_variance_kernel(const float* d_rgc_map, float* d_mean_image, float* d_var_image, int frame_idx, int total_pixels, int total_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don’t access out-of-bounds memory
    if (idx >= total_pixels) return;

    float new_value = d_rgc_map[idx];  // Current frame pixel value
    float old_mean = d_mean_image[idx];  // Previous mean
    float old_M2 = d_var_image[idx];    // Previous M2 sum

    // Incremental mean update
    float new_mean = (frame_idx * old_mean + new_value) / (frame_idx + 1);

    // Incremental variance update (Welford's algorithm)
    float delta = new_value - old_mean;
    float new_M2 = old_M2 + delta * (new_value - new_mean);

    // Update mean and M2
    d_mean_image[idx] = new_mean;
    d_var_image[idx] = new_M2;

    // Compute the var if we are at the last frame
    if (frame_idx + 1 == total_frames) {
        d_var_image[idx] = new_M2 / total_frames;
    }
}

__global__ void incremental_autocorrelation_kernel(
    const float* d_rgc_map,      // Current frame data
    float* d_autocorr_image,     // Incremental autocorrelation sum
    float* d_mean_image,         // Incremental mean image
    float* d_buffer,             // Buffer storing previous frame pixels
    float* d_sum_x,              // Sum of previous pixels
    float* d_sum_y,              // Sum of next pixels
    float* d_sum_xy,             // Sum of pixel products
    int total_pixels,            // Total number of pixels
    int frame_idx,               // Current frame index
    int total_frames             // Total number of frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_pixels) {
        float new_pixel = d_rgc_map[idx];
        float prev_pixel = (frame_idx > 0) ? d_buffer[idx] : 0.0f;
        float old_mean = d_mean_image[idx];

        // Incrementally update sums
        if (frame_idx > 0) {
            float sum_xy_local = new_pixel * prev_pixel;
            float sum_x_local = prev_pixel;
            float sum_y_local = new_pixel;

            atomicAdd(&d_sum_xy[idx], sum_xy_local);
            atomicAdd(&d_sum_x[idx], sum_x_local);
            atomicAdd(&d_sum_y[idx], sum_y_local);
        }

        // Incremental mean update
        float new_mean = old_mean + (new_pixel - old_mean) / (frame_idx + 1);
        d_mean_image[idx] = new_mean;

        // Store current pixel for next iteration
        d_buffer[idx] = new_pixel;

        // If last frame, finalize the autocorrelation computation
        if (frame_idx == total_frames - 1) {
            int N = total_frames - 1;
            float mean = new_mean;
            float correction = (d_sum_xy[idx] - mean * (d_sum_x[idx] + d_sum_y[idx]) + N * mean * mean) / N;
            d_autocorr_image[idx] = correction;
        }
    }
}

extern "C" {
    void temporal(TemporalParams &params, float *d_rgc_map) {
        int total_pixels = params.rowsM * params.colsM;

        // Define block and grid size
        dim3 blockSize(THREADS_PER_BLOCK);
        dim3 gridSize((total_pixels + blockSize.x - 1) / blockSize.x);

        if (params.type == 0) {
            incremental_average_kernel<<<gridSize, blockSize>>>(d_rgc_map, params.d_sr_image, params.frame_idx, total_pixels);
        } else if (params.type == 1) {
            incremental_variance_kernel<<<gridSize, blockSize>>>(d_rgc_map, params.d_mean_image, params.d_sr_image, params.frame_idx, total_pixels, params.frames);
        } else if (params.type == 2){
            incremental_autocorrelation_kernel<<<gridSize, blockSize>>>(
                d_rgc_map, params.d_sr_image, params.d_mean_image, params.d_buffer, params.d_sum_x, params.d_sum_y, params.d_sum_xy, total_pixels, params.frame_idx, params.frames
            );
        } else {
            printf("ERROR: Unsupported Temporal Type: %d\n", params.type);
            return;
        }
        // Synchronize to ensure kernel execution is complete
        cudaDeviceSynchronize();
        // convert_double_to_float<<<gridSize, blockSize>>>(params.d_mean_image, params.d_sr_image, total_pixels);

    }
}
