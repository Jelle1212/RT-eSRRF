#include "pipeline.hu"
#include "spatial.hu"
#include "temporal.hu"
#include "shift_magnify.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

cudaStream_t stream1, stream2, stream3, stream4;
SpatialParams spatialParams;
TemporalParams temporalParams;

__global__ void normalizeUint16ToFloat(const unsigned short* input, float* output, int width, int height, uint16_t max_value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = static_cast<float>(input[idx]) / static_cast<float>(max_value);
    }
}

extern "C" void launchNormalizationKernel(const unsigned short* d_input, float* d_output, int width, int height, uint16_t max_value, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    
    normalizeUint16ToFloat<<<gridSize, blockSize, 0, stream>>>(d_input, d_output, width, height, max_value);
}

extern "C" void initPipeline(const struct ESRRFParams* eSRRFParams) {
    int rowsM = eSRRFParams->rows * eSRRFParams->magnification;
    int colsM = eSRRFParams->cols * eSRRFParams->magnification;
    int total_pixels = rowsM * colsM;

    CHECK_CUDA(cudaMalloc(&temporalParams.d_sum_x, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_sum_y, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_sum_xy, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_buffer, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_sr_image, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_mean_image, total_pixels * sizeof(float))); // Pre-allocated buffer
    CHECK_CUDA(cudaMalloc(&spatialParams.d_image_in, eSRRFParams->rows * eSRRFParams->cols * sizeof(unsigned short)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_n_image_in, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_rgc_map, total_pixels * sizeof(float))); // Pre-allocated buffer
    CHECK_CUDA(cudaMalloc(&spatialParams.d_magnified_image, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_gradient_col, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_gradient_row, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_gradient_col_interp, 2 * rowsM * 2 * colsM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&spatialParams.d_gradient_row_interp, 2 * rowsM * 2 * colsM * sizeof(float)));
    CHECK_CUDA(cudaStreamCreate(&spatialParams.stream1));
    CHECK_CUDA(cudaStreamCreate(&spatialParams.stream2));
    CHECK_CUDA(cudaStreamCreate(&spatialParams.stream3));
    CHECK_CUDA(cudaStreamCreate(&spatialParams.stream4));
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));
    CHECK_CUDA(cudaStreamCreate(&stream4));

    // Initialize all allocated memory to 0.0
    CHECK_CUDA(cudaMemset(temporalParams.d_sum_x, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_sum_y, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_sum_xy, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_buffer, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_sr_image, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_mean_image, 0, total_pixels * sizeof(float))); 
    CHECK_CUDA(cudaMemset(spatialParams.d_image_in, 0, eSRRFParams->rows * eSRRFParams->cols * sizeof(unsigned short)));
    CHECK_CUDA(cudaMemset(spatialParams.d_n_image_in, 0, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_rgc_map, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_magnified_image, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_gradient_col, 0, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_gradient_row, 0, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_gradient_col_interp, 0, 2 * rowsM * 2 * colsM * sizeof(float)));
    CHECK_CUDA(cudaMemset(spatialParams.d_gradient_row_interp, 0, 2 * rowsM * 2 * colsM * sizeof(float)));

    spatialParams.rows = eSRRFParams->rows;
    spatialParams.cols = eSRRFParams->cols;
    spatialParams.shift = eSRRFParams->shift;
    spatialParams.magnification = eSRRFParams->magnification;
    spatialParams.radius = eSRRFParams->radius;
    spatialParams.sensitivity = eSRRFParams->sensitivity;
    spatialParams.doIntensityWeighting = eSRRFParams->doIntensityWeighting;

    temporalParams.rowsM = rowsM;
    temporalParams.colsM = colsM;
    temporalParams.type = eSRRFParams->temporalType;
    temporalParams.frames = eSRRFParams->nFrames;
    temporalParams.frame_idx = 0;
}

extern "C" void processFrame(const unsigned short* image_in, float* sr_image, int frame_index) {
    int rowsM = spatialParams.rows * spatialParams.magnification;
    int colsM = spatialParams.cols * spatialParams.magnification;
    // int rowsM = spatialParams.rows * 1;
    // int colsM = spatialParams.cols * 1;
    int total_pixels = rowsM * colsM;

    // Async Copy: Host to Device
    CHECK_CUDA(cudaMemcpyAsync(spatialParams.d_image_in, image_in, spatialParams.rows * spatialParams.cols * sizeof(unsigned short), cudaMemcpyHostToDevice, stream1));
    
    // Ensure synchronization if needed
    CHECK_CUDA(cudaStreamSynchronize(stream1));

    // Normalize input image
    launchNormalizationKernel(spatialParams.d_image_in, spatialParams.d_n_image_in, spatialParams.rows, spatialParams.cols, UINT16_MAX, stream2);

    // Ensure synchronization if needed
    CHECK_CUDA(cudaStreamSynchronize(stream2));
    
    // Spatial processing: directly writes to pre-allocated d_output_frame
    spatial(spatialParams);

    // Temporal processing
    temporal(temporalParams, spatialParams.d_rgc_map);

    // Update frame index
    temporalParams.frame_idx++;

    // Trigger temporal processing only when buffer is full
    if (frame_index >= temporalParams.frames - 1) {
        CHECK_CUDA(cudaMemcpyAsync(sr_image, temporalParams.d_sr_image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream4));
        // Ensure synchronization on stream4
        CHECK_CUDA(cudaStreamSynchronize(stream4));  // Synchronize stream4, making sure memory is copied to host
    }
}

extern "C" void deintPipeline() {
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_row_interp));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_col_interp));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_row));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_col));
    CHECK_CUDA(cudaFree(spatialParams.d_magnified_image));
    CHECK_CUDA(cudaFree(spatialParams.d_rgc_map));
    CHECK_CUDA(cudaFree(spatialParams.d_image_in));
    CHECK_CUDA(cudaFree(spatialParams.d_n_image_in));
    CHECK_CUDA(cudaFree(temporalParams.d_sr_image));
    CHECK_CUDA(cudaFree(temporalParams.d_mean_image));

    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream1));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream2));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream3));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream4));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaStreamDestroy(stream4));
}