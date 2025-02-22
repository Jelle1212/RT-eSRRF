#include "pipeline.hu"
#include "spatial.hu"
#include "temporal.hu"
#include <cuda_runtime.h>
#include <stdio.h>

cudaStream_t stream1, stream2, stream3, stream4;
SpatialParams spatialParams;
TemporalParams temporalParams;

extern "C" void initPipeline(const struct ESRRFParams* eSRRFParams) {
    int rowsM = eSRRFParams->rows * eSRRFParams->magnification;
    int colsM = eSRRFParams->cols * eSRRFParams->magnification;
    int total_pixels = rowsM * colsM;

    // CHECK_CUDA(cudaMalloc(&temporalParams.d_rgc_map, eSRRFParams->nFrames * total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_sr_image, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&temporalParams.d_mean_image, total_pixels * sizeof(float))); // Pre-allocated buffer
    CHECK_CUDA(cudaMalloc(&spatialParams.d_image_in, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
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
    CHECK_CUDA(cudaMemset(temporalParams.d_sr_image, 0, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMemset(temporalParams.d_mean_image, 0, total_pixels * sizeof(float))); 
    CHECK_CUDA(cudaMemset(spatialParams.d_image_in, 0, eSRRFParams->rows * eSRRFParams->cols * sizeof(float)));
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

extern "C" void processFrame(const float* image_in, float* sr_image, int frame_index) {
    int rowsM = spatialParams.rows * spatialParams.magnification;
    int colsM = spatialParams.cols * spatialParams.magnification;
    int total_pixels = rowsM * colsM;
    // int buffer_offset = (frame_index % temporalParams.frames) * total_pixels;

    // Async Copy: Host to Device
    CHECK_CUDA(cudaMemcpyAsync(spatialParams.d_image_in, image_in, spatialParams.rows * spatialParams.cols * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // Spatial processing: directly writes to pre-allocated d_output_frame
    spatial(spatialParams);

    // Async Copy: Device to Device
    // CHECK_CUDA(cudaMemcpyAsync(temporalParams.d_rgc_maps + buffer_offset, spatialParams.d_rgc_map, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream3));
    temporal(temporalParams, spatialParams.d_rgc_map);
    temporalParams.frame_idx++;
    // Trigger temporal processing only when buffer is full
    if (frame_index >= temporalParams.frames - 1) {
        CHECK_CUDA(cudaMemcpyAsync(sr_image, temporalParams.d_sr_image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream4));
    }
}

extern "C" void deintPipeline() {
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_row_interp));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_col_interp));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_row));
    CHECK_CUDA(cudaFree(spatialParams.d_gradient_col));
    CHECK_CUDA(cudaFree(spatialParams.d_magnified_image));
    CHECK_CUDA(cudaFree(temporalParams.d_mean_image));
    CHECK_CUDA(cudaFree(spatialParams.d_rgc_map));
    CHECK_CUDA(cudaFree(spatialParams.d_image_in));
    CHECK_CUDA(cudaFree(temporalParams.d_sr_image));
    // CHECK_CUDA(cudaFree(temporalParams.d_rgc_maps));

    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream1));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream2));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream3));
    CHECK_CUDA(cudaStreamDestroy(spatialParams.stream4));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaStreamDestroy(stream4));
}