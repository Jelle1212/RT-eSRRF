#include "pipeline.hu"
#include "spatial.hu"
#include "temporal.hu"
#include <cuda_runtime.h>
#include <stdio.h>

float *d_rgc_maps, *d_sr_image, *d_input_frame, *d_output_frame, *d_mean_image;
cudaStream_t stream1, stream2, stream3, stream4;

extern "C" void initPipeline(int nFrames, int rows, int cols, int magnification) {
    int rowsM = rows * magnification;
    int colsM = cols * magnification;
    int total_pixels = rowsM * colsM;

    CHECK_CUDA(cudaMalloc(&d_rgc_maps, nFrames * total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sr_image, total_pixels * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_input_frame, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_frame, total_pixels * sizeof(float))); // Pre-allocated buffer
    CHECK_CUDA(cudaMalloc(&d_mean_image, total_pixels * sizeof(float))); // Pre-allocated buffer
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));
    CHECK_CUDA(cudaStreamCreate(&stream4));
}

extern "C" void processFrame(const float* image_in, float* sr_image, int frame_index, int nFrames, int rows, int cols, int magnification, int shift, int radius, int sensitivity, bool doIntensityWeighting, int type) {
    int rowsM = rows * magnification;
    int colsM = cols * magnification;
    int total_pixels = rowsM * colsM;
    int buffer_offset = (frame_index % nFrames) * total_pixels;

    // Async Copy: Host to Device
    CHECK_CUDA(cudaMemcpyAsync(d_input_frame, image_in, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream1));

    // Spatial processing: directly writes to pre-allocated d_output_frame
    spatial(d_input_frame, d_output_frame, rows, cols, shift, magnification, radius, sensitivity, doIntensityWeighting);

    // Async Copy: Device to Device
    CHECK_CUDA(cudaMemcpyAsync(d_rgc_maps + buffer_offset, d_output_frame, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream3));

    // Trigger temporal processing only when buffer is full
    if (frame_index >= nFrames - 1) {
        temporal(d_rgc_maps, d_sr_image, d_mean_image, type, nFrames, rowsM, colsM);
        CHECK_CUDA(cudaMemcpyAsync(sr_image, d_sr_image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream4));
    }
}

extern "C" void deintPipeline() {
    CHECK_CUDA(cudaFree(d_rgc_maps));
    CHECK_CUDA(cudaFree(d_sr_image));
    CHECK_CUDA(cudaFree(d_input_frame));
    CHECK_CUDA(cudaFree(d_output_frame));
    CHECK_CUDA(cudaFree(d_mean_image));

    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaStreamDestroy(stream4));
}