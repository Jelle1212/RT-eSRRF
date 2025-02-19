#include <cuda_runtime.h>
#include <stdio.h>

// Preallocate GPU memory for spatial processing
float *d_rgc_maps, *d_sr_image, *d_input_frame, *d_output_frame;
cudaStream_t stream1, stream2, stream3, stream4;

extern "C" void initPipeline(int nFrames, int rows, int cols, int magnification) {
    int rowsM = rows * magnification;
    int colsM = cols * magnification;
    int total_pixels = rowsM * colsM;

    cudaMalloc(&d_rgc_maps, nFrames * total_pixels * sizeof(float));
    cudaMalloc(&d_sr_image, total_pixels * sizeof(float));
    cudaMalloc(&d_input_frame, rows * cols * sizeof(float));
    cudaMalloc(&d_output_frame, total_pixels * sizeof(float));

    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
}

extern "C" void processFrame(const float* image_in, float* sr_image, int frame_index, int nFrames, int rows, int cols, int magnification) {
    int rowsM = rows * magnification;
    int colsM = cols * magnification;
    int total_pixels = rowsM * colsM;

    // Copy frame to GPU
    cudaMemcpyAsync(d_input_frame, image_in, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream1);

    // Run spatial processing
    spatial(d_input_frame, d_output_frame, rows, cols, shift, magnification, radius, sensitivity, doIntensityWeighting);

    // Copy processed frame into GPU buffer
    int buffer_offset = (frame_index % nFrames) * total_pixels;
    cudaMemcpyAsync(d_rgc_maps + buffer_offset, d_output_frame, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice, stream2);

    // When buffer is full, trigger temporal processing
    if (frame_index >= nFrames - 1) {
        int blockSize = 256;
        int gridSize = (total_pixels + blockSize - 1) / blockSize;
        temporal_kernel<<<gridSize, blockSize, 0, stream3>>>(d_rgc_maps, d_sr_image, nFrames, rowsM, colsM);
        cudaMemcpyAsync(sr_image, d_sr_image, total_pixels * sizeof(float), cudaMemcpyDeviceToHost, stream4);
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
}