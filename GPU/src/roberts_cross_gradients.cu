#include "roberts_cross_gradients.hu"

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for Roberts cross gradient
__global__ void roberts_cross_kernel(const float *image, float *imGc, float *imGr, int rows, int cols) {
    // Calculate 2D thread indices
    int c1 = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int r1 = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Check if the thread is within the image bounds
    if (r1 < rows && c1 < cols) {
        // Handle boundary conditions
        int c0 = (c1 > 0) ? c1 - 1 : 0;
        int r0 = (r1 > 0) ? r1 - 1 : 0;

        // Fetch neighboring pixel values
        float im_c0_r1 = image[r0 * cols + c1];
        float im_c1_r0 = image[r1 * cols + c0];
        float im_c0_r0 = image[r0 * cols + c0];
        float im_c1_r1 = image[r1 * cols + c1];

        // Compute gradients
        imGc[r1 * cols + c1] = im_c0_r1 - im_c1_r0 + im_c1_r1 - im_c0_r0;
        imGr[r1 * cols + c1] = -im_c0_r1 + im_c1_r0 + im_c1_r1 - im_c0_r0;
    }
}

// Host function to call the CUDA kernel
extern "C" void roberts_cross_gradients(const float *image, float *gradient_col, float *gradient_row, 
                                       int rows, int cols, cudaStream_t stream) {
    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, 
                  (rows + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    roberts_cross_kernel<<<gridSize, blockSize, 0, stream>>>(image, gradient_col, gradient_row, rows, cols);
    
    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();
}