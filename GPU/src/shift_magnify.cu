#include "shift_magnify.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float cubic(float v) {
    const float a = 0.5f;
    const float v_abs = fabsf(v);
    const float v2 = v_abs * v_abs;
    const float v3 = v2 * v_abs;

    return (v_abs < 1.0f) * (v3 * (-a + 2.0f) + v2 * (a - 3.0f) + 1.0f) +
           (v_abs >= 1.0f && v_abs < 2.0f) * (v3 * (-a) + v2 * (5.0f * a) - v_abs * (8.0f * a) + (4.0f * a));
}
  
__device__ float interpolate(const float *image, float r, float c, int rows, int cols) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) return 0.0f;

    const int r_int = __float2int_rd(r - 0.5f);
    const int c_int = __float2int_rd(c - 0.5f);
    
    float row_weights[4], col_weights[4];

    #pragma unroll
    for (int i = 0; i < 4; i++) row_weights[i] = cubic(r - (r_int - 1 + i + 0.5f));
    #pragma unroll
    for (int j = 0; j < 4; j++) col_weights[j] = cubic(c - (c_int - 1 + j + 0.5f));

    float q = 0.0f;
    
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int c_idx = c_int - 1 + j;
        if (c_idx < 0 || c_idx >= cols) continue;

        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r_idx = r_int - 1 + i;
            if (r_idx < 0 || r_idx >= rows) continue;
            p += image[r_idx * cols + c_idx] * row_weights[i];
        }
        q += p * col_weights[j];
    }

    return q;
}

__global__ void shift_magnify_kernel(const float *image_in, float *image_out, 
                                    int rows, int cols, 
                                    float shift_row, float shift_col, 
                                    float inv_magnification_row, float inv_magnification_col) {
    int rowsM = (int)(rows / inv_magnification_row);
    int colsM = (int)(cols / inv_magnification_col);

    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i < rowsM && j < colsM) {
        float row = i * inv_magnification_row - shift_row;
        float col = j * inv_magnification_col - shift_col;
        image_out[i * colsM + j] = interpolate(image_in, row, col, rows, cols);
    }
}

extern "C" void shift_magnify(const float *image_in, float *image_out, 
                     int rows, int cols, float shift_row, float shift_col, 
                     float magnification_row, float magnification_col, cudaStream_t stream) {

    int rowsM = (int)(rows * magnification_row);
    int colsM = (int)(cols * magnification_col);

    float inv_magnification_row = 1.0f / magnification_row;
    float inv_magnification_col = 1.0f / magnification_col;

    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((colsM + blockSize.x - 1) / blockSize.x, 
                  (rowsM + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    shift_magnify_kernel<<<gridSize, blockSize, 0, stream>>>(image_in, image_out, 
                                                            rows, cols, 
                                                            shift_row, shift_col, 
                                                            inv_magnification_row, inv_magnification_col);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();
}
