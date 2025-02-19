#include "shift_magnify.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__device__ float cubic(float v) {
  float a = 0.5f;
  float v_abs = fabsf(v); // Absolute value of v
  float z = 0.0f;

  // Precompute constants
  float term1 = -a + 2.0f;
  float term2 = a - 3.0f;

  // Evaluate cubic function without branches
  float v_lt1 = v_abs < 1.0f;
  float v_lt2 = v_abs < 2.0f;

  z = v_lt1 * (v_abs * v_abs * (v_abs * term1 + term2) + 1.0f) +
      (v_lt2 && !v_lt1) * (-a * v_abs * v_abs * v_abs + 5.0f * a * v_abs * v_abs - 8.0f * a * v_abs + 4.0f * a);

  return z;
}
  
__device__ float interpolate(const float *image, float r, float c, int rows, int cols) {
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
      return 0.0f;
  }

  const int r_int = __float2int_rd(r - 0.5f); // Fast rounding
  const int c_int = __float2int_rd(c - 0.5f); // Fast rounding
  float q = 0.0f;

  // Precompute cubic weights for rows and columns
  float row_weights[4], col_weights[4];
  for (int i = 0; i < 4; i++) {
      int r_neighbor = r_int - 1 + i;
      row_weights[i] = cubic(r - (r_neighbor + 0.5f));
  }
  for (int j = 0; j < 4; j++) {
      int c_neighbor = c_int - 1 + j;
      col_weights[j] = cubic(c - (c_neighbor + 0.5f));
  }

  // Perform interpolation
  for (int j = 0; j < 4; j++) {
      int c_neighbor = c_int - 1 + j;
      if (c_neighbor < 0 || c_neighbor >= cols) {
          continue;
      }

      float p = 0.0f;
      for (int i = 0; i < 4; i++) {
          int r_neighbor = r_int - 1 + i;
          if (r_neighbor < 0 || r_neighbor >= rows) {
              continue;
          }
          p += image[r_neighbor * cols + c_neighbor] * row_weights[i];
      }
      q += p * col_weights[j];
  }

  return q;
}

__global__ void shift_magnify_kernel(const float *image_in, float *image_out, 
                                    int rows, int cols, 
                                    float shift_row, float shift_col, 
                                    float magnification_row, float magnification_col) {
    int rowsM = (int)(rows * magnification_row);
    int colsM = (int)(cols * magnification_col);

    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    if (i < rowsM && j < colsM) {
      float row = i / magnification_row - shift_row;
      float col = j / magnification_col - shift_col;
      image_out[i * colsM + j] = interpolate(image_in, row, col, rows, cols);
  }
}
  
extern "C" void shift_magnify(const float *image_in, float *image_out, 
                     int rows, int cols, float shift_row, float shift_col, 
                     float magnification_row, float magnification_col, cudaStream_t stream) {

    int rowsM = (int)(rows * magnification_row);
    int colsM = (int)(cols * magnification_col);

    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((colsM + blockSize.x - 1) / blockSize.x, 
                  (rowsM + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    shift_magnify_kernel<<<gridSize, blockSize, 0, stream>>>(image_in, image_out, 
                                                            rows, cols, 
                                                            shift_row, shift_col, 
                                                            magnification_row, magnification_col);
}

