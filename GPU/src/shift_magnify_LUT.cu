// #include "shift_magnify.hu"
// #include <cuda_runtime.h>
// #include <stdio.h>
// #include <math.h>

// extern "C" float cubic(float v) {
//     const float a = 0.5f;
//     const float v_abs = fabsf(v);
//     const float v2 = v_abs * v_abs;
//     const float v3 = v2 * v_abs;

//     float w0 = 0.0f, w1 = 0.0f;

//     if (v_abs < 1.0f) {
//         w0 = fmaf(v3, (-a + 2.0f), fmaf(v2, (a - 3.0f), 1.0f));
//     } else if (v_abs < 2.0f) {
//         w1 = fmaf(v3, (-a), fmaf(v2, (5.0f * a), fmaf(-v_abs, (8.0f * a), (4.0f * a))));
//     }

//     return w0 + w1;
// }

// // Host function to initialize the LUT
// extern "C" void initialize_lut() {
//     float *lut_host = new float[LUT_SIZE];
//     float step = (LUT_MAX - LUT_MIN) / (LUT_SIZE - 1);
//     for (int i = 0; i < LUT_SIZE; i++) {
//         float v = LUT_MIN + i * step;
//         lut_host[i] = cubic(v);
//     }

//     // Copy LUT to constant memory
//     cudaMemcpyToSymbol(lut_constant, lut_host, LUT_SIZE * sizeof(float));

//     // Free host memory
//     delete[] lut_host;
// }

// __device__ __forceinline__ float cubic_from_lut(float v, const float scale) {
//     // Compute index directly
//     int index = __float2int_rn((v - LUT_MIN) * scale); // Use faster rounding mode

//     // Access LUT in constant memory
//     return __ldg(&lut_constant[index]); // Use __ldg for read-only access
// }

// __global__ void shift_magnify_kernel(const float *image_in, float *image_out, 
//                                     int rows, int cols, 
//                                     float shift_row, float shift_col, 
//                                     float inv_magnification_row, float inv_magnification_col) {
//     extern __shared__ float shared_image[];

//     int rowsM = (int)(rows / inv_magnification_row);
//     int colsM = (int)(cols / inv_magnification_col);

//     int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
//     int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

//     // Load the relevant portion of the image into shared memory
//     int shared_width = blockDim.x + 3; // 4x4 neighborhood
//     int shared_height = blockDim.y + 3;

//     for (int idx = threadIdx.y; idx < shared_height; idx += blockDim.y) {
//         for (int jdx = threadIdx.x; jdx < shared_width; jdx += blockDim.x) {
//             int r_idx = blockIdx.y * blockDim.y + idx - 1;
//             int c_idx = blockIdx.x * blockDim.x + jdx - 1;
//             if (r_idx >= 0 && r_idx < rows && c_idx >= 0 && c_idx < cols) {
//                 shared_image[idx * shared_width + jdx] = image_in[r_idx * cols + c_idx];
//             } else {
//                 shared_image[idx * shared_width + jdx] = 0.0f;
//             }
//         }
//     }
//     __syncthreads();

//     if (i < rowsM && j < colsM) {
//         float row = i * inv_magnification_row - shift_row;
//         float col = j * inv_magnification_col - shift_col;

//         const int r_int = __float2int_rd(row - 0.5f);
//         const int c_int = __float2int_rd(col - 0.5f);

//         float row_weights[4], col_weights[4];

//         // Precompute distances and weights
//         #pragma unroll
//         for (int k = 0; k < 4; k++) {
//             float dist_row = row - (r_int - 1 + k + 0.5f);
//             float dist_col = col - (c_int - 1 + k + 0.5f);
//             row_weights[k] = cubic_from_lut(dist_row, SCALE);
//             col_weights[k] = cubic_from_lut(dist_col, SCALE);
//         }


//         float q = 0.0f;
//         #pragma unroll
//         for (int jj = 0; jj < 4; jj++) {
//             int c_idx = c_int - 1 + jj - blockIdx.x * blockDim.x + threadIdx.x + 1;
//             if (c_idx < 0 || c_idx >= shared_width) continue;

//             float p = 0.0f;
//             #pragma unroll
//             for (int ii = 0; ii < 4; ii++) {
//                 int r_idx = r_int - 1 + ii - blockIdx.y * blockDim.y + threadIdx.y + 1;
//                 if (r_idx < 0 || r_idx >= shared_height) continue;
//                 p += shared_image[r_idx * shared_width + c_idx] * row_weights[ii];
//             }
//             q += p * col_weights[jj];
//         }

//         image_out[i * colsM + j] = q;
//     }
// }

// extern "C" void shift_magnify(const float *image_in, float *image_out, 
//                               int rows, int cols, 
//                               float shift_row, float shift_col, 
//                               float magnification_row, float magnification_col,
//                               cudaStream_t stream) {
//     int rowsM = (int)(rows * magnification_row);
//     int colsM = (int)(cols * magnification_col);

//     float inv_magnification_row = 1.0f / magnification_row;
//     float inv_magnification_col = 1.0f / magnification_col;

//     dim3 blockSize(16, 16);
//     dim3 gridSize((colsM + blockSize.x - 1) / blockSize.x, 
//                   (rowsM + blockSize.y - 1) / blockSize.y);

//     size_t size_memory = (blockSize.y + 3) * (blockSize.x + 3) * sizeof(float);

//     // Launch kernel
//     shift_magnify_kernel<<<gridSize, blockSize, size_memory, stream>>>( 
//         image_in, image_out,
//         rows, cols,
//         shift_row, shift_col,
//         inv_magnification_row, inv_magnification_col
//     );
// }
