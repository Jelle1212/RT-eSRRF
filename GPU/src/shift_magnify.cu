#include "shift_magnify.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// __device__ __forceinline__ float cubic(float v) {
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

extern "C" float cubic(float v) {
    const float a = 0.5f;
    const float v_abs = fabsf(v);
    const float v2 = v_abs * v_abs;
    const float v3 = v2 * v_abs;

    float w0 = 0.0f, w1 = 0.0f;

    if (v_abs < 1.0f) {
        w0 = fmaf(v3, (-a + 2.0f), fmaf(v2, (a - 3.0f), 1.0f));
    } else if (v_abs < 2.0f) {
        w1 = fmaf(v3, (-a), fmaf(v2, (5.0f * a), fmaf(-v_abs, (8.0f * a), (4.0f * a))));
    }

    return w0 + w1;
}

// Host function to initialize the LUT
extern "C" void initialize_lut() {
    float *lut_host = new float[LUT_SIZE];
    float step = (LUT_MAX - LUT_MIN) / (LUT_SIZE - 1);
    for (int i = 0; i < LUT_SIZE; i++) {
        float v = LUT_MIN + i * step;
        lut_host[i] = cubic(v);
    }

    // Copy LUT to constant memory
    cudaMemcpyToSymbol(lut_constant, lut_host, LUT_SIZE * sizeof(float));

    // Free host memory
    delete[] lut_host;
}

__device__ __forceinline__ float cubic_from_lut(float v, const float* __restrict__ lut) {
    // Compute index directly
    int index = __float2int_rz(__fmul_rz((v - LUT_MIN), SCALE));

    // Access LUT in constant memory
    return __ldg(&lut[index]); 
}

////////////////////////////////////////////////////////////////////////////////
// Shared-memory interpolation (mimicking the original function)
// Note: 'tile' is a subset of the global image.
////////////////////////////////////////////////////////////////////////////////
__device__ __forceinline__ float interpolate_shared(const float* __restrict__ sub_image, 
                                                     float r, float c, 
                                                     int tile_width, int tile_height) {
    // Early exit if (r, c) is entirely out of bounds.
    if (r < 0 || r >= tile_height || c < 0 || c >= tile_width)
        return 0.0f;

    // Compute integer base position.
    const int r_int = __float2int_rd(r - 0.5f);
    const int c_int = __float2int_rd(c - 0.5f);
    const int base_r = r_int - 1;
    const int base_c = c_int - 1;

    float row_weights[4], col_weights[4];
    int clamped_rows[4], clamped_cols[4];

    // Compute weights; unrolled for performance.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const float dr = r - (base_r + i + 0.5f);
        row_weights[i] = cubic_from_lut(dr, lut_constant);
        clamped_rows[i] = max(0, min(tile_height - 1, base_r + i));
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        const float dc = c - (base_c + j + 0.5f);
        col_weights[j] = cubic_from_lut(dc, lut_constant);
        clamped_cols[j] = max(0, min(tile_width - 1, base_c + j));
    }

    // if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("%f\n", col_weights[1]);
    // }

    float q = 0.0f;
    // Loop over columns
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int row_offset = clamped_rows[i] * tile_width;
        float p = 0.0f;
        // Loop over rows.
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Use FMA for each multiply-add, using the precomputed clamped indices.
            const float pixel = sub_image[row_offset + clamped_cols[j]];
            p = __fmaf_rn(pixel, col_weights[j], p);
        }
        q = __fmaf_rn(p, row_weights[i], q);
    }
    return q;
}

////////////////////////////////////////////////////////////////////////////////
// Shared memory tile configuration:
//   • Extra pixels on the top/bottom/left/right (for a 4x4 kernel, need 2 extra pixels)
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS 2

// Block dimensions.
#define BLOCK_WIDTH  16
#define BLOCK_HEIGHT 16

////////////////////////////////////////////////////////////////////////////////
// Shared-memory kernel using the same mapping as the original code.
////////////////////////////////////////////////////////////////////////////////
__global__ void shift_magnify_kernel(const float *__restrict__ image_in, float *__restrict__ image_out, int magnification,
                                       int rows, int cols,
                                       float shift_row, float shift_col,
                                       float inv_magnification_row, float inv_magnification_col) {
    // Compute output dimensions.
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);

    // Global output pixel coordinates.
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int i = blockIdx.y * blockDim.y + threadIdx.y;      // row index

    // Shared memory tile dimensions (with a 2-pixel halo on all sides)
    const int tile_width  = BLOCK_WIDTH + 2 * KERNEL_RADIUS;  // 16 + 4 = 20
    const int tile_height = BLOCK_HEIGHT + 2 * KERNEL_RADIUS; // 16 + 4 = 20

    // Declare the shared memory tile.
    __shared__ float tile[tile_height][tile_width];

    // Global coordinate for the top-left element of this tile (adjusted for smaller tile)
    int tile_origin_i = (blockIdx.y * blockDim.y) / magnification - KERNEL_RADIUS;
    int tile_origin_j = (blockIdx.x * blockDim.x) / magnification - KERNEL_RADIUS;

    // Load shared tile with halo, ensuring we don't go out of bounds
    int numTileElements = tile_width * tile_height;
    int numThreads = blockDim.x * blockDim.y;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    #pragma unroll
    for (int idx = threadId; idx < numTileElements; idx += numThreads) {
        int local_i = idx / tile_width;
        int local_j = idx % tile_width;
        int global_i = tile_origin_i + local_i;
        int global_j = tile_origin_j + local_j;

        if (global_i >= 0 && global_i < rows && global_j >= 0 && global_j < cols)
            tile[local_i][local_j] = __ldg(&image_in[global_i * cols + global_j]);
        else
            tile[local_i][local_j] = 0.0f; // Zero-padding for out-of-bounds values
    }

    __syncthreads();

    // if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("%f\n", lut_constant[1]);
    // }

    // // For debugging: print the tile from block (0,0) by thread (0,0).
    // if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("Tile values:\n");
    //     for (int ii = 0; ii < tile_height; ii++) {
    //         for (int jj = 0; jj < tile_width; jj++) {
    //             printf("%f ", tile[ii][jj]);
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    // Process valid output pixels.
    if (i < rowsM && j < colsM) {
        float row = (i * inv_magnification_row) - shift_row;
        float col = (j * inv_magnification_col) - shift_col;
        // Compute local coordinate in the tile.
        float local_row = row - tile_origin_i;
        float local_col = col - tile_origin_j;

        float val = interpolate_shared(&tile[0][0], local_row, local_col, tile_width, tile_height);

        image_out[i * colsM + j] = val;
        // Optional debug print for output pixel (0,0).
        // if (i == 0 && j == 15) {
        //     printf("Output[%d, %d]: row=%f, col=%f, local_row=%f, local_col=%f, val=%f\n", 
        //         (i*5 + int_step_row), ((j*5) + int_step_col), row, col, local_row, local_col, val);
        // }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host function that launches the kernel.
////////////////////////////////////////////////////////////////////////////////
extern "C" void shift_magnify(const float *image_in, float *image_out,
                              int rows, int cols, float shift_row, float shift_col,
                              float magnification_row, float magnification_col, cudaStream_t stream) {
    int rowsM = (int)(rows * magnification_row);
    int colsM = (int)(cols * magnification_col);
    float inv_magnification_row = 1.0f / magnification_row;
    float inv_magnification_col = 1.0f / magnification_col;
    int magnification = rowsM / rows;

    dim3 blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 gridSize((colsM + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
                  (rowsM + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);

    shift_magnify_kernel<<<gridSize, blockSize, 0, stream>>>(image_in, image_out, magnification,
                                                             rows, cols,
                                                             shift_row, shift_col,
                                                             inv_magnification_row, inv_magnification_col);
    cudaDeviceSynchronize();
}
