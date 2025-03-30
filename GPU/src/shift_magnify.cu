#include "shift_magnify.hu"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

////////////////////////////////////////////////////////////////////////////////
// Cubic convolution kernel (using a = 0.5)
////////////////////////////////////////////////////////////////////////////////
__device__ float cubic(float v) {
    const float a = 0.5f;
    float v_abs = fabsf(v);
    float v2 = v_abs * v_abs;
    float v3 = v2 * v_abs;
    return (v_abs < 1.0f) * (v3 * (-a + 2.0f) + v2 * (a - 3.0f) + 1.0f) +
           (v_abs >= 1.0f && v_abs < 2.0f) * (v3 * (-a) + v2 * (5.0f * a) - v_abs * (8.0f * a) + (4.0f * a));
}

////////////////////////////////////////////////////////////////////////////////
// Shared-memory interpolation (mimicking the original function)
// Note: 'tile' is a subset of the global image.
////////////////////////////////////////////////////////////////////////////////
__device__ float interpolate_shared(const float *sub_image, float r_global, float c_global, float r_local, float c_local, int tile_width, int tile_height) {
    // Check bounds on the shared tile.
    if (r_local < 0 || r_local >= tile_height || c_local < 0 || c_local >= tile_width)
        return 0.0f;

    // Replicate original mapping: use a half-pixel shift inside the interpolation.
    const int r_int_global = __float2int_rd(r_global - 0.5f);
    const int c_int_global = __float2int_rd(c_global - 0.5f);

    const int r_int_local = __float2int_rd(r_local - 0.5f);
    const int c_int_local = __float2int_rd(c_local - 0.5f);

    float row_weights[4], col_weights[4];

    #pragma unroll
    for (int i = 0; i < 4; i++)
        row_weights[i] = cubic(r_global - (r_int_global - 1 + i + 0.5f));
    #pragma unroll
    for (int j = 0; j < 4; j++)
        col_weights[j] = cubic(c_global - (c_int_global - 1 + j + 0.5f));

    // if (blockIdx.x == 1 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("Weights row: ");
    //     for (int i = 0; i < 4; i++) printf("%f ", row_weights[i]);
    //     printf("\n");
    // }

    float q = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int c_idx = c_int_local - 1 + j;
        if (c_idx < 0 || c_idx >= tile_width)
            continue;
        float p = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r_idx = r_int_local - 1 + i;
            if (r_idx < 0 || r_idx >= tile_height)
                continue;
            p += sub_image[r_idx * tile_width + c_idx] * row_weights[i];
        }
        q += p * col_weights[j];
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
__global__ void shift_magnify_kernel(const float *image_in, float *image_out, int magnification,
                                       int rows, int cols,
                                       float shift_row, float shift_col,
                                       float inv_magnification_row, float inv_magnification_col) {
    // Compute output dimensions.
    int rowsM = (int)(rows / inv_magnification_row);
    int colsM = (int)(cols / inv_magnification_col);

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
    for (int idx = threadId; idx < numTileElements; idx += numThreads) {
        int local_i = idx / tile_width;
        int local_j = idx % tile_width;
        int global_i = tile_origin_i + local_i;
        int global_j = tile_origin_j + local_j;

        if (global_i >= 0 && global_i < rows && global_j >= 0 && global_j < cols)
            tile[local_i][local_j] = image_in[global_i * cols + global_j];
        else
            tile[local_i][local_j] = 0.0f; // Zero-padding for out-of-bounds values
    }
    __syncthreads();

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

        float val = interpolate_shared(&tile[0][0], row, col, local_row, local_col, tile_width, tile_height);

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
