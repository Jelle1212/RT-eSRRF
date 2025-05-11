// #include <cuda_runtime.h>
// #include <math.h>
// #include <stdio.h>

// #define BLOCK_SIZE 8

// // Device function to calculate distance weight
// __device__ float calculate_dw(float distance, float tSS) {
//     float d2 = distance * distance;
//     float d4 = d2 * d2;
//     return d4 * __expf(-4.0f * d2 / tSS);
// }

// __device__ float calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
//     float Dk = fabsf(Gy * dx - Gx * dy) / sqrtf(Gx * Gx + Gy * Gy);
//     if (isnan(Dk)) {
//         Dk = distance;
//     }
//     return 1 - Dk / distance;
// }

// // Device function to calculate RGC
// __device__ float calculate_rgc(int xM, int yM, const float* imIntGx, const float* imIntGy,
//                                const float* imIntGxTest, const float* imIntGyTest, int nr_steps, int x_start,  int y_start,
//                                int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, 
//                                float fwhm, float tSO, float tSS, float sensitivity) {
//     float xc = (xM + 0.5f) / magnification;
//     float yc = (yM + 0.5f) / magnification;

//     float RGC = 0;
//     float distanceWeightSum = 0;

//     int diameter = (int) 2 * Gx_Gy_MAGNIFICATION * fwhm + 1;

//     int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
//     int _end = (int)(Gx_Gy_MAGNIFICATION * fwhm + 1);

//     for (int j = _start; j < _end; j++) {
//         float vy = (int)(Gx_Gy_MAGNIFICATION * yc) + j;
//         vy /= Gx_Gy_MAGNIFICATION;

//         if (0 < vy && vy <= rowsM - 1) {
//             for (int i = _start; i < _end; i++) {
//                 float vx = (int)(Gx_Gy_MAGNIFICATION * xc) + i;
//                 vx /= Gx_Gy_MAGNIFICATION;

//                 if (0 < vx && vx <= colsM - 1) {
//                     float dx = vx - xc;
//                     float dy = vy - yc;
//                     float distance = sqrtf(dx * dx + dy * dy);

//                     if (distance != 0 && distance <= tSO) {
//                         int y = (int)(vy * magnification * Gx_Gy_MAGNIFICATION);
//                         int x = (int)(vx * magnification * Gx_Gy_MAGNIFICATION);

//                         int local_x = (x - x_start) / magnification;
//                         int local_y = (y - y_start) / magnification;

//                         int local_idx = local_y * nr_steps + local_x;

//                         // float Gx_local = imIntGx[local_y * nr_steps + local_x];

//                         // int index = (int)(vy * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)(vx * magnification * Gx_Gy_MAGNIFICATION); 
//                         // float Gx_global = imIntGxTest[index];
//                         // if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0 && threadIdx.y == 5) {
//                         //     printf("x: %d, y: %d, l_x: %d, l_y: %d, i: %d, j: %d, val_global: %f, val_local: %f\n", x, y, local_x, local_y, i, j, Gx_global, Gx_local);
//                         //     // printf("local_x: %d, local_y: %d\n", local_x, local_y);
//                         //     // printf("local_idx: %d\n", local_idx);
//                         // }

//                         // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 1 && blockIdx.y == 0) {
//                         //     printf("x: %d, y: %d, j: %d, i: %d, threadidx.x: %d, threadidx.y: %d, xM: %d, yM: %d\n", x, y, j, i, threadIdx.x, threadIdx.y, xM, yM);
//                         //     // printf("local_x: %d, local_y: %d\n", local_x, local_y);
//                         //     // printf("local_idx: %d\n", local_idx);
//                         // }
//                         // if ((threadIdx.x == 7 | threadIdx.x == 0) && threadIdx.y == 0 && blockIdx.x == 6 && blockIdx.y == 0) {
//                         //     if ((i == -4 | i == -3)  && threadIdx.x == 0) {
//                         //         printf("x: %d, i: %d, threadidx.x: %d, xM: %d\n", x, i, threadIdx.x, threadIdx.y, xM);
//                         //     }
//                         //     if (i == 4 && threadIdx.x == 7) {
//                         //         printf("x: %d, i: %d, threadidx.x: %d, xM: %d\n", x, i, threadIdx.x, threadIdx.y, xM);
//                         //     }
//                         //     // printf("local_x: %d, local_y: %d\n", local_x, local_y);
//                         //     // printf("local_idx: %d\n", local_idx);
//                         // }

//                         // if (local_idx >= 12*12) {
//                         //     printf("thread.x=%d, thread.y=%d, blockidx=%d blockidy=%d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
//                         // }

//                         float Gx = imIntGx[local_idx];
//                         float Gy = imIntGy[local_idx];

//                         float distanceWeight = calculate_dw(distance, tSS);

//                         distanceWeightSum += distanceWeight;
//                         float GdotR = Gx * dx + Gy * dy;

//                         if (GdotR < 0) {
//                             float Dk = calculate_dk(Gx, Gy, dx, dy, distance);
//                             RGC += Dk * distanceWeight;
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     if (distanceWeightSum != 0) {
//         RGC /= distanceWeightSum;
//     }

//     if (RGC >= 0 && sensitivity > 1) {
//         RGC = __powf(RGC, sensitivity);
//     } else if (RGC < 0) {
//         RGC = 0;
//     }

//     return RGC;
// }

// struct TileMemory {
//     float* gradient_col;
//     float* gradient_row;
// };

// __device__ TileMemory get_tile_memory(float* shared_mem, int tile_size) {
//     TileMemory tm;
//     tm.gradient_col = shared_mem;
//     tm.gradient_row = shared_mem + tile_size;
//     return tm;
// }

// __constant__ int d_offsets[100];

// __device__ int calculate_gradient_coordinate(int xm, int i, int magnification) {
//     return ((int)((2 * xm + 1) / magnification) + i) * magnification;
// }


// // CUDA kernel for radial gradient convergence
// __global__ void radial_gradient_convergence_kernel(const float *__restrict__ gradient_col_interp, const float *__restrict__ gradient_row_interp, 
//                                                    const float *__restrict__ image_interp, int rowsM, int colsM, int magnification, 
//                                                    float radius, float sensitivity, int doIntensityWeighting, float *rgc_map) {
//     // Calculate 2D thread indices
//     int cM = blockIdx.x * blockDim.x + threadIdx.x + magnification * 2; // Column index
//     int rM = blockIdx.y * blockDim.y + threadIdx.y + magnification * 2; // Row index

//     extern __shared__ float shared_mem[];

//     // Precompute constants
//     float sigma = radius / 2.355f;
//     float fwhm = radius;
//     float tSS = 2 * sigma * sigma;
//     float tSO = 2 * sigma + 1;
//     float Gx_Gy_MAGNIFICATION = 2.0f;

//     int _start = (int) fwhm * Gx_Gy_MAGNIFICATION;

//     // Put gradient col and row on shared memory
//     int startX = blockIdx.x * blockDim.x + magnification * 2; // first thread
//     int startY = blockIdx.y * blockDim.y + magnification * 2; // first thread

//     int end = blockIdx.x * blockDim.x + (blockDim.x-1) + magnification * 2; // last thread of block

//     int x_start = calculate_gradient_coordinate(startX, -_start, magnification);
//     int y_start = calculate_gradient_coordinate(startY, -_start, magnification);
//     int x_stop = calculate_gradient_coordinate(end, _start, magnification);
    
//     int nr_steps = ((x_stop - x_start) / magnification) + 1;

//     int offset_x = calculate_gradient_coordinate(blockIdx.x * blockDim.x + magnification * 2, -_start, magnification);
//     int offset_y = calculate_gradient_coordinate(blockIdx.y * blockDim.y + magnification * 2, -_start, magnification);

//     int tile_size = nr_steps*nr_steps;
//     TileMemory tile = get_tile_memory(shared_mem, tile_size);

//     // Check if the thread is within the image bounds
//     if (rM < rowsM - magnification * 2 && cM < colsM - magnification * 2) {

//         // Load tile in shared memory
//         for (int j = threadIdx.y; j < nr_steps; j++) {
//             for (int i = threadIdx.x; i < nr_steps; i++) {
            
//                 int x = offset_x + i * magnification;
//                 int y = offset_y + j * magnification;

//                 int local_idx = j * nr_steps + i;
//                 // if (blockIdx.x == 40 && blockIdx.y == 0) {
//                 //     printf("x: %d, y: %d, xM: %d, yM: %d, steps: %d, offset_x: %d, offset_y: %d, x_start: %d, x_stop: %d\n", x, y, cM, rM, nr_steps, offset_x, offset_y, x_start, x_stop);
//                 //     // printf("local_x: %d, local_y: %d\n", local_x, local_y);
//                 //     // printf("local_idx: %d\n", local_idx);
//                 // }

//                 if ((0 < x && x <= (2 * colsM) - 1) && (0 < y && y <= (2 * rowsM) - 1)) {
//                     tile.gradient_col[local_idx] = __ldg(&gradient_col_interp[(int)(y *colsM * Gx_Gy_MAGNIFICATION + x)]);
//                     tile.gradient_row[local_idx] = __ldg(&gradient_row_interp[(int)(y *colsM * Gx_Gy_MAGNIFICATION + x)]);
//                 } else {
//                     tile.gradient_col[local_idx] = 0.0f;
//                     tile.gradient_row[local_idx] = 0.0f;
//                 }
//             }
//         }

//         __syncthreads(); // Make sure all threads finished copying
        
//         // Calculate RGC value
//         float rgc_value = calculate_rgc(cM, rM, tile.gradient_col, tile.gradient_row, gradient_col_interp, gradient_row_interp, nr_steps, x_start, y_start, colsM, rowsM, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity);
//         // float rgc_value = 0.0f;

//         // Apply intensity weighting if needed
//         if (doIntensityWeighting) {
//             rgc_map[rM * colsM + cM] = rgc_value * image_interp[rM * colsM + cM];
//         } else {
//             rgc_map[rM * colsM + cM] = rgc_value;
//         }
//     }
// }

// #define BLOCK_SIZE 8

// // Host function to call the CUDA kernel
// extern "C" void radial_gradient_convergence(const float *gradient_col_interp, const float *gradient_row_interp, 
//                                             const float *image_interp, int rowsM, int colsM, int magnification, 
//                                             float radius, float sensitivity, int doIntensityWeighting, float *rgc_map) {
//     // Define block and grid sizes
//     dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
//     dim3 gridSize((colsM - 2 * magnification + blockSize.x - 1) / blockSize.x, 
//                   (rowsM - 2 * magnification + blockSize.y - 1) / blockSize.y);

//     float fwhm = radius;                                 
//     float Gx_Gy_MAGNIFICATION = 2.0f;
//     int diameter = (int) 2 * Gx_Gy_MAGNIFICATION * fwhm;
//     int pattern_width = ((2 * BLOCK_SIZE + magnification - 1) / magnification) + 1;

//     // Determine shared memory
//     size_t shared_mem_bytes = 2 * (diameter + pattern_width) * sizeof(float);
//     // int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);

//     // cudaMemcpyToSymbol(d_offsets, h_offsets, diameter * sizeof(int));
//     // Launch the CUDA kernel
//     radial_gradient_convergence_kernel<<<gridSize, blockSize, shared_mem_bytes>>>(gradient_col_interp, gradient_row_interp, 
//                                                                 image_interp, rowsM, colsM, magnification, 
//                                                                 radius, sensitivity, doIntensityWeighting, rgc_map);

//     // Synchronize to ensure kernel execution is complete
//     cudaDeviceSynchronize();
// }