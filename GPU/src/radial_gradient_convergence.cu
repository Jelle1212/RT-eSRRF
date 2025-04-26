#include "radial_gradient_convergence.hu"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Device function to calculate distance weight
__device__ float calculate_dw(float distance, float tSS) {
    float d2 = distance * distance;
    float d4 = d2 * d2;
    return d4 * __expf(-4.0f * d2 / tSS);
}

__device__ float calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
    float Dk = fabsf(Gy * dx - Gx * dy) / sqrtf(Gx * Gx + Gy * Gy);
    if (isnan(Dk)) {
        Dk = distance;
    }
    return 1 - Dk / distance;
}

struct TileMemory {
    float* distance_weight;
    float* rgc;
};

__device__ TileMemory get_tile_memory(float* shared_mem, int tile_size) {
    TileMemory tm;
    tm.distance_weight = shared_mem;
    tm.rgc = shared_mem + tile_size;
    return tm;
}

// CUDA kernel for radial gradient convergence
__global__ void radial_gradient_convergence_kernel(const float *gradient_col_interp, const float *gradient_row_interp, 
                                                   const float *image_interp, int rowsM, int colsM, int magnification, 
                                                   float fwhm, float tSO, float tSS, float sensitivity, 
                                                   float Gx_Gy_MAGNIFICATION, int doIntensityWeighting, float *rgc_map) {
    extern __shared__ float shared_mem[];

    int tile_size = blockDim.x * blockDim.y;
    TileMemory tile = get_tile_memory(shared_mem, tile_size);
                                            
    int diameter = blockDim.x; // assuming square block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int local_idx = ty * diameter + tx;

    // Use blockIdx.x / y to determine the output pixel
    int cM = blockIdx.x + 2 * magnification;
    int rM = blockIdx.y + 2 * magnification;

    // Pixels of interest in the gradient vector
    float xc = (cM + 0.5f) / magnification;
    float yc = (rM + 0.5f) / magnification;

    // Absolute offset 
    int x_abs_offset = d_offsets[tx];
    int y_abs_offset = d_offsets[ty];

    // Coordinates super res
    float vy = (int)(Gx_Gy_MAGNIFICATION * yc) + y_abs_offset;
    vy /= Gx_Gy_MAGNIFICATION;
    float vx = (int)(Gx_Gy_MAGNIFICATION * xc) + x_abs_offset;
    vx /= Gx_Gy_MAGNIFICATION;

    // Setting tile file to 0
    tile.rgc[local_idx] = 0.0f;
    tile.distance_weight[local_idx] = 0.0f;

    int index = (int) (vy * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)(vx * magnification * Gx_Gy_MAGNIFICATION);

    float RGC = 0.0f;


    if (rM < rowsM - magnification * 2 && cM < colsM - magnification * 2) {
        if (0 < vy && vy <= rowsM - 1 && 0 < vx && vx <= colsM - 1) {
            float dx = vx - xc;
            float dy = vy - yc;

            float distance = sqrtf(dx*dx + dy*dy);

            if (distance != 0 && distance <= tSO) {
                float distanceWeight = calculate_dw(distance, tSS);

                tile.distance_weight[local_idx] = distanceWeight;

                float Gx = gradient_col_interp[index];
                float Gy = gradient_row_interp[index];

                float GdotR = Gx * dx + Gy * dy;

                float Dk = GdotR < 0 ? calculate_dk(Gx, Gy, dx, dy, distance) : 0.0f;

                RGC = Dk * distanceWeight;
            }
        }

        // store in shared memory
        tile.rgc[local_idx] = RGC; // reusing the distance_vector buffer to save space

        __syncthreads();

        // Write to output map from thread 0
        if (local_idx == 0) {
            float RGCSum = 0.0f;
            float distanceWeightSum = 0.0f;

            for (int i = 0; i < tile_size; i++) {
                RGCSum += tile.rgc[i];
                distanceWeightSum += tile.distance_weight[i];
            }

            if (distanceWeightSum != 0) {
                RGCSum /= distanceWeightSum;
            }

            if (RGCSum >= 0 && sensitivity > 1) {
                RGCSum = __powf(RGCSum, sensitivity);
            } else if (RGCSum < 0) {
                RGCSum = 0;
            }

            float result = doIntensityWeighting ? RGCSum * image_interp[rM * colsM + cM] : RGCSum;
            rgc_map[rM * colsM + cM] = result;
        }
    }
}

extern "C" void radial_gradient_convergence(const float *gradient_col_interp, const float *gradient_row_interp, 
                                            const float *image_interp, int rowsM, int colsM, int magnification, 
                                            float radius, float sensitivity, int doIntensityWeighting, float *rgc_map) {
    // Precompute constants on CPU
    float sigma = radius / 2.355f;
    float fwhm = radius;
    float tSS = 2 * sigma * sigma;
    float tSO = 2 * sigma + 1;
    float Gx_Gy_MAGNIFICATION = 2.0f;

    int diameter = (int) 2 * Gx_Gy_MAGNIFICATION * fwhm + 1;
    size_t shared_mem_bytes = 2 * diameter * diameter * sizeof(float);

    // Define block and grid sizes
    dim3 blockSize(diameter, diameter);
    dim3 gridSize(colsM, rowsM);

    int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
    int _stop  =  (int)(Gx_Gy_MAGNIFICATION * fwhm);

    int* h_offsets = new int[diameter];
    for (int i = 0; i < diameter; ++i) {
        h_offsets[i] = _start + i;
    }

    cudaMemcpyToSymbol(d_offsets, h_offsets, diameter * sizeof(int));


    // Launch the CUDA kernel
    radial_gradient_convergence_kernel<<<gridSize, blockSize, shared_mem_bytes>>>(gradient_col_interp, gradient_row_interp, 
                                                                image_interp, rowsM, colsM, magnification, 
                                                                fwhm, tSO, tSS, sensitivity, Gx_Gy_MAGNIFICATION, 
                                                                doIntensityWeighting, rgc_map);

    delete h_offsets;
}