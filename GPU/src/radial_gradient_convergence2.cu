#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Device function to calculate distance weight
__device__ float calculate_dw(float distance, float inv_tSS) {
    float d2 = distance * distance;
    float d4 = d2 * d2;
    return d4 * __expf(inv_tSS * d2);
}

__device__ float calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
    float denom = Gx * Gx + Gy * Gy;
    if (denom > 0.0f) {
        float sqrt_denom = sqrtf(denom);
        float temp = fabsf(Gy * dx - Gx * dy);
        float Dk = temp / sqrt_denom;
        return 1.0f - Dk / distance;
    } else {
        return 0.0f;
    }
}

// Device function to calculate RGC
__device__ float calculate_rgc(int xM, int yM, const float*__restrict__ imIntGx, const float*__restrict__ imIntGy, 
                               int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, 
                               float fwhm, float tSO, float inv_tSS, float sensitivity) {
    // Precompute constants
    float xc = (xM + 0.5f) / magnification;
    float yc = (yM + 0.5f) / magnification;
    int Gx_Gy_xc = (int)(Gx_Gy_MAGNIFICATION * xc);
    int Gx_Gy_yc = (int)(Gx_Gy_MAGNIFICATION * yc);
    float Gx_Gy_inv = 1.0f / Gx_Gy_MAGNIFICATION;
    int mag_Gx_Gy = magnification * 2;
    int mag_Gx_Gy_2 = magnification * 2 * 2;

    // Initialize outputs
    float RGC = 0;
    float distanceWeightSum = 0;

    // Loop bounds
    int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
    int _end = (int)(Gx_Gy_MAGNIFICATION * fwhm + 1);

    #pragma unroll                            
    for (int j = _start; j < _end; j++) {
        float vy = (Gx_Gy_yc + j) * Gx_Gy_inv;
        if (0 < vy && vy <= rowsM - 1) {
            #pragma unroll
            for (int i = _start; i < _end; i++) {
                float vx = (Gx_Gy_xc + i) * Gx_Gy_inv;
                if (0 < vx && vx <= colsM - 1) {
                    float dx = vx - xc;
                    float dy = vy - yc;
                    float distance = sqrtf(dx * dx + dy * dy);

                    if (distance != 0 && distance <= tSO) {
                        int index = (int)(vy * mag_Gx_Gy_2 * colsM) + (int)(vx * mag_Gx_Gy);

                        float Gx = imIntGx[index];
                        float Gy = imIntGy[index];

                        float distanceWeight = calculate_dw(distance, inv_tSS);

                        distanceWeightSum += distanceWeight;
                        float GdotR = Gx * dx + Gy * dy;

                        if (GdotR < 0) {
                            float Dk = calculate_dk(Gx, Gy, dx, dy, distance);
                            RGC += Dk * distanceWeight;
                        }
                    }
                }
            }
        }
    }

    if (distanceWeightSum != 0) {
        RGC /= distanceWeightSum;
    }

    if (RGC >= 0 && sensitivity > 1) {
        RGC = sensitivity == 2.0f ? RGC * RGC : __powf(RGC, sensitivity);
    } else if (RGC < 0) {
        RGC = 0;
    }

    return RGC;
}

// CUDA kernel for radial gradient convergence
__global__ void radial_gradient_convergence_kernel(const float*__restrict__ gradient_col_interp, const float*__restrict__ gradient_row_interp, 
                                                   const float*__restrict__ image_interp, int rowsM, int colsM, int magnification, 
                                                   float radius, float sensitivity, int doIntensityWeighting, float *rgc_map,
                                                   float sigma, float fwhm, float inv_tSS, float tSO) {
    // Calculate 2D thread indices
    int cM = blockIdx.x * blockDim.x + threadIdx.x + magnification * 2; // Column index
    int rM = blockIdx.y * blockDim.y + threadIdx.y + magnification * 2; // Row index

    // Check if the thread is within the image bounds
    if (rM < rowsM - magnification * 2 && cM < colsM - magnification * 2) {
        // Precompute constants

        float Gx_Gy_MAGNIFICATION = 2.0f;

        // Calculate RGC value
        float rgc_value = calculate_rgc(cM, rM, gradient_col_interp, gradient_row_interp, colsM, rowsM, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, inv_tSS, sensitivity);

        // Apply intensity weighting if needed
        if (doIntensityWeighting) {
            rgc_map[rM * colsM + cM] = rgc_value * image_interp[rM * colsM + cM];
        } else {
            rgc_map[rM * colsM + cM] = rgc_value;
        }
    }
}

// Host function to call the CUDA kernel
extern "C" void radial_gradient_convergence(const float *gradient_col_interp, const float *gradient_row_interp, 
                                            const float *image_interp, int rowsM, int colsM, int magnification, 
                                            float radius, float sensitivity, int doIntensityWeighting, float *rgc_map) {
    // Define block and grid sizes
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((colsM - 2 * magnification + blockSize.x - 1) / blockSize.x, 
                  (rowsM - 2 * magnification + blockSize.y - 1) / blockSize.y);

    float sigma = radius / 2.355f;
    float fwhm = radius;
    float tSS = 2 * sigma * sigma;
    float inv_tSS = -4.0f / tSS;
    float tSO = 2 * sigma + 1;

    // Launch the CUDA kernel
    radial_gradient_convergence_kernel<<<gridSize, blockSize>>>(gradient_col_interp, gradient_row_interp, 
                                                                image_interp, rowsM, colsM, magnification, 
                                                                radius, sensitivity, doIntensityWeighting, rgc_map,
                                                                sigma, fwhm, inv_tSS, tSO);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();
}