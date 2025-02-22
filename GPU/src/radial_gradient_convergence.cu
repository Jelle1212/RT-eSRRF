#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Device function to calculate distance weight
__device__ float calculate_dw(float distance, float tSS) {
    float dist_sq = distance * distance;
    return __powf((distance * __expf(-dist_sq / tSS)), 4);
}

// Device function to calculate Dk
__device__ float calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
    float Dk = fabsf(Gy * dx - Gx * dy) / sqrtf(Gx * Gx + Gy * Gy);
    if (isnan(Dk)) {
        Dk = distance;
    }
    return 1 - Dk / distance;
}

// Device function to calculate RGC
__device__ float calculate_rgc(int xM, int yM, const float* imIntGx, const float* imIntGy, 
                               int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, 
                               float fwhm, float tSO, float tSS, float sensitivity) {
    float xc = (xM + 0.5f) / magnification;
    float yc = (yM + 0.5f) / magnification;

    float RGC = 0;
    float distanceWeightSum = 0;

    int _start = -(int)(Gx_Gy_MAGNIFICATION * fwhm);
    int _end = (int)(Gx_Gy_MAGNIFICATION * fwhm + 1);

    for (int j = _start; j < _end; j++) {
        float vy = (int)(Gx_Gy_MAGNIFICATION * yc) + j;
        vy /= Gx_Gy_MAGNIFICATION;

        if (0 < vy && vy <= rowsM - 1) {
            for (int i = _start; i < _end; i++) {
                float vx = (int)(Gx_Gy_MAGNIFICATION * xc) + i;
                vx /= Gx_Gy_MAGNIFICATION;

                if (0 < vx && vx <= colsM - 1) {
                    float dx = vx - xc;
                    float dy = vy - yc;
                    float distance = sqrtf(dx * dx + dy * dy);

                    if (distance != 0 && distance <= tSO) {
                        int index = (int)(vy * magnification * Gx_Gy_MAGNIFICATION * colsM * Gx_Gy_MAGNIFICATION) + (int)(vx * magnification * Gx_Gy_MAGNIFICATION);
                        float Gx = imIntGx[index];
                        float Gy = imIntGy[index];

                        float distanceWeight = calculate_dw(distance, tSS);
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
        RGC = __powf(RGC, sensitivity);
    } else if (RGC < 0) {
        RGC = 0;
    }

    return RGC;
}

// CUDA kernel for radial gradient convergence
__global__ void radial_gradient_convergence_kernel(const float *gradient_col_interp, const float *gradient_row_interp, 
                                                   const float *image_interp, int rowsM, int colsM, int magnification, 
                                                   float radius, float sensitivity, int doIntensityWeighting, float *rgc_map) {
    // Calculate 2D thread indices
    int cM = blockIdx.x * blockDim.x + threadIdx.x + magnification * 2; // Column index
    int rM = blockIdx.y * blockDim.y + threadIdx.y + magnification * 2; // Row index

    // Check if the thread is within the image bounds
    if (rM < rowsM - magnification * 2 && cM < colsM - magnification * 2) {
        // Precompute constants
        float sigma = radius / 2.355f;
        float fwhm = radius;
        float tSS = 2 * sigma * sigma;
        float tSO = 2 * sigma + 1;
        float Gx_Gy_MAGNIFICATION = 2.0f;

        // Calculate RGC value
        float rgc_value = calculate_rgc(cM, rM, gradient_col_interp, gradient_row_interp, colsM, rowsM, magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, sensitivity);

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

    // Launch the CUDA kernel
    radial_gradient_convergence_kernel<<<gridSize, blockSize>>>(gradient_col_interp, gradient_row_interp, 
                                                                image_interp, rowsM, colsM, magnification, 
                                                                radius, sensitivity, doIntensityWeighting, rgc_map);
}