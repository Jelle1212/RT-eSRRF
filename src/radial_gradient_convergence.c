#include "radial_gradient_convergence.h"
#include <math.h>

// Inline small functions to reduce overhead
float calculate_dw(float distance, float tSS) {
    float dist_sq = distance * distance;
    return pow((distance * exp(-dist_sq / tSS)), 4);
}

float calculate_dk(float Gx, float Gy, float dx, float dy, float distance) {
    float Dk = fabs(Gy * dx - Gx * dy) / sqrt(Gx * Gx + Gy * Gy);
    if (isnan(Dk)) {
        Dk = distance;
    }
    return 1 - Dk / distance;
}

float calculate_rgc(int xM, int yM, const float* imIntGx, const float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity) {
    float xc = (xM + 0.5) / magnification;
    float yc = (yM + 0.5) / magnification;

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
                    float distance = sqrt(dx * dx + dy * dy);

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
        RGC = pow(RGC, sensitivity);
    } else if (RGC < 0) {
        RGC = 0;
    }

    return RGC;
}

void radial_gradient_convergence(const float *gradient_col_interp, const float *gradient_row_interp, const float *image_interp, 
                                  int rowsM, int colsM, int magnification, float radius, float sensitivity, 
                                  int doIntensityWeighting, float *rgc_map) {
    float sigma = radius / 2.355;
    float fwhm = radius;
    float tSS = 2 * sigma * sigma;
    float tSO = 2 * sigma + 1;
    float Gx_Gy_MAGNIFICATION = 2.0;

    int _magnification = magnification;
    float _sensitivity = sensitivity;
    int _doIntensityWeighting = doIntensityWeighting;

    // Loop over rows and columns
    for (int rM = _magnification * 2; rM < rowsM - _magnification * 2; rM++) {
        for (int cM = _magnification * 2; cM < colsM - _magnification * 2; cM++) {
            float rgc_value = calculate_rgc(cM, rM, gradient_col_interp, gradient_row_interp, colsM, rowsM, _magnification, Gx_Gy_MAGNIFICATION, fwhm, tSO, tSS, _sensitivity);
            if (_doIntensityWeighting) {
                rgc_map[rM * colsM + cM] = rgc_value * image_interp[rM * colsM + cM];
            } else {
                rgc_map[rM * colsM + cM] = rgc_value;
            }
        }
    }
}