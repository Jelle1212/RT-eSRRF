#ifndef RADIAL_GRADIENT_CONVERGENCE_H
#define RADIAL_GRADIENT_CONVERGENCE_H

float calculate_dw(float distance, float tSS);
float calculate_dk(float Gx, float Gy, float dx, float dy, float distance);
float calculate_rgc(int xM, int yM, const float* imIntGx, const float* imIntGy, int colsM, int rowsM, int magnification, float Gx_Gy_MAGNIFICATION, float fwhm, float tSO, float tSS, float sensitivity);
void radial_gradient_convergence(const float *gradient_col_interp, const float *gradient_row_interp, const float *image_interp, int rowsM, int colsM, int magnification, float radius, float sensitivity, int doIntensityWeighting, float *rgc_map); 

#endif