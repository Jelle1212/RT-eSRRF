#include "spatial.h"
#include "shift_magnify.h"
#include "roberts_cross_gradients.h"
#include "radial_gradient_convergence.h"
#include <stdio.h>
#include <stdlib.h>

float* spatial(const float *image_in, int nFrames, int rows, int cols, 
             float shift, float magnification, float radius, 
             float sensitivity, bool doIntensityWeighting) {
    
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);
    float *magnified_image = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));

    float *gradient_col = (float *)malloc(nFrames * rows * cols * sizeof(float));
    float *gradient_row = (float *)malloc(nFrames * rows * cols * sizeof(float));

    float *gradient_col_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));
    float *gradient_row_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));

    float *rgc_map = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));

    // Call the shift_magnify function to apply shift and magnification
    shift_magnify(image_in, magnified_image, nFrames, rows, cols, shift, shift, magnification, magnification);
    roberts_cross_gradients(image_in, gradient_col, gradient_row, nFrames, rows, cols);
    shift_magnify(gradient_col, gradient_col_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);
    shift_magnify(gradient_row, gradient_row_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);
    radial_gradient_convergence(gradient_col_interp, gradient_row_interp, magnified_image, nFrames, rowsM, colsM, magnification, radius, sensitivity, doIntensityWeighting, rgc_map);

    // Free allocated memory
    free(magnified_image);
    free(gradient_col);
    free(gradient_row);
    free(gradient_col_interp);
    free(gradient_row_interp);

    return rgc_map;
}
