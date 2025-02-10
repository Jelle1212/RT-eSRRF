#include "spatial.h"
#include "shift_magnify.h"
#include "roberts_cross_gradients.h"
#include "radial_gradient_convergence.h"
#include <stdio.h>
#include <stdlib.h>

void spatial(const float *image_in, int nFrames, int rows, int cols, 
             float shift, float magnification, float radius, 
             float sensitivity, bool doIntensityWeighting, float *rgc_map) {
    
    // Calculate new dimensions based on magnification
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);

    // Allocate memory for intermediate images
    float *magnified_image = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));
    float *gradient_col = (float *)malloc(nFrames * rows * cols * sizeof(float));
    float *gradient_row = (float *)malloc(nFrames * rows * cols * sizeof(float));
    float *gradient_col_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));
    float *gradient_row_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));

    if (!magnified_image || !gradient_col || !gradient_row || !gradient_col_interp || !gradient_row_interp) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        free(magnified_image);
        free(gradient_col);
        free(gradient_row);
        free(gradient_col_interp);
        free(gradient_row_interp);
        return;
    }

    // Apply shift and magnification
    shift_magnify(image_in, rgc_map, nFrames, rows, cols, shift, shift, magnification, magnification);

    // Compute Roberts Cross Gradients
    roberts_cross_gradients(image_in, gradient_col, gradient_row, nFrames, rows, cols);

    // Apply shift and magnification to gradients
    shift_magnify(gradient_col, gradient_col_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);
    shift_magnify(gradient_row, gradient_row_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);

    // Compute radial gradient convergence
    radial_gradient_convergence(gradient_col_interp, gradient_row_interp, magnified_image, nFrames, rowsM, colsM, magnification, radius, sensitivity, doIntensityWeighting, rgc_map);

    // Free allocated memory
    free(magnified_image);
    free(gradient_col);
    free(gradient_row);
    free(gradient_col_interp);
    free(gradient_row_interp);
}

