#include "spatial.h"
#include "shift_magnify.h"
#include "roberts_cross_gradients.h"
#include "radial_gradient_convergence.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_ROWS    (125 * 5)   
#define MAX_COLS    (106 * 5)
#define MAX_INPUT_ROWS  125
#define MAX_INPUT_COLS  106

float* spatial(const float *image_in, int rows, int cols, 
             float shift, float magnification, float radius, 
             float sensitivity, bool doIntensityWeighting) {
    
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);

    static float magnified_image[MAX_ROWS * MAX_COLS];
    static float gradient_col[MAX_INPUT_ROWS * MAX_INPUT_COLS];
    static float gradient_row[MAX_INPUT_ROWS * MAX_INPUT_COLS];

    static float gradient_col_interp[2 * MAX_ROWS * 2 * MAX_COLS];
    static float gradient_row_interp[2 * MAX_ROWS * 2 * MAX_COLS];

    static float rgc_map[MAX_ROWS * MAX_COLS];

    // Call the shift_magnify function to apply shift and magnification
    shift_magnify(image_in, magnified_image, rows, cols, shift, shift, magnification, magnification);
    roberts_cross_gradients(image_in, gradient_col, gradient_row, rows, cols);
    shift_magnify(gradient_col, gradient_col_interp, rows, cols, shift, shift, magnification * 2, magnification * 2);
    shift_magnify(gradient_row, gradient_row_interp, rows, cols, shift, shift, magnification * 2, magnification * 2);
    radial_gradient_convergence(gradient_col_interp, gradient_row_interp, magnified_image, rowsM, colsM, magnification, radius, sensitivity, doIntensityWeighting, rgc_map);

    return rgc_map;
}
