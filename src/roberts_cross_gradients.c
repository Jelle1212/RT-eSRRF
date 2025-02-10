#include "roberts_cross_gradients.h"

void roberts_cross_gradients(const float *image, float *gradient_col, float *gradient_row, 
                              int nFrames, int rows, int cols) {
    for (int n = 0; n < nFrames; n++) {
        gradient(&image[n * rows * cols], &gradient_col[n * rows * cols], &gradient_row[n * rows * cols], rows, cols);
    }
}

void gradient(const float* image, float* imGc, float* imGr, int rows, int cols) {
    int c1, r1, c0, r0;
    float im_c0_r1, im_c1_r0, im_c0_r0, im_c1_r1;

    for (r1 = 0; r1 < rows; r1++) {
        for (c1 = 0; c1 < cols; c1++) {

            c0 = c1 > 0 ? c1 - 1 : 0;
            r0 = r1 > 0 ? r1 - 1 : 0;

            im_c0_r1 = image[r0 * cols + c1];
            im_c1_r0 = image[r1 * cols + c0];
            im_c0_r0 = image[r0 * cols + c0];
            im_c1_r1 = image[r1 * cols + c1];

            imGc[r1 * cols + c1] = im_c0_r1 - im_c1_r0 + im_c1_r1 - im_c0_r0;
            imGr[r1 * cols + c1] = -im_c0_r1 + im_c1_r0 + im_c1_r1 - im_c0_r0;
        }
    }
}