#ifndef ROBERTS_CROSS_GRADIENTS_H
#define ROBERTS_CROSS_GRADIENTS_H

void roberts_cross_gradients(const float *image, float *gradient_col, float *gradient_row, 
                              int nFrames, int rows, int cols);
void gradient(const float* image, float* imGc, float* imGr, int rows, int cols);

#endif