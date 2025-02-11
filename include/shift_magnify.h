#ifndef SHIFT_MAGNIFY_H
#define SHIFT_MAGNIFY_H

float cubic(float v);
float interpolate(const float *image, float r, float c, int rows, int cols);
void shift_magnify(const float *image_in, float *image_out, 
                   int rows, int cols, 
                   float shift_row, float shift_col, 
                   float magnification_row, float magnification_col);
#endif