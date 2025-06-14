#ifndef TEMPORAL_H
#define TEMPORAL_H

float* temporal(const float* image_stack, int type, int frames, int rows, int cols);
void average(const float* image_stack, float* image_out, int frames, int rows, int cols);
void variance(const float* image_stack, float* image_out, float* mean_image, int frames, int rows, int cols);
void temporal_auto_correlation(const float* image_stack, float* image_out, float* mean_image, int frames, int rows, int cols);
#endif
