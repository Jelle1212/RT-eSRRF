#ifndef TEMPORAL_H
#define TEMPORAL_H

float* temporal(const float* image_stack, int frames, int rows, int cols);
void average(const float* image_stack, float* image_stack_out, int frames, int rows, int cols);
void variance(const float* image_stack, float* image_stack_out, int frames, int rows, int cols);
void temporal_auto_correlation(const float* image_stack, float* image_stack_out, int frames, int rows, int cols);
#endif
