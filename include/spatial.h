#ifndef SPATIAL_H
#define SPATIAL_H

#include <stdio.h>
#include <stdbool.h>

float* spatial(const float *image_in, int rows, int cols, 
             float shift, float magnification, float radius, 
             float sensitivity, bool doIntensityWeighting);
#endif
