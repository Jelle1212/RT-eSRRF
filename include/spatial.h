#ifndef SPATIAL_H
#define SPATIAL_H

#include <stdio.h>
#include <stdbool.h>

void spatial(const float *image_in, int nFrames, int rows, int cols, 
             float shift, float magnification, float radius, 
             float sensitivity, bool doIntensityWeighting, float *rgc_map);
#endif
