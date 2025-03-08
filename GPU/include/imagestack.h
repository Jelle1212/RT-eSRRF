#ifndef IMAGESTACK_H
#define IMAGESTACK_H

#include <stdint.h>

// Structure to represent an ImageStack
typedef struct {
    float** stack;  // Array of pointers to images
    int stack_size;
    int capacity;
    int rows;
    int cols;
} ImageStack;

// Function declarations
ImageStack* createImageStack(int rows, int cols);
void pushImage(ImageStack* stack, float* image);
float* popImage(ImageStack* stack);
void saveAsTiffFloat32(const char* filename, ImageStack* stack);
void saveAsTiff16(const char* filename, ImageStack* stack);
void freeImageStack(ImageStack* stack);

// TIFF saving functions
void save_tiff_float32(const char* filename, const float* image, uint32_t width, uint32_t height, int nFrames);
void save_tiff(const char* filename, const float* image, uint32_t width, uint32_t height, int nFrames);

#endif  // IMAGESTACK_H
