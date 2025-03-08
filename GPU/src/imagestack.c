#include "imagestack.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

// Create a new ImageStack
ImageStack* createImageStack(int rows, int cols) {
    ImageStack* stack = (ImageStack*)malloc(sizeof(ImageStack));
    if (!stack) {
        fprintf(stderr, "Error: Could not allocate memory for ImageStack.\n");
        return NULL;
    }
    stack->rows = rows;
    stack->cols = cols;
    stack->stack_size = 0;
    stack->capacity = 10;
    stack->stack = (float**)malloc(stack->capacity * sizeof(float*));
    if (!stack->stack) {
        fprintf(stderr, "Error: Could not allocate memory for stack images.\n");
        free(stack);
        return NULL;
    }
    return stack;
}

// Push an image onto the stack
void pushImage(ImageStack* stack, float* image) {
    if (stack->stack_size >= stack->capacity) {
        stack->capacity *= 2;
        stack->stack = (float**)realloc(stack->stack, stack->capacity * sizeof(float*));
        if (!stack->stack) {
            fprintf(stderr, "Error: Could not allocate memory while expanding stack.\n");
            return;
        }
    }

    float* imgCopy = (float*)malloc(stack->rows * stack->cols * sizeof(float));
    if (!imgCopy) {
        fprintf(stderr, "Error: Could not allocate memory for image copy.\n");
        return;
    }
    memcpy(imgCopy, image, stack->rows * stack->cols * sizeof(float));

    stack->stack[stack->stack_size++] = imgCopy;
}

// Pop an image from the stack
float* popImage(ImageStack* stack) {
    if (stack->stack_size == 0) {
        fprintf(stderr, "Error: Stack is empty!\n");
        return NULL;
    }

    float* img = stack->stack[--stack->stack_size];

    return img;
}
// Save stack as a 32-bit float TIFF
void saveAsTiffFloat32(const char* filename, ImageStack* stack) {
    if (stack->stack_size == 0) {
        fprintf(stderr, "Error: Image stack is empty, nothing to save.\n");
        return;
    }

    int nFrames = stack->stack_size;
    float* imageData = (float*)malloc(nFrames * stack->rows * stack->cols * sizeof(float));
    if (!imageData) {
        fprintf(stderr, "Error: Could not allocate memory for TIFF data.\n");
        return;
    }

    for (int i = 0; i < nFrames; i++) {
        memcpy(imageData + (i * stack->rows * stack->cols), stack->stack[i], stack->rows * stack->cols * sizeof(float));
    }

    save_tiff_float32(filename, imageData, stack->cols, stack->rows, nFrames);
    free(imageData);
}

// Save stack as a 16-bit TIFF
void saveAsTiff16(const char* filename, ImageStack* stack) {
    if (stack->stack_size == 0) {
        fprintf(stderr, "Error: Image stack is empty, nothing to save.\n");
        return;
    }

    int nFrames = stack->stack_size;
    float* imageData = (float*)malloc(nFrames * stack->rows * stack->cols * sizeof(float));
    if (!imageData) {
        fprintf(stderr, "Error: Could not allocate memory for TIFF data.\n");
        return;
    }

    for (int i = 0; i < nFrames; i++) {
        memcpy(imageData + (i * stack->rows * stack->cols), stack->stack[i], stack->rows * stack->cols * sizeof(float));
    }

    save_tiff(filename, imageData, stack->cols, stack->rows, nFrames);
    free(imageData);
}

// Free the ImageStack
void freeImageStack(ImageStack* stack) {
    for (int i = 0; i < stack->stack_size; i++) {
        free(stack->stack[i]);
    }
    free(stack->stack);
    free(stack);
}

// Function to save a 32-bit float TIFF
void save_tiff_float32(const char* filename, const float* image, uint32_t width, uint32_t height, int nFrames) {
    TIFF* out = TIFFOpen(filename, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output TIFF file %s\n", filename);
        return;
    }

    for (int frame = 0; frame < nFrames; frame++) {
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(out, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width));
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, frame, nFrames);

        const float* frame_data = image + (frame * width * height);
        for (uint32_t row = 0; row < height; row++) {
            TIFFWriteScanline(out, (tdata_t)(frame_data + row * width), row, 0);
        }

        TIFFWriteDirectory(out);
    }

    TIFFClose(out);
}

// Function to save a 16-bit TIFF
void save_tiff(const char* filename, const float* image, uint32_t width, uint32_t height, int nFrames) {
    TIFF* out = TIFFOpen(filename, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output TIFF file %s\n", filename);
        return;
    }

    uint16_t* row_buffer = (uint16_t*)malloc(width * sizeof(uint16_t));
    if (!row_buffer) {
        fprintf(stderr, "Error: Memory allocation failed for row buffer\n");
        TIFFClose(out);
        return;
    }

    for (int frame = 0; frame < nFrames; frame++) {
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width));
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        TIFFSetField(out, TIFFTAG_PAGENUMBER, frame, nFrames);

        const float* frame_data = image + (frame * width * height);
        for (uint32_t row = 0; row < height; row++) {
            for (uint32_t col = 0; col < width; col++) {
                row_buffer[col] = (uint16_t)(frame_data[row * width + col] * 65535.0f);
            }
            TIFFWriteScanline(out, row_buffer, row, 0);
        }

        TIFFWriteDirectory(out);
    }

    free(row_buffer);
    TIFFClose(out);
}
