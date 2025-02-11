#include "temporal.h"
#include "settings.h"
#include <stdio.h>

float* temporal(const float* image_stack, int type, int frames, int rows, int cols) {

}

void average(const float* image_stack, float* image_stack_out, int frames, int rows, int cols) {
    // Initialize the output array to zero
    for (int i = 0; i < rows * cols; i++) {
        image_stack_out[i] = 0.0f;
    }

    // Sum the pixel values across all frames
    for (int f = 0; f < frames; f++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int index = f * rows * cols + r * cols + c;
                image_stack_out[r * cols + c] += image_stack[index];
            }
        }
    }

    // Divide by the number of frames to compute the average
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            image_stack_out[r * cols + c] /= frames;
        }
    }
}

void variance(const float* image_stack, float* image_stack_out, int frames, int rows, int cols) {
    
    static float mean_image[MAX_INPUT_ROWS * MAX_INPUT_COLS];
    
    // Compute the mean of the image stack
    average(image_stack, mean_image, frames, rows, cols);

    // Initialize the output array to zero
    for (int i = 0; i < rows * cols; i++) {
        image_stack_out[i] = 0.0f;
    }

    // Compute the squared differences from the mean
    for (int f = 0; f < frames; f++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int index = f * rows * cols + r * cols + c;
                float diff = image_stack[index] - mean_image[r * cols + c];
                image_stack_out[r * cols + c] += diff * diff;
            }
        }
    }

    // Divide by the number of frames to compute the variance
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            image_stack_out[r * cols + c] /= frames;
        }
    }

}

void temporal_auto_correlation(const float* image_stack, float* image_stack_out, int frames, int rows, int cols) {
    // Static array for the mean image
    static float mean_image[MAX_INPUT_ROWS * MAX_INPUT_COLS];

    // Step 1: Compute the mean of the image stack along the time axis
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            mean_image[r * cols + c] = 0.0f;
            for (int f = 0; f < frames; f++) {
                int index = f * rows * cols + r * cols + c;
                mean_image[r * cols + c] += image_stack[index];
            }
            mean_image[r * cols + c] /= frames;
        }
    }

    // Step 2: Compute the temporal auto-correlation at lag 1
    int nlag = 1;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            image_stack_out[r * cols + c] = 0.0f;
            for (int f = 0; f < frames - nlag; f++) {
                int index1 = f * rows * cols + r * cols + c;
                int index2 = (f + nlag) * rows * cols + r * cols + c;
                // Center the data on-the-fly and compute the product
                float centered1 = image_stack[index1] - mean_image[r * cols + c];
                float centered2 = image_stack[index2] - mean_image[r * cols + c];
                image_stack_out[r * cols + c] += centered1 * centered2;
            }
            // Step 3: Average the result along the time axis
            image_stack_out[r * cols + c] /= (frames - nlag);
        }
    }
}