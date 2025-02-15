#include "temporal.h"
#include "settings.h"
#include <stdio.h>

void average(const float* image_stack, float* image_out, int frames, int rows, int cols) {
    // Compute the mean directly in one pass
    for (int i = 0; i < rows * cols; i++) {
        image_out[i] = 0.0f;
    }

    for (int f = 0; f < frames; f++) {
        for (int i = 0; i < rows * cols; i++) {
            image_out[i] += image_stack[f * rows * cols + i];
        }
    }

    for (int i = 0; i < rows * cols; i++) {
        image_out[i] /= frames;
    }
}

void variance(const float* image_stack, float* image_out, float* mean_image, int frames, int rows, int cols) {
    // Compute the mean
    average(image_stack, mean_image, frames, rows, cols);

    // Compute variance in a single pass
    for (int i = 0; i < rows * cols; i++) {
        image_out[i] = 0.0f;
    }

    for (int f = 0; f < frames; f++) {
        for (int i = 0; i < rows * cols; i++) {
            float diff = image_stack[f * rows * cols + i] - mean_image[i];
            image_out[i] += diff * diff;
        }
    }

    for (int i = 0; i < rows * cols; i++) {
        image_out[i] /= frames;
    }
}

void temporal_auto_correlation(const float* image_stack, float* image_out, float* mean_image, int frames, int rows, int cols) {
    int nlag = 1;

    // Compute the mean
    average(image_stack, mean_image, frames, rows, cols);

    // Compute the temporal auto-correlation at lag 1
    for (int i = 0; i < rows * cols; i++) {
        image_out[i] = 0.0f;
    }

    for (int f = 0; f < frames - nlag; f++) {
        for (int i = 0; i < rows * cols; i++) {
            float centered1 = image_stack[f * rows * cols + i] - mean_image[i];
            float centered2 = image_stack[(f + nlag) * rows * cols + i] - mean_image[i];
            image_out[i] += centered1 * centered2;
        }
    }

    for (int i = 0; i < rows * cols; i++) {
        image_out[i] /= (frames - nlag);
    }
}

float* temporal(const float* image_stack, int type, int frames, int rows, int cols) {
    static float SR[MAX_ROWS * MAX_COLS];
    static float temp_mean[MAX_ROWS * MAX_COLS];  // Temporary buffer for mean

    if (type == 0) {
        average(image_stack, SR, frames, rows, cols);
    } else if (type == 1) {
        variance(image_stack, SR, temp_mean, frames, rows, cols);
    } else {
        temporal_auto_correlation(image_stack, SR, temp_mean, frames, rows, cols);
    }

    return SR;
}