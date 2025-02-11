#include "shift_magnify.h"
#include <stdio.h>
#include <math.h>

float cubic(float v) {
  float a = 0.5;
  float z = 0;
  if (v < 0) {
    v = -v;
  }
  if (v < 1) {
    z = v * v * (v * (-a + 2) + (a - 3)) + 1;
  } else if (v < 2) {
    z = -a * v * v * v + 5 * a * v * v - 8 * a * v + 4 * a;
  }
  return z;
}

float interpolate(const float *image, float r, float c, int rows, int cols) {
  // return 0 if r OR c positions do not exist in image
  if (r < 0 || r >= rows || c < 0 || c >= cols) {
    return 0;
  }

  const int r_int = (int)floor((float) (r - 0.5));
  const int c_int = (int)floor((float) (c - 0.5));
  float q = 0;
  float p = 0;

  int r_neighbor, c_neighbor;

  for (int j = 0; j < 4; j++) {
    c_neighbor = c_int - 1 + j;
    p = 0;
    if (c_neighbor < 0 || c_neighbor >= cols) {
      continue;
    }

    for (int i = 0; i < 4; i++) {
      r_neighbor = r_int - 1 + i;
      if (r_neighbor < 0 || r_neighbor >= rows) {
        continue;
      }
      p = p + image[r_neighbor * cols + c_neighbor] *
                  cubic(r - (r_neighbor + 0.5));
    }
    q = q + p * cubic(c - (c_neighbor + 0.5));
  }
  return q;
}

void shift_magnify(const float *image_in, float *image_out, 
                   int rows, int cols, 
                   float shift_row, float shift_col, 
                   float magnification_row, float magnification_col) {
    
    int rowsM = (int)(rows * magnification_row);
    int colsM = (int)(cols * magnification_col);

    int i, j;
    float row, col;

    // Single frame, so we remove the outer f loop
    for (j = 0; j < colsM; j++) {
        col = j / magnification_col - shift_col;
        for (i = 0; i < rowsM; i++) {
            row = i / magnification_row - shift_row;
            // Adjust indexing for image_in (no longer need to consider the f dimension)
            image_out[i * colsM + j] = interpolate(&image_in[0], row, col, rows, cols);  // Process a single frame
        }
    }
}