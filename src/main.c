#include "spatial.h"
#include "temporal.h"
#include "shift_magnify.h"
#include "roberts_cross_gradients.h"
#include "radial_gradient_convergence.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tiffio.h>

void print_gradients(float *gradient, int rowsM, int colsM) {
    for (int i = 0; i < rowsM; i++) {
        for (int j = 0; j < colsM; j++) {
            // Access flattened 1D array with 2D indexing
            printf("%f ", gradient[i * colsM + j]);
        }
        printf("\n");
    }
}

void save_tiff(const char *filename, const float *image, uint32_t width, uint32_t height) {
    TIFF *out = TIFFOpen(filename, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output TIFF file %s\n", filename);
        return;
    }

    // Set TIFF tags
    TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 16);  // Set 16 bits per sample for 16-bit images
    TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1); // Grayscale image (1 channel)
    TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, width));

    // Allocate buffer for row data (16-bit per pixel)
    uint16_t *row_buffer = (uint16_t *)malloc(width * sizeof(uint16_t));
    if (!row_buffer) {
        fprintf(stderr, "Error: Memory allocation failed for row buffer\n");
        TIFFClose(out);
        return;
    }

    // Write image data (assuming the image is grayscale and normalized between [0, 1])
    for (uint32_t row = 0; row < height; row++) {
        for (uint32_t col = 0; col < width; col++) {
            // Convert float (0.0-1.0) to uint16_t (0-65535)
            row_buffer[col] = (uint16_t)(image[row * width + col] * 65535); // Assuming the values are normalized
        }
        // Write the row to the TIFF file
        TIFFWriteScanline(out, row_buffer, row, 0);
    }

    // Free resources
    free(row_buffer);
    TIFFClose(out);
    printf("Saved magnified image to %s\n", filename);
}

void load_tiff_and_process(const char *input_filename, const char *output_filename, 
                           float shift, float magnification, float radius,
                           float sensitivity, bool doIntensityWeighting) {
    // Open the TIFF file
    TIFF *tif = TIFFOpen(input_filename, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file %s\n", input_filename);
        return;
    }

    // Set the directory to the first image in the TIFF file (page 0)
    if (TIFFSetDirectory(tif, 0) != 1) {
        fprintf(stderr, "Error: Could not set to the first directory (image)\n");
        TIFFClose(tif);
        return;
    }

    uint32_t width, height;
    size_t npixels;
    uint16_t *raster;  // Use uint16_t to handle 16-bit images

    // Get image dimensions
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;

    // Allocate memory for pixel data (16-bit)
    raster = (uint16_t *)_TIFFmalloc(npixels * sizeof(uint16_t));
    if (!raster) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        TIFFClose(tif);
        return;
    }

    // Read image data (loop through all scanlines)
    for (uint32_t row = 0; row < height; row++) {
        if (TIFFReadScanline(tif, &raster[row * width], row, 0) != 1) {
            fprintf(stderr, "Error: Could not read scanline %d\n", row);
            _TIFFfree(raster);
            TIFFClose(tif);
            return;
        }
    }

    printf("Loaded TIFF image: %dx%d\n", width, height);

    // Convert the 16-bit image into a grayscale float image
    float *image_in = (float *)malloc(width * height * sizeof(float));
    if (!image_in) {
        fprintf(stderr, "Error: Memory allocation failed for grayscale image\n");
        _TIFFfree(raster);
        TIFFClose(tif);
        return;
    }

    // Convert the 16-bit image to grayscale (normalize it to [0, 1])
    for (size_t i = 0; i < npixels; i++) {
        image_in[i] = (float)raster[i] / 65535.0f;  // Normalize to [0, 1]
    }

    // Free the raster data as we no longer need it
    _TIFFfree(raster);
    TIFFClose(tif);

    // Now, use the shift_magnify function on the first image
    int rows = height;
    int cols = width;
    int nFrames = 1;  // We are processing only the first image (frame)
    
    // Allocate memory for the output image
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);
    float *magnified_image = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));

    float *gradient_col = (float *)malloc(nFrames * rows * cols * sizeof(float));
    float *gradient_row = (float *)malloc(nFrames * rows * cols * sizeof(float));

    float *gradient_col_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));
    float *gradient_row_interp = (float *)malloc(nFrames * 2 * rowsM * 2 * colsM * sizeof(float));

    float *rgc_map = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));
    // Call the shift_magnify function to apply shift and magnification
    shift_magnify(image_in, magnified_image, nFrames, rows, cols, shift, shift, magnification, magnification);
    roberts_cross_gradients(image_in, gradient_col, gradient_row, nFrames, rows, cols);
    shift_magnify(gradient_col, gradient_col_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);
    shift_magnify(gradient_row, gradient_row_interp, nFrames, rows, cols, shift, shift, magnification * 2, magnification * 2);
    radial_gradient_convergence(gradient_col_interp, gradient_row_interp, magnified_image, nFrames, rowsM, colsM, magnification, radius, sensitivity, doIntensityWeighting, rgc_map);
    // Save the image (we're saving the original image for testing)
    save_tiff(output_filename, rgc_map, colsM, rowsM);

    // Free the allocated memory
    free(rgc_map);
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image.tif> <output_image.tif>\n", argv[0]);
        return 1;
    }

    // Example parameters
    float shift = 0.0f;
    float magnification = 5.0;
    float radius = 2.0;
    float sensitivity = 1.0;
    bool doIntensityWeighting = true;

    load_tiff_and_process(argv[1], argv[2], shift, magnification, radius, sensitivity, doIntensityWeighting);

    return 0;
}