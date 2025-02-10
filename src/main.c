#include "spatial.h"
#include "temporal.h"
#include "shift_magnify.h"
#include "roberts_cross_gradients.h"
#include "radial_gradient_convergence.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tiffio.h>
#include <string.h>

void print_gradients(float *gradient, int rowsM, int colsM) {
    for (int i = 0; i < rowsM; i++) {
        for (int j = 0; j < colsM; j++) {
            // Access flattened 1D array with 2D indexing
            printf("%f ", gradient[i * colsM + j]);
        }
        printf("\n");
    }
}

void save_tiff(const char *filename, const float *image, uint32_t width, uint32_t height, int nFrames) {
    TIFF *out = TIFFOpen(filename, "w");
    if (!out) {
        fprintf(stderr, "Error: Could not open output TIFF file %s\n", filename);
        return;
    }

    uint16_t *row_buffer = (uint16_t *)malloc(width * sizeof(uint16_t));
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
        TIFFSetField(out, TIFFTAG_PAGENUMBER, frame, nFrames); // Page index

        // Write image data for this frame
        const float *frame_data = image + (frame * width * height); // Offset for each frame
        for (uint32_t row = 0; row < height; row++) {
            for (uint32_t col = 0; col < width; col++) {
                row_buffer[col] = (uint16_t)(frame_data[row * width + col] * 65535);
            }
            TIFFWriteScanline(out, row_buffer, row, 0);
        }

        TIFFWriteDirectory(out); // Move to the next image (multi-frame TIFF)
    }

    free(row_buffer);
    TIFFClose(out);
    printf("Saved %d-frame TIFF to %s\n", nFrames, filename);
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
    
    float *output_frame = spatial(image_in, 1, rows, cols, shift, magnification, radius, sensitivity, doIntensityWeighting);
    
    if (!output_frame) {
        fprintf(stderr, "Error: spatial function returned NULL\n");
        free(output_frame);
        return;
    }

    // Save the image (we're saving the original image for testing)
    save_tiff(output_filename, output_frame, colsM, rowsM, nFrames);
    // Free the allocated memory
    free(output_frame);
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