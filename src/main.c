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
#include <math.h>

// Function to load a TIFF image into a float array
float* load_tiff_image(const char *filename, int *width, int *height) {
    // Open the TIFF file
    TIFF *tif = TIFFOpen(filename, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file %s\n", filename);
        return NULL;
    }

    // Reset to the first frame
    TIFFSetDirectory(tif, 0);

    // Get image dimensions from the first frame
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);

    size_t npixels = (*width) * (*height);

    // Allocate memory for all frames
    float *image_in = (float *)malloc(npixels * sizeof(float));

    if (!image_in) {
        fprintf(stderr, "Error: Memory allocation failed for image data\n");
        TIFFClose(tif);
        return NULL;
    }

    // Allocate memory for the raster buffer (uint16_t since TIFF images are often 16-bit)
    uint16_t *raster = (uint16_t *)_TIFFmalloc(npixels * sizeof(uint16_t));
    if (!raster) {
        fprintf(stderr, "Error: Memory allocation failed for raster\n");
        free(image_in);
        TIFFClose(tif);
        return NULL;
    }

    // Read each scanline of the image (assuming only one frame)
    for (uint32_t row = 0; row < *height; row++) {
        if (TIFFReadScanline(tif, &raster[row * (*width)], row, 0) != 1) {
            fprintf(stderr, "Error: Could not read scanline %d\n", row);
            _TIFFfree(raster);
            free(image_in);
            TIFFClose(tif);
            return NULL;
        }
    }

    // Convert 16-bit raster data to normalized float grayscale and store in image_in
    for (size_t i = 0; i < npixels; i++) {
        image_in[i] = (float)raster[i] / 65535.0f;  // Normalize to [0, 1]
    }

    // Free the raster buffer after processing
    _TIFFfree(raster);

    // Close the TIFF file
    TIFFClose(tif);

    printf("Successfully loaded image from TIFF\n");
    // Return the processed image
    return image_in;
}

// Compute Mean Squared Error (MSE)
float compute_mse(const float *image1, const float *image2, int size) {
    float mse = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = image1[i] - image2[i];
        mse += diff * diff;
    }
    return mse / size;
}

// Compute Structural Similarity Index (SSIM)
float compute_ssim(const float *image1, const float *image2, int size) {
    float mean1 = 0, mean2 = 0, var1 = 0, var2 = 0, covar = 0;
    const float C1 = 0.0001f, C2 = 0.0009f; // Small constants for stability

    for (int i = 0; i < size; i++) {
        mean1 += image1[i];
        mean2 += image2[i];
    }
    mean1 /= size;
    mean2 /= size;

    for (int i = 0; i < size; i++) {
        float diff1 = image1[i] - mean1;
        float diff2 = image2[i] - mean2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        covar += diff1 * diff2;
    }

    var1 /= size;
    var2 /= size;
    covar /= size;

    return ((2 * mean1 * mean2 + C1) * (2 * covar + C2)) /
           ((mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2));
}

// Compute Peak Signal-to-Noise Ratio (PSNR)
float compute_psnr(const float *image1, const float *image2, int size) {
    float mse = compute_mse(image1, image2, size);
    if (mse == 0) return INFINITY; // Perfect match

    float max_pixel_value = 1.0f; // Assuming images are normalized to [0,1]
    return 10.0f * log10((max_pixel_value * max_pixel_value) / mse);
}

// Compute Mean Absolute Error (MAE)
float compute_mae(const float *image1, const float *image2, int size) {
    float mae = 0.0f;
    for (int i = 0; i < size; i++) {
        mae += fabs(image1[i] - image2[i]);
    }
    return mae / size;
}

// Compute Maximum Absolute Error (MaxAE)
float compute_maxae(const float *image1, const float *image2, int size) {
    float max_error = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabs(image1[i] - image2[i]);
        if (diff > max_error) max_error = diff;
    }
    return max_error;
}

// Compute Pearson Correlation Coefficient (PCC)
float compute_pcc(const float *image1, const float *image2, int size) {
    float mean1 = 0, mean2 = 0, sum1 = 0, sum2 = 0, sum_xy = 0;

    for (int i = 0; i < size; i++) {
        mean1 += image1[i];
        mean2 += image2[i];
    }
    mean1 /= size;
    mean2 /= size;

    for (int i = 0; i < size; i++) {
        float diff1 = image1[i] - mean1;
        float diff2 = image2[i] - mean2;
        sum1 += diff1 * diff1;
        sum2 += diff2 * diff2;
        sum_xy += diff1 * diff2;
    }

    return sum_xy / (sqrt(sum1) * sqrt(sum2) + 1e-10); // Adding epsilon to avoid division by zero
}

// Compare two TIFF images with additional metrics
void compare_tiff_images(const char *generated_file, const char *ground_truth_file) {
    int width1, height1, width2, height2;
    float *image1 = load_tiff_image(generated_file, &width1, &height1);
    float *image2 = load_tiff_image(ground_truth_file, &width2, &height2);

    if (!image1 || !image2) {
        fprintf(stderr, "Error: Could not load images for comparison\n");
        return;
    }

    if (width1 != width2 || height1 != height2) {
        fprintf(stderr, "Error: Image dimensions do not match!\n");
        free(image1);
        free(image2);
        return;
    }

    size_t npixels = width1 * height1;
    
    float mse   = compute_mse(image1, image2, npixels);
    float ssim  = compute_ssim(image1, image2, npixels);
    float psnr  = compute_psnr(image1, image2, npixels);
    float mae   = compute_mae(image1, image2, npixels);
    float maxae = compute_maxae(image1, image2, npixels);
    float pcc   = compute_pcc(image1, image2, npixels);

    printf("Comparison Results:\n");
    printf("  - MSE:  %.6f\n", mse);
    printf("  - SSIM: %.6f\n", ssim);
    printf("  - PSNR: %.2f dB\n", psnr);
    printf("  - MAE:  %.6f\n", mae);
    printf("  - MaxAE: %.6f\n", maxae);
    printf("  - PCC:  %.6f\n", pcc);

    free(image1);
    free(image2);
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

    // Check if the image contains negative values
    bool has_negatives = false;
    for (uint32_t i = 0; i < width * height * nFrames; i++) {
        if (image[i] < 0) {
            has_negatives = true;
            break;
        }
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
                float value = frame_data[row * width + col];

                if (has_negatives) {
                    // Scale from [-1,1] to [0,1]
                    value = (value + 1.0f) / 2.0f;
                }

                // Convert to uint16_t with clamping
                float scaled_value = value * 65535.0f;
                if (scaled_value < 0) scaled_value = 0;
                if (scaled_value > 65535) scaled_value = 65535;

                row_buffer[col] = (uint16_t)scaled_value;            
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
                           float sensitivity, bool doIntensityWeighting, int temporalType) {
    // Open the TIFF file
    TIFF *tif = TIFFOpen(input_filename, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file %s\n", input_filename);
        return;
    }

    // Determine the number of frames (directories) in the TIFF
    int nFrames = 0;
    do {
        nFrames++;
    } while (TIFFReadDirectory(tif));  // Moves to the next frame

    // Reset to the first frame
    TIFFSetDirectory(tif, 0);

    uint32_t width, height;
    size_t npixels;
    uint16_t *raster;

    // Get image dimensions from the first frame
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height);
    npixels = width * height;

    // Allocate memory for all frames
    float *image_in = (float *)malloc(nFrames * npixels * sizeof(float));
    if (!image_in) {
        fprintf(stderr, "Error: Memory allocation failed for all frames\n");
        TIFFClose(tif);
        return;
    }

    // Read each frame and store it in `image_in`
    for (int frame = 0; frame < nFrames; frame++) {
        // Allocate memory for the current frame
        raster = (uint16_t *)_TIFFmalloc(npixels * sizeof(uint16_t));
        if (!raster) {
            fprintf(stderr, "Error: Memory allocation failed for frame %d\n", frame);
            free(image_in);
            TIFFClose(tif);
            return;
        }

        // Read the current frame
        for (uint32_t row = 0; row < height; row++) {
            if (TIFFReadScanline(tif, &raster[row * width], row, 0) != 1) {
                fprintf(stderr, "Error: Could not read scanline %d in frame %d\n", row, frame);
                _TIFFfree(raster);
                free(image_in);
                TIFFClose(tif);
                return;
            }
        }

        // Convert 16-bit image to normalized float grayscale and store in `image_in`
        for (size_t i = 0; i < npixels; i++) {
            image_in[frame * npixels + i] = (float)raster[i] / 65535.0f;  // Normalize to [0,1]
        }

        // Free raster buffer
        _TIFFfree(raster);

        // Move to next frame
        if (frame < nFrames - 1 && !TIFFReadDirectory(tif)) {
            fprintf(stderr, "Error: Could not move to frame %d\n", frame + 1);
            free(image_in);
            TIFFClose(tif);
            return;
        }
    }

    printf("Loaded %d frames from TIFF image (%dx%d each)\n", nFrames, width, height);

    // Now, use the shift_magnify function on the first image
    int rows = height;
    int cols = width;
    
    // Allocate memory for the output image
    int rowsM = (int)(rows * magnification);
    int colsM = (int)(cols * magnification);

    float *rgc_maps = (float *)malloc(nFrames * rowsM * colsM * sizeof(float));
    if (!rgc_maps) {
        fprintf(stderr, "Error: Memory allocation failed for rgc_maps\n");
        return;
    }

    for (int frame = 0; frame < nFrames; frame++) {
        // Offset to process one frame at a time
        const float *input_frame = image_in + (frame * rows * cols);
        float *output_frame = spatial(input_frame, rows, cols, shift, magnification, radius, sensitivity, doIntensityWeighting);
        
        if (!output_frame) {
            fprintf(stderr, "Error: spatial function returned NULL\n");
            free(rgc_maps);
            return;
        }

        // Copy processed frame into rgc_maps
        memcpy(rgc_maps + (frame * rowsM * colsM), output_frame, rowsM * colsM * sizeof(float));
    }

    float *sr_image = (float *)malloc(rowsM * colsM * sizeof(float));
    if (!sr_image) {
        fprintf(stderr, "Error: Memory allocation failed for rgc_maps\n");
        return;
    }

    sr_image = temporal(rgc_maps, temporalType, nFrames, rowsM, colsM);
    
    if (!rgc_maps) {
        fprintf(stderr, "Error: spatial function returned NULL\n");
        free(rgc_maps);
        return;
    }

    // Save the image (we're saving the original image for testing)
    // save_tiff(output_filename, rgc_maps, colsM, rowsM, nFrames);
    save_tiff(output_filename, sr_image, colsM, rowsM, 1);

    // Free the allocated memory
    free(rgc_maps);
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
    int temporalType = 0;

    load_tiff_and_process(argv[1], argv[2], shift, magnification, radius, sensitivity, doIntensityWeighting, temporalType);
    compare_tiff_images(argv[2], argv[3]);

    return 0;
}