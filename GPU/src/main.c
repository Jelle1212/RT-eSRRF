#include "pipeline.hu"
#include "imagestack.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <tiffio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// Define a small tolerance value
#define TOLERANCE 1e-6

void test_image_stack() {
    int rows = 2, cols = 3;

    // Create an ImageStack
    ImageStack* stack = createImageStack(rows, cols);
    assert(stack != NULL);
    printf("✅ ImageStack created successfully.\n");

    // Create test images
    float image1[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    float image2[6] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

    // Push images onto the stack
    pushImage(stack, image1);
    pushImage(stack, image2);
    assert(stack->stack_size == 2);
    printf("✅ Images pushed successfully. Stack size: %d\n", stack->stack_size);

    // Pop an image from the stack
    float* poppedImage = popImage(stack);
    assert(poppedImage != NULL);
    assert(fabs(poppedImage[0] - 1.1) < TOLERANCE);
    assert(fabs(poppedImage[5] - 1.6) < TOLERANCE);
    printf("✅ Popped image has expected values.\n");
    free(poppedImage); // Free popped image

    // Check stack size after popping
    assert(stack->stack_size == 1);
    printf("✅ Stack size is correct after pop: %d\n", stack->stack_size);

    // Save images as TIFF (optional test)
    saveAsTiffFloat32("test_output_float32.tiff", stack);
    saveAsTiff16("test_output_16bit.tiff", stack);
    printf("✅ TIFF files saved successfully.\n");

    // Free stack
    freeImageStack(stack);
    printf("✅ ImageStack freed successfully.\n");

    printf("\n🎉 All tests passed!\n");
}

// Function to load all frames from a multi-frame TIFF into a float array
float* load_tiff_image_stack(const char *filename, int *width, int *height, int *nFrames) {
    // Open the TIFF file
    TIFF *tif = TIFFOpen(filename, "r");
    if (!tif) {
        fprintf(stderr, "Error: Could not open TIFF file %s\n", filename);
        return NULL;
    }

    // Get the first frame's dimensions
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);

    size_t npixels = (*width) * (*height);  // Number of pixels per frame

    // Count the number of frames
    *nFrames = 0;
    do { (*nFrames)++; } while (TIFFReadDirectory(tif));

    // Allocate memory for all frames
    float *imageStack = (float *)malloc((*nFrames) * npixels * sizeof(float));
    if (!imageStack) {
        fprintf(stderr, "Error: Memory allocation failed for image stack\n");
        TIFFClose(tif);
        return NULL;
    }

    // Reset to the first frame
    TIFFSetDirectory(tif, 0);

    // Allocate memory for a single frame raster buffer
    uint16_t *raster = (uint16_t *)_TIFFmalloc(npixels * sizeof(uint16_t));
    if (!raster) {
        fprintf(stderr, "Error: Memory allocation failed for raster\n");
        free(imageStack);
        TIFFClose(tif);
        return NULL;
    }

    // Load all frames
    for (int frame = 0; frame < *nFrames; frame++) {
        TIFFSetDirectory(tif, frame);  // Move to the current frame

        // Read the entire image into the raster buffer
        for (uint32_t row = 0; row < *height; row++) {
            if (TIFFReadScanline(tif, &raster[row * (*width)], row, 0) != 1) {
                fprintf(stderr, "Error: Could not read scanline %d in frame %d\n", row, frame);
                _TIFFfree(raster);
                free(imageStack);
                TIFFClose(tif);
                return NULL;
            }
        }

        // Convert 16-bit raster data to normalized float grayscale and store in imageStack
        for (size_t i = 0; i < npixels; i++) {
            imageStack[frame * npixels + i] = (float)raster[i] / 65535.0f;  // Normalize to [0, 1]
        }
    }

    // Free resources
    _TIFFfree(raster);
    TIFFClose(tif);

    printf("Successfully loaded %d frames from TIFF file %s\n", *nFrames, filename);
    return imageStack;
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

void compare_tiff_image_stacks(const char *generated_file, const char *ground_truth_file) {
    int width1, height1, width2, height2, nFrames1, nFrames2;
    float *imageStack1 = load_tiff_image_stack(generated_file, &width1, &height1, &nFrames1);
    float *imageStack2 = load_tiff_image_stack(ground_truth_file, &width2, &height2, &nFrames2);

    if (!imageStack1 || !imageStack2) {
        fprintf(stderr, "Error: Could not load image stacks for comparison\n");
        return;
    }

    if (width1 != width2 || height1 != height2 || nFrames1 != nFrames2) {
        fprintf(stderr, "Error: Image dimensions do not match!\n");
        free(imageStack1);
        free(imageStack2);
        return;
    }

    int nFrames = nFrames1;

    size_t npixels = width1 * height1;

    printf("Frame-wise Comparison Results:\n");
    
    for (int frame = 0; frame < nFrames; frame++) {
        float *image1 = imageStack1 + (frame * npixels);
        float *image2 = imageStack2 + (frame * npixels);

        float mse   = compute_mse(image1, image2, npixels);
        float ssim  = compute_ssim(image1, image2, npixels);
        float psnr  = compute_psnr(image1, image2, npixels);
        float mae   = compute_mae(image1, image2, npixels);
        float maxae = compute_maxae(image1, image2, npixels);
        float pcc   = compute_pcc(image1, image2, npixels);

        printf("Frame %d:\n", frame);
        printf("  - MSE:  %.6f\n", mse);
        printf("  - SSIM: %.6f\n", ssim);
        printf("  - PSNR: %.2f dB\n", psnr);
        printf("  - MAE:  %.6f\n", mae);
        printf("  - MaxAE: %.6f\n", maxae);
        printf("  - PCC:  %.6f\n", pcc);
    }

    free(imageStack1);
    free(imageStack2);
}

void load_tiff_and_process(const char *input_filename, const char *output_filename, struct ESRRFParams *eSRRFParams) {
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
    uint16_t *image_in = (uint16_t *)malloc(nFrames * npixels * sizeof(uint16_t));
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
            image_in[frame * npixels + i] = raster[i];  // Normalize to [0,1]
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
    eSRRFParams->rows = height;
    eSRRFParams->cols = width;
    
    // Allocate memory for the output image
    int rowsM = (int)(eSRRFParams->rows * eSRRFParams->magnification);
    int colsM = (int)(eSRRFParams->cols * eSRRFParams->magnification);

    initPipeline(eSRRFParams);
    // float *sr_image = (float *)malloc(rowsM * colsM * eSRRFParams->nFrames * sizeof(float));

    // Create stack and push image
    ImageStack* stack = createImageStack(rowsM, colsM);
    // ImageStack* stack = createImageStack(height, width);

    if (!stack) {
        return;
    }

    // Create CUDA events for profiling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float *sr_image = (float *)malloc(rowsM * colsM * sizeof(float));

    // Variable to accumulate total time
    float total_time = 0.0f;

    for (int frame = 0; frame < eSRRFParams->nFrames; frame++) {
        // Offset to process one frame at a time
        const unsigned short *input_frame = image_in + (frame * eSRRFParams->rows * eSRRFParams->cols);
        // Record the start event
        cudaEventRecord(start, 0);
        
        processFrame(input_frame, sr_image, frame);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);  // Ensure all GPU work is done

        // Calculate elapsed time for this frame
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Accumulate total time
        total_time += milliseconds;
        // Print time for this frame (optional)
        printf("Frame %d time: %.2f ms\n", frame, milliseconds);
    }
    // Compute and print the average time per frame
    float average_time = total_time / eSRRFParams->nFrames;
    printf("Total time for %d frames: %.2f ms\n", eSRRFParams->nFrames, total_time);
    printf("Average time per frame: %.2f ms\n", average_time);

    // Clean up CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (!sr_image) {
        fprintf(stderr, "Error: Memory allocation failed for rgc_maps\n");
        return;
    }

    pushImage(stack, sr_image);

    // Save the image (we're saving the original image for testing)
    saveAsTiff16(output_filename, stack);

    // Free the allocated memory
    deintPipeline();
    freeImageStack(stack);
    free(sr_image);
}

#define NUM_FRAMES 100
#define CSV_FILE "latency_results.csv"

void testPerformanceOverSizes(struct ESRRFParams *eSRRFParams) {
    FILE *csv = fopen(CSV_FILE, "w");
    if (!csv) {
        perror("Could not open CSV file");
        return;
    }

    fprintf(csv, "Width,Height,AverageTime_ms,Variance_ms2\n");

    int sizes[] = {128, 256, 512, 1024, 2048};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        int width = sizes[i];
        int height = sizes[i];
        int num_pixels = width * height;

        eSRRFParams->cols = width;
        eSRRFParams->rows = height;

        eSRRFParams->nFrames = NUM_FRAMES;
        int rowsM = (int)(eSRRFParams->rows * eSRRFParams->magnification);
        int colsM = (int)(eSRRFParams->cols * eSRRFParams->magnification);

        printf("Testing size: %dx%d\n", width, height);

        // Allocate input
        unsigned short *image_in = (unsigned short *)malloc(num_pixels * sizeof(unsigned short));
        float *sr_image = (float *)malloc(rowsM * colsM * sizeof(float));

        // Fill image with dummy data
        for (int j = 0; j < num_pixels; ++j) {
            image_in[j] = (unsigned short)(j % 65535);
        }

        // Setup CUDA events
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float times[NUM_FRAMES];
        float total_time = 0.0f;

        initPipeline(eSRRFParams);

        printf("Warming up GPU...\n");
        for (int warm = 0; warm < 5; ++warm) {
            processFrame(image_in, sr_image, warm);  // Warm-up run, not timed
        }
        cudaDeviceSynchronize();  // Ensure all GPU work is done before timing

        for (int frame = 0; frame < NUM_FRAMES; ++frame) {            
            cudaEventRecord(start, 0);
            processFrame(image_in, sr_image, frame);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            times[frame] = milliseconds;
            total_time += milliseconds;
        }

        // Compute average
        float average = total_time / NUM_FRAMES;

        // Compute variance
        float variance = 0.0f;
        for (int j = 0; j < NUM_FRAMES; ++j) {
            float diff = times[j] - average;
            variance += diff * diff;
        }
        variance /= NUM_FRAMES;

        // Write to CSV
        fprintf(csv, "%d,%d,%.4f,%.4f\n", width, height, average, variance);

        // Clean up
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(image_in);
        free(sr_image);
        deintPipeline();
    }
    fclose(csv);
    printf("Results saved to %s\n", CSV_FILE);
}



int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <input_image.tif> <output_image.tif>\n", argv[0]);
        return 1;
    }

    struct ESRRFParams eSRRFParams = {
        .nFrames = atoi(argv[4]), 
        .rows = 0, 
        .cols = 0, 
        .shift = atof(argv[5]), 
        .magnification = atof(argv[6]), 
        .radius = atof(argv[7]), 
        .sensitivity = atof(argv[8]), 
        .doIntensityWeighting = atoi(argv[9]), 
        .temporalType = atoi(argv[10])
    };
    // test_image_stack();
    // load_tiff_and_process(argv[1], argv[2], &eSRRFParams);
    // compare_tiff_image_stacks(argv[2], argv[3]);
    testPerformanceOverSizes(&eSRRFParams);
    return 0;
}