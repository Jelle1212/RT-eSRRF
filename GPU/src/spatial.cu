#include "spatial.hu"
#include "shift_magnify.hu"
#include "roberts_cross_gradients.hu"
#include "radial_gradient_convergence.hu"
#include "settings.hu"
#include <stdio.h>
#include <stdlib.h>


extern "C" {
    float* spatial(const float *image_in, int rows, int cols, 
                float shift, float magnification, float radius, 
                float sensitivity, bool doIntensityWeighting) {

        float *d_magnified_image;
        float *d_gradient_col;
        float *d_gradient_row;
        float *d_gradient_col_interp;
        float *d_gradient_row_interp;
        float *d_rgc_map;
        float *d_image_in;
        
        int rowsM = (int)(rows * magnification);
        int colsM = (int)(cols * magnification);

        // Allocate memory on the GPU
        cudaMalloc((void**)&d_magnified_image, MAX_ROWS * MAX_COLS * sizeof(float));
        cudaMalloc((void**)&d_gradient_col, MAX_INPUT_ROWS * MAX_INPUT_COLS * sizeof(float));
        cudaMalloc((void**)&d_gradient_row, MAX_INPUT_ROWS * MAX_INPUT_COLS * sizeof(float));
        cudaMalloc((void**)&d_gradient_col_interp, 2 * MAX_ROWS * 2 * MAX_COLS * sizeof(float));
        cudaMalloc((void**)&d_gradient_row_interp, 2 * MAX_ROWS * 2 * MAX_COLS * sizeof(float));
        cudaMalloc((void**)&d_rgc_map, MAX_ROWS * MAX_COLS * sizeof(float));
        cudaMalloc((void**)&d_image_in, MAX_INPUT_ROWS * MAX_INPUT_COLS * sizeof(float));

        // Step 2: Copy data from host to GPU
        cudaMemcpy(d_image_in, image_in, MAX_INPUT_ROWS * MAX_INPUT_COLS * sizeof(float), cudaMemcpyHostToDevice);

        static float rgc_map[MAX_ROWS * MAX_COLS];

        // Create CUDA streams
        cudaStream_t stream1, stream2, stream3, stream4;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        cudaStreamCreate(&stream4);

        // Call the shift_magnify function to apply shift and magnification
        shift_magnify(d_image_in, d_magnified_image, rows, cols, shift, shift, magnification, magnification, stream1);
        roberts_cross_gradients(d_image_in, d_gradient_col, d_gradient_row, rows, cols, stream2);

        // Synchronize streams to ensure both operations are complete
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        shift_magnify(d_gradient_col, d_gradient_col_interp, rows, cols, shift, shift, magnification * 2, magnification * 2, stream3);
        shift_magnify(d_gradient_row, d_gradient_row_interp, rows, cols, shift, shift, magnification * 2, magnification * 2, stream4);
        
        // Synchronize streams to ensure both operations are complete
        cudaStreamSynchronize(stream3);
        cudaStreamSynchronize(stream4);
        
        radial_gradient_convergence(d_gradient_col_interp, d_gradient_row_interp, d_magnified_image, rowsM, colsM, magnification, radius, sensitivity, doIntensityWeighting, d_rgc_map);
        
        // Copy data from GPU to CPU
        cudaMemcpy(rgc_map, d_rgc_map, MAX_ROWS * MAX_COLS * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_magnified_image);
        cudaFree(d_gradient_col);
        cudaFree(d_gradient_row);
        cudaFree(d_gradient_col_interp);
        cudaFree(d_gradient_row_interp);
        cudaFree(d_rgc_map);
        cudaFree(d_image_in);

        // Destroy streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
        cudaStreamDestroy(stream4);
        return rgc_map;
    }
}

