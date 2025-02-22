#include "spatial.hu"
#include "shift_magnify.hu"
#include "roberts_cross_gradients.hu"
#include "radial_gradient_convergence.hu"
#include "settings.hu"
#include <stdio.h>
#include <stdlib.h>


extern "C" {
    void spatial(SpatialParams &params) {

        int rowsM = (int)(params.rows * params.magnification);
        int colsM = (int)(params.cols * params.magnification);

        // Create CUDA streams
        cudaStream_t stream1, stream2, stream3, stream4;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        cudaStreamCreate(&stream4);

        // Call the shift_magnify function to apply shift and magnification
        shift_magnify(params.d_image_in, params.d_magnified_image, params.rows, params.cols, params.shift, params.shift, params.magnification, params.magnification, stream1);
        roberts_cross_gradients(params.d_image_in, params.d_gradient_col, params.d_gradient_row, params.rows, params.cols, stream2);

        // Synchronize streams to ensure both operations are complete
        cudaStreamSynchronize(stream1);
        cudaStreamSynchronize(stream2);

        shift_magnify(params.d_gradient_col, params.d_gradient_col_interp, params.rows, params.cols, params.shift, params.shift, params.magnification * 2, params.magnification * 2, stream3);
        shift_magnify(params.d_gradient_row, params.d_gradient_row_interp, params.rows, params.cols, params.shift, params.shift, params.magnification * 2, params.magnification * 2, stream4);
        
        // Synchronize streams to ensure both operations are complete
        cudaStreamSynchronize(stream3);
        cudaStreamSynchronize(stream4);
        
        radial_gradient_convergence(params.d_gradient_col_interp, params.d_gradient_row_interp, params.d_magnified_image, rowsM, colsM, params.magnification, params.radius, params.sensitivity, params.doIntensityWeighting, params.d_rgc_map);

        // Destroy streams
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaStreamDestroy(stream3);
        cudaStreamDestroy(stream4);
    }
}

