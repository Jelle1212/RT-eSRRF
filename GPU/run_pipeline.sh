#!/bin/bash

# Default values
DEFAULT_NFRAMES=1
DEFAULT_SHIFT=0.0
DEFAULT_MAGNIFICATION=5.0
DEFAULT_RADIUS=2.0
DEFAULT_SENSITIVITY=1.0
DEFAULT_DO_INTENSITY_WEIGHTING=1
DEFAULT_TEMPORAL_TYPE=1

# Ensure at least 3 arguments (input, output, and ground truth)
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_tiff> <output_tiff> <output_gt_tiff> [nFrames] [rows] [cols] [shift] [magnification] [radius] [sensitivity] [doIntensityWeighting] [temporalType]"
    exit 1
fi

# Assign input files
INPUT_TIFF="$1"
OUTPUT_TIFF="$2"
OUTPUT_GT_TIFF="$3"

# Assign optional parameters with defaults
NFRAMES="${4:-$DEFAULT_NFRAMES}"
SHIFT="${7:-$DEFAULT_SHIFT}"
MAGNIFICATION="${8:-$DEFAULT_MAGNIFICATION}"
RADIUS="${9:-$DEFAULT_RADIUS}"
SENSITIVITY="${10:-$DEFAULT_SENSITIVITY}"
DO_INTENSITY_WEIGHTING="${11:-$DEFAULT_DO_INTENSITY_WEIGHTING}"
TEMPORAL_TYPE="${12:-$DEFAULT_TEMPORAL_TYPE}"

# Run `make` before executing the program
echo "Compiling with make..."
make
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed."
    exit 1
fi
echo "Compilation successful."

# Display parameter values
echo "Running CUDA Pipeline with:"
echo "  Input TIFF: $INPUT_TIFF"
echo "  Output TIFF: $OUTPUT_TIFF"
echo "  Ground Truth TIFF: $OUTPUT_GT_TIFF"
echo "  Frames: $NFRAMES"
echo "  Shift: $SHIFT"
echo "  Magnification: $MAGNIFICATION"
echo "  Radius: $RADIUS"
echo "  Sensitivity: $SENSITIVITY"
echo "  Intensity Weighting: $DO_INTENSITY_WEIGHTING"
echo "  Temporal Type: $TEMPORAL_TYPE"
echo ""

# Run CUDA pipeline
# nsys profile -o my_profile_report ./ESRRF_GPU "$INPUT_TIFF" "$OUTPUT_TIFF" "$OUTPUT_GT_TIFF" \
#     "$NFRAMES" "$SHIFT" "$MAGNIFICATION" \
#     "$RADIUS" "$SENSITIVITY" "$DO_INTENSITY_WEIGHTING" "$TEMPORAL_TYPE"

# nsys-ui my_profile_report.nsys-rep

ncu -f -o detailed_kernel_report ./ESRRF_GPU "$INPUT_TIFF" "$OUTPUT_TIFF" "$OUTPUT_GT_TIFF" \
    "$NFRAMES" "$SHIFT" "$MAGNIFICATION" \
    "$RADIUS" "$SENSITIVITY" "$DO_INTENSITY_WEIGHTING" "$TEMPORAL_TYPE"

ncu-ui detailed_kernel_report.ncu-rep

# ./ESRRF_GPU "$INPUT_TIFF" "$OUTPUT_TIFF" "$OUTPUT_GT_TIFF" \
#     "$NFRAMES" "$SHIFT" "$MAGNIFICATION" \
#     "$RADIUS" "$SENSITIVITY" "$DO_INTENSITY_WEIGHTING" "$TEMPORAL_TYPE"

# Check if execution was successful
if [ $? -ne 0 ]; then
    echo "Error: CUDA processing failed."
    exit 1
fi

echo "Processing complete."
