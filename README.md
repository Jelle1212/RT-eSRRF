This project is a CUDA-accelerated implementation of **eSRRF** (enhanced Super-Resolution Radial Fluctuations), a widely used algorithm for live-cell super-resolution microscopy. The goal was to reduce latency and computational overhead to enable **real-time super-resolved imaging**—useful for live feedback, microscope tuning, and high-throughput analysis.

## About the Project
To sharpen my CUDA programming skills, I re-implemented the original eSRRF algorithm in C and optimized it with CUDA. While the core principles remain unchanged, this version introduces **incremental temporal analysis** for AVG, VAR, and TAC2 calculations.

### Key Improvements
- **Incremental statistics**: Replaces full-window recomputation with efficient updates as new frames arrive.
- **Real-time performance**: Supports continuous frame-by-frame output at the camera’s frame rate.
- **Low latency**: Enables real-time imaging and on-the-fly adjustments.

### Performance
The optimized implementation achieves significant speedups compared to the default:

| Image Size | Default FPS | Optimized FPS |
| ---------- | ----------- | ------------- |
| 2048×2048  | <1 FPS      | **18 FPS**    |
| 1024×1024  | <1 FPS      | **73 FPS**    |
