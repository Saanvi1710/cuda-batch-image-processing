# CUDA-Accelerated Batch Image Processing

## Project Overview

This project implements a GPU-accelerated image-processing pipeline using CUDA.

The pipeline processes a batch of 100 synthetic 1024x1024 RGB images using:

1. Grayscale conversion
2. 3x3 Gaussian blur
3. Sobel edge detection

Both CPU and GPU implementations are provided. Execution time is recorded for performance comparison.

## Technologies

- C++
- CUDA
- OpenCV
- NVIDIA GPU
- Python for dataset generation and analysis

## Project Structure

cuda-image-processing/
- src/
  - main.cu
  - kernels.cu
  - kernels.cuh
  - cpu_processor.cpp
- data/
  - input/
  - output/
- results/
  - performance.csv
- screenshots/
- Makefile
- run.sh
- README.md

## CUDA Implementation

### Grayscale Conversion

Each CUDA thread processes one image pixel and converts the pixel to grayscale.

### Gaussian Blur

A 3x3 Gaussian-style filter is applied to reduce image noise and smooth the image.

### Sobel Edge Detection

The Sobel operator calculates horizontal and vertical gradients to detect edges.

The CUDA implementation assigns image pixels to CUDA threads using 2D thread blocks.

## CPU Baseline

The same three processing stages are implemented using standard C++ loops.

This provides a CPU baseline for performance comparison.

## Dataset

The experiment uses 100 procedurally generated 1024x1024 images containing geometric patterns such as lines and circles.

The dataset is generated locally so that the experiment can be reproduced without requiring an external image dataset.

## Build

The project requires:

- NVIDIA CUDA Toolkit
- OpenCV
- C++17 compiler
- NVIDIA GPU

Example compilation:

    nvcc -O3 --std=c++17 src/main.cu src/kernels.cu src/cpu_processor.cpp -o gpu_image_processor $(pkg-config --cflags --libs opencv4)

## Run

    ./gpu_image_processor ./data/input ./data/output ./results/performance.csv

The program accepts:

- Input directory
- Output directory
- Performance CSV path

## Performance Results

The experiment processed 100 images.

| Metric | Result |
|---|---:|
| Images processed | 100 |
| Average CPU time | 38.2229 ms |
| Average GPU kernel time | 1.38168 ms |
| Measured kernel speedup | 27.6641x |

The GPU timing reported here measures CUDA kernel execution time for the image-processing stages.

## Output

For every input image, the GPU produces a processed image containing the result of the grayscale, Gaussian blur, and Sobel pipeline.

Performance measurements are stored in:

    results/performance.csv

The CSV contains per-image CPU time, GPU kernel time, and calculated speedup.

## Lessons Learned

- Image-processing operations can be naturally parallelized because individual pixels can often be processed independently.
- CUDA threads can be mapped directly to image coordinates using 2D thread blocks.
- GPU kernel execution can provide substantial acceleration for repetitive pixel-level operations.
- Memory transfers and kernel execution should be considered separately when evaluating GPU performance.
- Kernel optimization techniques such as shared memory and constant memory can provide further optimization opportunities.

## Future Improvements

Possible future improvements include:

- Using shared memory for neighborhood-based filters.
- Using CUDA constant memory for convolution kernels.
- Processing multiple images concurrently using CUDA streams.
- Comparing different block dimensions.
- Measuring complete GPU execution time including host-device memory transfers.
- Testing the implementation on larger real-world datasets.
