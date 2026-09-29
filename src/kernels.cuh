
#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

void launchGrayscale(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
);

void launchGaussianBlur(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
);

void launchSobel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
);

#endif
