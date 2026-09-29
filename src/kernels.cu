
#include "kernels.cuh"

__global__ void grayscaleKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    int rgb_idx = idx * 3;

    float gray =
        0.299f * input[rgb_idx] +
        0.587f * input[rgb_idx + 1] +
        0.114f * input[rgb_idx + 2];

    output[idx] = static_cast<unsigned char>(gray);
}


__global__ void gaussianBlurKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const float kernel[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };

    float sum = 0.0f;
    float weight = 0.0f;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int idx = py * width + px;

            float w = kernel[ky + 1][kx + 1];

            sum += input[idx] * w;
            weight += w;
        }
    }

    output[y * width + x] =
        static_cast<unsigned char>(sum / weight);
}


__global__ void sobelKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    const int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    const int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    int gradientX = 0;
    int gradientY = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {

            int px = min(max(x + kx, 0), width - 1);
            int py = min(max(y + ky, 0), height - 1);

            int value = input[py * width + px];

            gradientX += value * gx[ky + 1][kx + 1];
            gradientY += value * gy[ky + 1][kx + 1];
        }
    }

    int magnitude = abs(gradientX) + abs(gradientY);

    output[y * width + x] =
        static_cast<unsigned char>(min(magnitude, 255));
}


void launchGrayscale(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    grayscaleKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}


void launchGaussianBlur(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    gaussianBlurKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}


void launchSobel(
    const unsigned char* d_input,
    unsigned char* d_output,
    int width,
    int height
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    sobelKernel<<<grid, block>>>(
        d_input,
        d_output,
        width,
        height
    );
}
