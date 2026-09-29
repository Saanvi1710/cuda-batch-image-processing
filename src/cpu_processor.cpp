
#include <opencv2/opencv.hpp>
#include <string>

void cpuGrayscale(
    const cv::Mat& input,
    cv::Mat& output
) {
    output.create(input.rows, input.cols, CV_8UC1);

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            const cv::Vec3b& pixel = input.at<cv::Vec3b>(y, x);

            float gray =
                0.299f * pixel[2] +
                0.587f * pixel[1] +
                0.114f * pixel[0];

            output.at<unsigned char>(y, x) =
                static_cast<unsigned char>(gray);
        }
    }
}


void cpuGaussianBlur(
    const cv::Mat& input,
    cv::Mat& output
) {
    output.create(input.rows, input.cols, CV_8UC1);

    const float kernel[3][3] = {
        {1.0f, 2.0f, 1.0f},
        {2.0f, 4.0f, 2.0f},
        {1.0f, 2.0f, 1.0f}
    };

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            float sum = 0.0f;
            float weight = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    int px = std::min(
                        std::max(x + kx, 0),
                        input.cols - 1
                    );

                    int py = std::min(
                        std::max(y + ky, 0),
                        input.rows - 1
                    );

                    float w = kernel[ky + 1][kx + 1];

                    sum += input.at<unsigned char>(py, px) * w;
                    weight += w;
                }
            }

            output.at<unsigned char>(y, x) =
                static_cast<unsigned char>(sum / weight);
        }
    }
}


void cpuSobel(
    const cv::Mat& input,
    cv::Mat& output
) {
    output.create(input.rows, input.cols, CV_8UC1);

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

    for (int y = 0; y < input.rows; y++) {
        for (int x = 0; x < input.cols; x++) {

            int gradientX = 0;
            int gradientY = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {

                    int px = std::min(
                        std::max(x + kx, 0),
                        input.cols - 1
                    );

                    int py = std::min(
                        std::max(y + ky, 0),
                        input.rows - 1
                    );

                    int value =
                        input.at<unsigned char>(py, px);

                    gradientX +=
                        value * gx[ky + 1][kx + 1];

                    gradientY +=
                        value * gy[ky + 1][kx + 1];
                }
            }

            int magnitude =
                std::abs(gradientX) +
                std::abs(gradientY);

            output.at<unsigned char>(y, x) =
                static_cast<unsigned char>(
                    std::min(magnitude, 255)
                );
        }
    }
}


void runCPU(
    const cv::Mat& input,
    cv::Mat& output
) {
    cv::Mat gray;
    cv::Mat blurred;

    cpuGrayscale(input, gray);
    cpuGaussianBlur(gray, blurred);
    cpuSobel(blurred, output);
}
