
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "kernels.cuh"

namespace fs = std::filesystem;

void runCPU(
    const cv::Mat& input,
    cv::Mat& output
);

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t error = call;                             \
        if (error != cudaSuccess) {                           \
            std::cerr << "CUDA error: "                         \
                      << cudaGetErrorString(error) << "\n";   \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


double processGPU(
    const cv::Mat& input,
    cv::Mat& output
) {
    int width = input.cols;
    int height = input.rows;

    size_t inputBytes =
        width * height * 3 * sizeof(unsigned char);

    size_t outputBytes =
        width * height * sizeof(unsigned char);

    unsigned char* d_input = nullptr;
    unsigned char* d_gray = nullptr;
    unsigned char* d_blur = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_CHECK(cudaMalloc(&d_gray, outputBytes));
    CUDA_CHECK(cudaMalloc(&d_blur, outputBytes));
    CUDA_CHECK(cudaMalloc(&d_output, outputBytes));

    CUDA_CHECK(cudaMemcpy(
        d_input,
        input.data,
        inputBytes,
        cudaMemcpyHostToDevice
    ));

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    launchGrayscale(
        d_input,
        d_gray,
        width,
        height
    );

    launchGaussianBlur(
        d_gray,
        d_blur,
        width,
        height
    );

    launchSobel(
        d_blur,
        d_output,
        width,
        height
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    output.create(height, width, CV_8UC1);

    CUDA_CHECK(cudaMemcpy(
        output.data,
        d_output,
        outputBytes,
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_blur));
    CUDA_CHECK(cudaFree(d_output));

    return milliseconds;
}


int main(int argc, char** argv) {

    std::string inputDir = "../data/input";
    std::string outputDir = "../data/output";
    std::string resultsFile = "../results/performance.csv";

    if (argc > 1)
        inputDir = argv[1];

    if (argc > 2)
        outputDir = argv[2];

    if (argc > 3)
        resultsFile = argv[3];

    fs::create_directories(outputDir);
    fs::create_directories(
        fs::path(resultsFile).parent_path()
    );

    std::vector<fs::path> imageFiles;

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file())
            continue;

        std::string extension =
            entry.path().extension().string();

        if (extension == ".png" ||
            extension == ".jpg" ||
            extension == ".jpeg") {
            imageFiles.push_back(entry.path());
        }
    }

    std::sort(imageFiles.begin(), imageFiles.end());

    if (imageFiles.empty()) {
        std::cerr << "No images found in "
                  << inputDir << "\n";
        return EXIT_FAILURE;
    }

    std::cout << "CUDA Batch Image Processing\n";
    std::cout << "Images: " << imageFiles.size() << "\n";

    std::ofstream csv(resultsFile);

    csv << "image,cpu_ms,gpu_kernel_ms,speedup\n";

    double totalCPU = 0.0;
    double totalGPU = 0.0;

    for (size_t i = 0; i < imageFiles.size(); i++) {

        cv::Mat input =
            cv::imread(
                imageFiles[i].string(),
                cv::IMREAD_COLOR
            );

        if (input.empty()) {
            std::cerr << "Could not read "
                      << imageFiles[i] << "\n";
            continue;
        }

        cv::Mat cpuOutput;
        cv::Mat gpuOutput;

        auto cpuStart =
            std::chrono::high_resolution_clock::now();

        runCPU(input, cpuOutput);

        auto cpuEnd =
            std::chrono::high_resolution_clock::now();

        double cpuMs =
            std::chrono::duration<double, std::milli>(
                cpuEnd - cpuStart
            ).count();

        double gpuMs =
            processGPU(input, gpuOutput);

        double speedup =
            cpuMs / gpuMs;

        totalCPU += cpuMs;
        totalGPU += gpuMs;

        std::string outputName =
            (fs::path(outputDir) /
             imageFiles[i].filename()).string();

        cv::imwrite(outputName, gpuOutput);

        csv << imageFiles[i].filename().string()
            << ","
            << cpuMs
            << ","
            << gpuMs
            << ","
            << speedup
            << "\n";

        std::cout
            << "Processed "
            << (i + 1)
            << "/"
            << imageFiles.size()
            << " | CPU: "
            << cpuMs
            << " ms | GPU: "
            << gpuMs
            << " ms | Speedup: "
            << speedup
            << "x\n";
    }

    double averageCPU =
        totalCPU / imageFiles.size();

    double averageGPU =
        totalGPU / imageFiles.size();

    double overallSpeedup =
        averageCPU / averageGPU;

    csv.close();

    std::cout << "\n===== Performance Summary =====\n";
    std::cout << "Images processed: "
              << imageFiles.size() << "\n";

    std::cout << "Average CPU time: "
              << averageCPU << " ms\n";

    std::cout << "Average GPU kernel time: "
              << averageGPU << " ms\n";

    std::cout << "Overall speedup: "
              << overallSpeedup << "x\n";

    std::cout << "Results saved to: "
              << resultsFile << "\n";

    return EXIT_SUCCESS;
}
