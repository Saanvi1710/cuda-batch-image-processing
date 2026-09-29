CXX = nvcc
CXXFLAGS = -O3 --std=c++17

OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4)

TARGET = gpu_image_processor

SRC = src/main.cu src/kernels.cu src/cpu_processor.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(OPENCV_FLAGS)

run:
	./$(TARGET) ./data/input ./data/output ./results/performance.csv

clean:
	rm -f $(TARGET)
