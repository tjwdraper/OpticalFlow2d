g++ -std=c++20 -O3 -I. opticalflow2d_middlebury.cpp src/ImageRegistration.cpp src/IterativeSolver.cpp -o ./build/opticalflow2d_middlebury
./build/opticalflow2d_middlebury config_middlebury.json