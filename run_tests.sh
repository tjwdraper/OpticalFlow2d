#!/bin/bash

# clean up before building
rm -rf build
mkdir -p build

# compile source files into object files
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/mainTest.cpp -o ./build/mainTest.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testCoord2d.cpp -o ./build/testCoord2d.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testField.cpp -o ./build/testField.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testImage.cpp -o ./build/testImage.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testMotion.cpp -o ./build/testMotion.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/integration/testConv2d.cpp -o ./build/testConv2d.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/integration/testSeparableConv2d.cpp -o ./build/testSeparableConv2d.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/integration/testInterp2d.cpp -o ./build/testInterp2d.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/integration/testGradients.cpp -o ./build/testGradients.o

# create test executable
g++ -std=c++20 ./build/mainTest.o ./build/testCoord2d.o ./build/testField.o ./build/testImage.o ./build/testMotion.o ./build/testConv2d.o ./build/testSeparableConv2d.o ./build/testInterp2d.o ./build/testGradients.o -o ./build/testOpticalFlow2d -L/usr/local/lib -lgtest

# run tests
./build/testOpticalFlow2d