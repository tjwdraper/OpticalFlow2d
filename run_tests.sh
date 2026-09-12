#!/bin/bash

# clean up before building
rm -rf build
mkdir -p build

# compile source files into object files
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/mainTest.cpp -o ./build/mainTest.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testCoord2d.cpp -o ./build/testCoord2d.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testField.cpp -o ./build/testField.o
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testImage.cpp -o ./build/testImage.o

# create test executable
g++ -std=c++20 ./build/mainTest.o ./build/testCoord2d.o ./build/testField.o ./build/testImage.o -o ./build/testCoord2d -L/usr/local/lib -lgtest

# run tests
./build/testCoord2d