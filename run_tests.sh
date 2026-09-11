#!/bin/bash

# clean up before building
rm -rf build
mkdir -p build

# compile source files into object files
g++ -std=c++20 -I. -I/usr/local/include -c -g ./tests/unit/testCoord2d.cpp -o ./build/testCoord2d.o

# create test executable
g++ -std=c++20 ./build/testCoord2d.o -o ./build/testCoord2d -L/usr/local/lib -lgtest

# run tests
./build/testCoord2d