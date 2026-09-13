#!/bin/bash

set -e

rm -f OpticalFlow2d.mex

echo "Compiling..."

mkoctfile --mex -o OpticalFlow2d.mex WrapperOpticalFlow2d.cpp src/ImageRegistration.cpp src/IterativeSolver.cpp 

echo "Compilation successful!"