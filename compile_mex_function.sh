#!/bin/bash

set -e

rm -f OpticalFlow2d.mex

echo "Compiling..."

mkoctfile --mex -o build/OpticalFlow2d.mex src/WrapperOpticalFlow2d.cpp src/ImageRegistration.cpp src/IterativeSolver.cpp 

echo "Compilation successful!"

octave --persist opticalflow2d_middlebury.m