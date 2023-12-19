clc;
clear functions

if isfile("OpticalFlow2d.mex");
  delete("OpticalFlow2d.mex");
end

disp("Compiling...");

mkoctfile --mex -o "OpticalFlow2d.mex" WrapperOpticalFlow2d.cpp src/Image.cpp src/Motion.cpp src/Logger.cpp src/ImageRegistration.cpp src/ImageRegistrationSolver.cpp src/OpticalFlow.cpp src/OpticalFlowSpectral.cpp

disp("Compilation success!");
