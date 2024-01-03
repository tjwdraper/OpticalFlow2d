clc;
clear functions

if isfile("OpticalFlow2d.mex");
  delete("OpticalFlow2d.mex");
end

disp("Compiling...");

mkoctfile --mex -o "OpticalFlow2d.mex" WrapperOpticalFlow2d.cpp src/Image.cpp src/Motion.cpp src/Logger.cpp src/OpticalFlow.cpp src/OpticalFlowDiffusion.cpp src/OpticalFlowCurvature.cpp src/OpticalFlowElastic.cpp src/ImageRegistration.cpp

disp("Compilation success!");
