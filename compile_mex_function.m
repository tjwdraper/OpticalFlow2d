clc;
clear functions

if isfile("OpticalFlow2d.mex");
  delete("OpticalFlow2d.mex");
end

disp("Compiling...");

mkoctfile --mex ...
  -o "OpticalFlow2d.mex" ...
  WrapperOpticalFlow2d.cpp ...
  src/Image.cpp ...
  src/Motion.cpp ...
  src/Logger.cpp ...
  src/regularization/OpticalFlow.cpp ...
  src/regularization/OpticalFlowDiffusion.cpp ...
  src/regularization/OpticalFlowCurvature.cpp ...
  src/regularization/OpticalFlowElastic.cpp ...
  src/regularization/OpticalFlowThirionsDemons.cpp ...
  src/ImageRegistration.cpp ...
  src/ImageRegistrationOpticalFlow.cpp ...
  src/ImageRegistrationDemons.cpp ...
  -L[pwd, "/fftw"] ...
  -I[pwd, "/fftw"] ...
  -lfftw3

disp("Compilation success!");
