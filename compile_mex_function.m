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
  src/regularization/IterativeSolver.cpp ...
  src/regularization/OpticalFlow/OpticalFlow.cpp ...
  src/regularization/OpticalFlow/OpticalFlowDiffusion.cpp ...
  src/regularization/OpticalFlow/OpticalFlowCurvature.cpp ...
  src/regularization/OpticalFlow/OpticalFlowElastic.cpp ...
  src/regularization/Demons/Demons.cpp ...
  src/regularization/Demons/OpticalFlowThirionsDemons.cpp ...
  src/regularization/Demons/OpticalFlowLogDemons.cpp ...
  src/ImageRegistration.cpp ...
  src/ImageRegistrationOpticalFlow.cpp ...
  src/ImageRegistrationDemons.cpp ...
  -L[pwd, "/fftw"] ...
  -I[pwd, "/fftw"] ...
  -lfftw3

disp("Compilation success!");
