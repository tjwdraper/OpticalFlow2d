clear functions

if isfile("OpticalFlow2d.mex");
  delete("OpticalFlow2d.mex");
end

disp("Compiling...");

mkoctfile --mex -o "OpticalFlow2d.mex" main.cpp Image.cpp Motion.cpp OpticalFlow.cpp OpticalFlowSolver.cpp

disp("Compilation success!");
