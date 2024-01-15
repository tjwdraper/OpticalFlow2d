# Compiler settings
MEX := T:/Software/MATLAB/R2018a/bin/mex
MEX_WINDOWS := $(shell cygpath -m $(MEX))

# Directories
PWD_PATH = $(PWD)/
PWD_PATH_WINDOWS = $(shell cygpath -m $(PWD_PATH))
INCLUDE_DIRS = -I$(PWD_PATH_WINDOWS)/src/ -I$(PWD_PATH_WINDOWS)/src/regularization/ -I$(PWD_PATH_WINDOWS)/src/regularization/demons/ -I$(PWD_PATH_WINDOWS)/src/regularization/opticalflow/

# Source files
SRCS_CPU = $(wildcard $(PWD_PATH_WINDOWS)/*.cpp) \
			$(wildcard $(PWD_PATH_WINDOWS)/src/*.cpp) \
			$(wildcard $(PWD_PATH_WINDOWS)/src/regularization/*.cpp) \
			$(wildcard $(PWD_PATH_WINDOWS)/src/regularization/demons/*.cpp) \
			$(wildcard $(PWD_PATH_WINDOWS)/src/regularization/opticalflow/*.cpp)

# Compilation flags
FFTW_FLAGS = -IT:/Libraries/Documents/Werk/OpticalFlow2d/fftw/ -LT:/Libraries/Documents/Werk/OpticalFlow2d/fftw/ -lfftw3-3 
MEX_FLAGS = -I. -I$(PWD_PATH_WINDOWS) -IT:/Software/MATLAB/R2018a/extern/include -L. -LT:/Software/MATLAB/R2018a/extern/lib/win64/microsoft -llibmex -llibmat -llibmx -Xlinker -shared

# Output file name
MEX_OUTPUT = OpticalFlow2d.mexw64

# Commands
OpticalFlow2d:
	g++ -o $(MEX_OUTPUT) $(FFTW_FLAGS) $(MEX_FLAGS) $(INCLUDE_DIRS) $(SRCS_CPU)
	rm -f OpticalFlow2d.exp OpticalFlow2d.lib

clean:
	rm -f $(MEX_OUTPUT)