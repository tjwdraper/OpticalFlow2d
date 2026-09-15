# OpticalFlow2d: A C++ implementation of Horn-Schunck optical flow for 2D deformable image registration

OpticalFlow2d estimates a deformation vector field (DVF) quantifying the motion between two misaligned images. This project provides a C++ implementation of the Horn-Schunck optical flow method to estimate the DVF. This method estimates the DVF as the minimizer of the cost function:

$\displaystyle \mathbf{u}^* = \underset{\mathbf{u}}{\textnormal{arg min}} \frac{1}{2}\int(I_t + \mathbf{u}\cdot\nabla I)^2\mathrm{d}\mathbf{x} + \frac{\alpha}{2} \int \lVert D\mathbf{u}\rVert_F^2\mathrm{d}\mathbf{x}$

Some details on the model implementation:
1. Coarse-to-fine multiresolution method for estimation of large deformation
2. Gaussian filtering between resolution levels for anti-aliasing.
3. Convergence if relative improvement is below user-defined threshold.
4. Image gradients ($I_t$ and $\nabla I$) are estimated with separable convolution kernels.
5. Follows the efficient numerical model derived from the Euler-Lagrange equations.
6. (Operations on) the Image and Motion class are implemented in header-only (.hpp) files, which should allow for an easy implementation of other image processing tasks in future projects.

# Compilation

Using the MEX API, the WrapperOpticalFlow2d.cpp interacts between variables from the Matlab/GNU Octave workspace, and the ImageRegistration class. Running from the working directory (Windows, Powershell):

```
.\compile_mex_function.ps1
```

or (Linux)

```
bash compile_mex_function.sh
```

creates a executable (.mex, .mexw64 or .mexa64) that can be launched from Matlab/GNU Octave. These files require access to the mex/mkoctfile compilers for Matlab and GNU Octave respectively.

The working directory contains two test files (*.m) to run the Horn-Schunck optical flow method on the Middlebury image alignment dataset.

# Syntax: Execute from Matlab/GNU Octave

Running the Horn-Schunck optical flow method can be done with the following steps:

```
config.struct()
config.size_image = int32(size(Iref));
config.niter      = int32([100 200 400]); // 400->200->100 iterations
config.alpha      = 0.5;
config.eps        = 1e-3;
config.nrefine    = 0;

OpticalFlow2d(config); // Initialize
```

Estimation of the DVF is then performed using:

```
OpticalFlow2d(Iref, Imov);
```

The registered image and the estimated DVF can be returned through:

```
motion = OpticalFlow2d();
Ireg = OpticalFlow2d(Imov);
```

To close the registration library:

```
OpticalFlow2d();
```

# Code testing

The tests directory contains unit, integration and system tests for the OpticalFlow2D codebase. Tests have been written using the Google's gtest testing framework. Tests can be compiled and run (on Linux) through

```
bash run_tests.sh
```

Which requires access to the gtest.h and gtest library during compilation. Alter the path in the bash script accordingly if required.

