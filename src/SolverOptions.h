#ifndef _SOLVER_OPTIONS_H_
#define _SOLVER_OPTIONS_H_

enum Regularisation {Diffusion = 0, Curvature = 1, Elastic = 2, ThirionsDemons = 3, DiffeomorphicDemons = 4, Fluid = 5};

enum Verbose {Off = 0, On = 1};

enum MotionAccumulation {Composition = 0, Addition = 1};

#endif