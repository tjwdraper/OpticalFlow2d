# OpticalFlow2d: A fast C++ library for 2-dimensional deformable image registration

This project implements several image registration algorithms into one single C++ library to estimate the optical flow between two-dimensional (medical) images. 

## Image registration as a minimization problem

All methods in this library estimate the optical flow by solving a minimization problem. Suppose the reference and moving image are denoted by $R$ and $T$ respectively, and the estimated motion field $u$ such that $T(x+u) = R(x)$. The transformation \(u\) is obtained by minimizing a two-part cost function:

$E[u] = \underset{\hat{u}}{\textnormal{arg min}} \textnormal{Sim}[\hat{u}] + \alpha \textnormal{Reg}[\hat{u}].$

This project focusses on mono-modal image registration methods. The data similarity metric is therefore chosen as the sum-of-squared distances (SSD) between the images, or a linearized version (L-SSD):

$\textnormal{Sim}[u] = ||T(x+u) - R(x)||$

$\textnormal{Sim}[u] = ||T(x) - R(x) + u\cdot\nabla T(x)||$

The regularisation imposes additional constraints of the motion field, weighted by the regularization weight $\alpha$. The minimization problem can be transformed to a partial differential equation via the Euler-Lagrange equation.

### Horn-Schunck model



### Curvature model

### Elastic model

### Thirion's Demons

### Diffeomorphic Demons

### Viscous fluid model


## Compilation

This library can be compiled and executed from 'GNU Octave' and 'MATLAB', using the 'mex' or 'mkoctfile' compiler respectively. 
During compilation, link against the fftw lib
Works on both Linux and Windows

## 