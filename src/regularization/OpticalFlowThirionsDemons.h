#ifndef _OPTICAL_FLOW_THIRIONS_DEMONS_H_
#define _OPTICAL_FLOW_THIRIONS_DEMONS_H_

#include <src/regularization/OpticalFlow.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlowThirionsDemons : public OpticalFlow {
    public:
        // Constructors and deconstructors
        OpticalFlowThirionsDemons(const dim dimin, 
            const float sigma_i = 1.0, const float sigma_x = 0.25, 
            const float sigma_diffusion = 2.0, const float sigma_fluid = 2.0,
            const unsigned int kernelwidth = 5);
        ~OpticalFlowThirionsDemons();

        // Overload method from base class
        void get_update(Motion *motion);

        void get_update(Motion* motion, const Image* Iref, const Image* Imov);

    private:
        // Create convolution kernels
        void create_gaussian_kernel(double *kernel, const float sigma) const; // Move outside class

        // Smooth motion field with Gaussian convolution
        void convolute(Motion *motion, const double *kernel) const; // Perhaps better to move to Motion class later

        // Do one iteration with the Demons algorithm
        void demons_iteration(Motion *motion);

        // Auxiliary fields
        Image *Iwar;

        Motion *force;
        Motion *correspondence;

        // Demons parameters
        float sigma_i;
        float sigma_x;
        float sigma_diffusion;
        float sigma_fluid;

        // Convolution kernels
        dim dimkernel;
        dim stepkernel;
        unsigned int sizekernel;

        double *kernel_diffusion;
        double *kernel_fluid;
};

#endif