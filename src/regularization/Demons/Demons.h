#ifndef _DEMONS_H_
#define _DEMONS_H_

#include <src/regularization/IterativeSolver.h>

class Demons : public IterativeSolver {
    public:
        // Constructors and deconstructors
        Demons(const dim dimin, 
            const float sigma_i = 1.0, const float sigma_x = 0.25, 
            const float sigma_diffusion = 2.0, const float sigma_fluid = 2.0,
            const unsigned int kernelwidth = 5);
        ~Demons();

        // Do one iteration
        virtual void get_update(Motion *motion, const Image* Iref, const Image* Imov) {};

    protected:
        // Create a Gaussian convolution kernel
        void create_gaussian_kernel(double *kernel, const float sigma) const;

        // Convolute motion field with kernel
        void convolute(Motion *motion, const double *kernel) const;

        // Do one iteration with the Demons algorithm
        void demons_iteration(Motion *motion);

        // Auxiliary fields
        Image *Iwar;
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