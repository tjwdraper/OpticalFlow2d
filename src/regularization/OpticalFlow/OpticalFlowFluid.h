#ifndef _OPTICAL_FLOW_FLUID_H_
#define _OPTICAL_FLOW_FLUID_H_

#include <src/regularization/opticalflow/OpticalFlow.h>
#include <src/Motion.h>
#include <fftw3.h>

class OpticalFlowFluid : public OpticalFlow {
    public:
        OpticalFlowFluid(const dim dimin, const float mu, const float lambda, const float tau = 1.0f);
        ~OpticalFlowFluid();

        // Overload method from base class
        void get_update(Motion* motion, const Image* Iref = NULL, const Image* Imov = NULL);

    private:
        void set_eigenvalues();

        void construct_rhs(const Motion *motion);

        void construct_motion(Motion *motion) const;

        void multiply_eigenvalues();

        void get_increment(const Motion* motion);

        void estimate_timestep();

        void integrate(Motion* motion) const;

        // Image dimensions in row- and column-major ordering
        dim step_rm;
        dim step_cm;

        // FFT plan for the forward and backward DCT
        fftw_plan pf_x, pf_y;
        fftw_plan pb_x, pb_y;

        // Auxiliary fields
        double *rhs_x, *rhs_y;

        // Eigenvalue matrix of the biharmonic operator
        double *eigenvalues;

        // Regularisation and relaxation parameters
        float mu;
        float lambda;
        float tau;
        
        // Adaptive timestep parameter
        float timestep;

        // Velocity field
        Motion *velocity;
        Motion *increment;
};

#endif