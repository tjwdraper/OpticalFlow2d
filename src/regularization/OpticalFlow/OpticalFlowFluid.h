#ifndef _OPTICAL_FLOW_FLUID_H_
#define _OPTICAL_FLOW_FLUID_H_

#include <src/regularization/OpticalFlow/OpticalFlow.h>
#include <src/Motion.h>
#include <fftw3.h>

class OpticalFlowFluid : public OpticalFlow {
    public:
        OpticalFlowFluid(const dim dimin, const float mu, const float lambda, const float omega = 0.66);
        ~OpticalFlowFluid();

        // Overload method from base class
        void get_update(Motion* motion, const Image* Iref = NULL, const Image* Imov = NULL);

    private:
        void SOR_iteration(Motion* motion) const;

        void get_increment(const Motion* motion);

        void estimate_timestep();

        void integrate(Motion* motion) const;

        // Regularisation and relaxation parameters
        float mu;
        float lambda;
        float omega;
        
        // Adaptive timestep parameter
        float timestep;
        const float dumax = 0.65f;

        // Velocity field
        Motion *velocity;
        Motion *increment;
};

#endif