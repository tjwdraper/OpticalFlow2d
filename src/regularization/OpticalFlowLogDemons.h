#ifndef _OPTICAL_FLOW_LOG_DEMONS_H_
#define _OPTICAL_FLOW_LOG_DEMONS_H_

#include <src/regularization/Demons.h>

class OpticalFlowLogDemons : public Demons {
    public:
        // Constructors and deconstructors
        OpticalFlowLogDemons(const dim dimin, 
            const float sigma_i = 1.0, const float sigma_x = 0.25, 
            const float sigma_diffusion = 2.0, const float sigma_fluid = 2.0,
            const unsigned int kernelwidth = 5);
        ~OpticalFlowLogDemons();

        // Overload method from base class
        void get_update(Motion *motion, const Image *Iref, const Image* Imov);

    private:
        // The the exponent of a vector field
        void expfield(Motion *motion) const; // Move outside of class
};

#endif