#ifndef _DEMONS_THIRIONS_H_
#define _DEMONS_THIRIONS_H_

#include <src/regularization/Demons/Demons.h>
#include <src/Image.h>
#include <src/Motion.h>
#include <src/SolverOptions.h>

class DemonsThirions : public Demons {
    public:
        // Constructors and deconstructors
        DemonsThirions(const dim dimin, 
            const float sigma_i = 1.0, const float sigma_x = 0.25, 
            const float sigma_diffusion = 2.0, const float sigma_fluid = 2.0,
            const unsigned int kernelwidth = 5,
            const MotionAccumulation motion_accumulation_method = MotionAccumulation::Composition);
        ~DemonsThirions();

        // Overload method from base class
        void get_update(Motion* motion, const Image* Iref, const Image* Imov);  

    private:
        MotionAccumulation motion_accumulation_method; 
};

#endif