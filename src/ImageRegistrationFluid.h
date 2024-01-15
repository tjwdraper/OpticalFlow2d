#ifndef _IMAGE_REGISTRATION_FLUID_H_
#define _IMAGE_REGISTRATION_FLUID_H_

#include <src/ImageRegistration.h>

class ImageRegistrationFluid : public ImageRegistration {
    public:
        // Constructors and deconstructors
        ImageRegistrationFluid(const dim dimin, 
                          const int nscales, const int* niter, const int nrefine, 
                          const Regularisation reg, const float* regparams, const unsigned int nparams,
                          const Verbose verbose);
        ~ImageRegistrationFluid();

    private:
        bool valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const;

        void set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams);

        void estimate_motion_at_current_resolution(Motion* motion, 
                                                   const Image *Iref, Image *Imov,
                                                   IterativeSolver *solver, 
                                                   const int niter,
                                                   const dim dimin, const int sizein);
};

#endif