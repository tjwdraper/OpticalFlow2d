#ifndef _IMAGE_REGISTRATION_OPTICAL_FLOW_H_
#define _IMAGE_REGISTRATION_OPTICAL_FLOW_H_

#include <src/ImageRegistration.h>
#include <src/coord2d.h>
#include <src/regularization/IterativeSolver.h>
#include <src/Motion.h>
#include <src/Image.h>
#include <src/SolverOptions.h>

class ImageRegistrationOpticalFlow : public ImageRegistration {
    public:
        // Constructors and deconstructors
        ImageRegistrationOpticalFlow(const dim dimin, 
                          const int nscales, const int* niter, const int nrefine, 
                          const Regularisation reg, const float* regparams, const unsigned int nparams);
        ~ImageRegistrationOpticalFlow();

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