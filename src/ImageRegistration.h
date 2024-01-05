#ifndef _IMAGE_REGISTRATION_H_
#define _IMAGE_REGISTRATION_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>
#include <src/OpticalFlow.h>
#include <src/SolverOptions.h>

class ImageRegistration {
    public:
        // Constructors and deconstructors
        ImageRegistration(const dim dimin, 
                          const int nscales, const int* niter, const int nrefine, 
                          const Regularisation reg, const float* regparams, const unsigned int nparams);
        ~ImageRegistration();

        // Getters and setters
        void set_reference_image(const Image& im);
        void set_moving_image(const Image& im);
        Motion* get_estimated_motion() const;

        // Copy the estimated motion
        void copy_estimated_motion(Motion& mo) const;

        // Estimate motion
        void estimate_motion();

    private:
        void display_registration_parameters(const Regularisation reg, const float* regparams, const unsigned int nparams) const;

        void estimate_motion_at_current_resolution(Motion* motion, 
                                                    const Image *Iref, Image *Imov,
                                                    OpticalFlow *solver, 
                                                    const int niter,
                                                    const dim dimin, const int sizein);

        dim *dimin;
        int *sizein;
        int nscales;
        int *niter;
        int nrefine;

        OpticalFlow **solver;

        Image **Iref;
        Image **Imov;

        Motion **motion;
};

#endif