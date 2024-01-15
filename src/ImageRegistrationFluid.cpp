#include <src/ImageRegistrationFluid.h>

// Constructors and deconstructors
ImageRegistrationFluid::ImageRegistrationFluid(const dim dimin, 
    const int nscales, const int* niter, const int nrefine, 
    const Regularisation reg, const float* regparams, const unsigned int nparams,
    const Verbose verbose) {

}

ImageRegistrationFluid::~ImageRegistrationFluid() {

}

bool ImageRegistrationFluid::valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const {

}

void ImageRegistrationFluid::set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams) {

}

void ImageRegistrationFluid::estimate_motion_at_current_resolution(Motion* motion, 
    const Image *Iref, Image *Imov,
    IterativeSolver *solver, 
    const int niter,
    const dim dimin, const int sizein) {

}