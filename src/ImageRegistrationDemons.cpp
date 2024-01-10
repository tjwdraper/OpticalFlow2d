#include <src/ImageRegistrationDemons.h>
#include <src/Logger.h>

#include <src/regularization/OpticalFlowThirionsDemons.h>

void ImageRegistrationDemons::display_registration_parameters(const Regularisation reg, const float* regparams, const unsigned int nparams) const {
    mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    mexPrintf("Optical flow image registration started... (2D C++ implementation)...\n");
    mexPrintf("Registration parameters:\n");

    // Image dimensions and multiresolution parameters
    mexPrintf("dimensions:\t\t\t\t(%d %d)\n", this->dimin[0].x, this->dimin[0].y);
    mexPrintf("niter:\t\t\t\t\t(%d", this->niter[0]);
    for (int s = 1; s < this->nscales+1; s++) {
        mexPrintf(" %d", this->niter[s]);
    }
    mexPrintf(")\n");
    mexPrintf("nscales:\t\t\t\t%d\n", this->nscales);
    mexPrintf("nrefine:\t\t\t\t%d\n", this->nrefine);

    // Regularisation method
    switch(reg) {
        case Regularisation::ThirionsDemons: mexPrintf("regularisation:\t\t\t\tThirions Demons\n"); break;
    }

    // Regularization parameters
    if (nparams == 1) {
        mexPrintf("reg. param:\t\t\t\t%.2f\n", regparams[0]);
    }
    else {
        mexPrintf("reg. params:\t\t\t\t(%.2f", regparams[0]);
        for (unsigned int p = 1; p < nparams; p++) {
            mexPrintf(" %.2f", regparams[p]);
        }
        mexPrintf(")\n");
    }

    mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");
}

bool ImageRegistrationDemons::valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const {
    return ((reg == Regularisation::ThirionsDemons) && (nparams == 5));
}

void ImageRegistrationDemons::set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams) {
    if (!this->ImageRegistrationDemons::valid_regularisation_parameters(reg, nparams)) {
        throw std::invalid_argument("Invalid number of regularisation parameters for given regularisation method.\n");
    }

    this->solver = new OpticalFlow*[this->nscales + 1];
    for (int s = this->nscales; s >= 0; s--) {
        switch(reg) {
            case Regularisation::ThirionsDemons: {
                // Get the regularisation parameter
                const float& sigma_i = regparams[0];
                const float& sigma_x = regparams[1];
                const float& sigma_diffusion = regparams[2];
                const float& sigma_fluid = regparams[3];
                const unsigned int kernelwidth = regparams[4];

                // Set solver
                this->solver[s] = new OpticalFlowThirionsDemons(this->dimin[s],
                    sigma_i, sigma_x,
                    sigma_diffusion, sigma_fluid,
                    kernelwidth);

                // Done
                break;
            }
        }
    }
}

ImageRegistrationDemons::ImageRegistrationDemons(const dim dimin, 
    const int nscales, 
    const int* niter, 
    const int nrefine, 
    const Regularisation reg, 
    const float* regparams, 
    const unsigned int nparams) : ImageRegistration(dimin,
        nscales,
        niter,
        nrefine,
        reg,
        regparams,
        nparams) {
    // Solver object
    this->ImageRegistrationDemons::set_solver(reg, regparams, nparams);

    // Show loaded registration parameters to terminal
    this->ImageRegistrationDemons::display_registration_parameters(reg, regparams, nparams);
}

ImageRegistrationDemons::~ImageRegistrationDemons() {
    for (int s = this->nscales; s >= 0; s--) {
        delete this->solver[s];
    }
    delete[] this->solver;
}

void ImageRegistrationDemons::estimate_motion_at_current_resolution(Motion* motion, 
    const Image *Iref, Image *Imov,
    OpticalFlow *solver, 
    const int niter,
    const dim dimin, const int sizein) {
    
    // Create auxiliary motion field
    Image *Iaux = new Image(dimin);

    // Create auxiliary motion field
    Motion *motion_est = new Motion(dimin);

    for (int refine = 0; refine < this->nrefine; refine++) {
        // Reset Iaux to input image
        *Iaux = *Imov;

        // Warp moving image with accumulated motion field
        Iaux->warp2d(*motion);

        // Create a Logger object
        Logger log(dimin, niter);

        // Iterate over resolution levels
        for (int iter = 0; iter < niter; iter++) {
            // Calculate the update step
            solver->get_update(motion_est, Iref, Iaux);

            // Calculate the difference between iterations
            log.update_error(motion_est);

            // Converge check
            if ((log.get_error_at_current_iteration() < 0.001f) &&
                (iter > 1)) {
                break;
            }
        }

        // Accumulate motion field
        motion->accumulate(*motion_est);

        // Reset auxiliary field
        motion_est->reset();

    }

    // Free up the mem
    delete motion_est;
    delete Iaux;
    
    // Done
    return;
}