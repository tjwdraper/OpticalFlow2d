#include <src/ImageRegistrationOpticalFlow.h>
#include <src/Logger.h>

#include <src/regularization/OpticalFlowDiffusion.h>
#include <src/regularization/OpticalFlowCurvature.h>
#include <src/regularization/OpticalFlowElastic.h>

void ImageRegistrationOpticalFlow::display_registration_parameters(const Regularisation reg, const float* regparams, const unsigned int nparams) const {
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
        case Regularisation::Diffusion: mexPrintf("regularisation:\t\t\t\tdiffusion\n"); break;
        case Regularisation::Curvature: mexPrintf("regularisation:\t\t\t\tcurvature\n"); break;
        case Regularisation::Elastic:   mexPrintf("regularisation:\t\t\t\telastic\n"); break;
        case Regularisation::ThirionsDemons: mexPrintf("regularisation:\t\t\t\tThirions Demons\n"); break;
        case Regularisation::LogDemons: mexPrintf("regularisation:\t\t\t\tLog-Demons:\n"); break;
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

bool ImageRegistrationOpticalFlow::valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const {
    return ((reg == Regularisation::Diffusion) && (nparams == 1)) || 
           ((reg == Regularisation::Curvature) && (nparams >= 1) && (nparams <= 2)) ||
           ((reg == Regularisation::Elastic) && (nparams >= 2) && (nparams <= 3));
}

void ImageRegistrationOpticalFlow::set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams) {
    if (!this->ImageRegistrationOpticalFlow::valid_regularisation_parameters(reg, nparams)) {
        throw std::invalid_argument("Invalid number of regularisation parameters for given regularisation method.\n");
    }

    this->solver = new OpticalFlow*[this->nscales + 1];
    for (int s = this->nscales; s >= 0; s--) {
        switch(reg) {
            case Regularisation::Diffusion: {
                // Get the regularisation parameter
                const float& alpha = regparams[0];

                // Set the solver
                this->solver[s] = new OpticalFlowDiffusion(this->dimin[s], alpha); 

                // Done
                break;
            }
            case Regularisation::Curvature: { 
                // Get the regularisation parameter
                const float &alpha = regparams[0];

                // Check if relaxation parameter is passed
                if (nparams == 1) {
                    this->solver[s] = new OpticalFlowCurvature(this->dimin[s], alpha);
                }
                else {
                    const float& omega = regparams[1];

                    this->solver[s] = new OpticalFlowCurvature(this->dimin[s], alpha, omega);                    
                }

                // Done
                break;
            }
            case Regularisation::Elastic: {
                // Get the regularisation parameters
                const float& mu = regparams[0];
                const float& lambda = regparams[1];

                // Check if time-step parameter is passed
                if (nparams != 3) {
                    this->solver[s] = new OpticalFlowElastic(this->dimin[s], mu, lambda);
                }
                else {
                    const float& tau = regparams[2];

                    this->solver[s] = new OpticalFlowElastic(this->dimin[s], mu, lambda, tau);
                }
                
                // Done
                break;
            }
        }
    }
}

ImageRegistrationOpticalFlow::ImageRegistrationOpticalFlow(const dim dimin, 
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
    this->ImageRegistrationOpticalFlow::set_solver(reg, regparams, nparams);

    // Show loaded registration parameters to terminal
    this->ImageRegistrationOpticalFlow::display_registration_parameters(reg, regparams, nparams);
}

ImageRegistrationOpticalFlow::~ImageRegistrationOpticalFlow() {
    for (int s = this->nscales; s >= 0; s--) {
        delete this->solver[s];
    }
    delete[] this->solver;
}

void ImageRegistrationOpticalFlow::estimate_motion_at_current_resolution(Motion* motion, 
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

        // Calculating the image gradients only has to be done once
        solver->get_image_gradients(Iref, Iaux);

        // Iterate over resolution levels
        for (int iter = 0; iter < niter; iter++) {
            // Calculate the update step
            solver->get_update(motion_est);

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