#include <src/ImageRegistrationOpticalFlow.h>
#include <src/Logger.h>

#include <src/regularization/OpticalFlow/OpticalFlowDiffusion.h>
#include <src/regularization/OpticalFlow/OpticalFlowCurvature.h>
#include <src/regularization/OpticalFlow/OpticalFlowElastic.h>

bool ImageRegistrationOpticalFlow::valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const {
    return ((reg == Regularisation::Diffusion) && (nparams == 1)) || 
           ((reg == Regularisation::Curvature) && (nparams >= 1) && (nparams <= 2)) ||
           ((reg == Regularisation::Elastic) && (nparams >= 2) && (nparams <= 3));
}

void ImageRegistrationOpticalFlow::set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams) {
    if (!this->ImageRegistrationOpticalFlow::valid_regularisation_parameters(reg, nparams)) {
        throw std::invalid_argument("Invalid number of regularisation parameters for given regularisation method.\n");
    }

    this->solver = new IterativeSolver*[this->nscales + 1];
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
    const unsigned int nparams,
    const Verbose verbose) : ImageRegistration(dimin,
        nscales,
        niter,
        nrefine,
        reg,
        regparams,
        nparams,
        verbose) {
    // Solver object
    this->ImageRegistrationOpticalFlow::set_solver(reg, regparams, nparams);
}

ImageRegistrationOpticalFlow::~ImageRegistrationOpticalFlow() {
    for (int s = this->nscales; s >= 0; s--) {
        delete this->solver[s];
    }
    delete[] this->solver;
}

void ImageRegistrationOpticalFlow::estimate_motion_at_current_resolution(Motion* motion, 
    const Image *Iref, Image *Imov,
    IterativeSolver *solver, 
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
        Logger log(dimin, niter, this->verbose);

        // Calculating the image gradients only has to be done once
        solver->set_derivatives(Iref, Iaux);

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