#include <src/ImageRegistration.h>
#include <src/regularization/OpticalFlowDiffusion.h>
#include <src/regularization/OpticalFlowCurvature.h>
#include <src/regularization/OpticalFlowElastic.h>
#include <src/regularization/OpticalFlowThirionsDemons.h>
#include <src/Logger.h>

#include <mex.h>
#include <cstring>

void ImageRegistration::display_registration_parameters(const Regularisation reg, const float* regparams, const unsigned int nparams) const {
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

bool ImageRegistration::valid_regularisation_parameters(const Regularisation reg, const unsigned int nparams) const {
    return ((reg == Regularisation::Diffusion) && (nparams == 1)) || 
           ((reg == Regularisation::Curvature) && (nparams >= 1) && (nparams <= 2)) ||
           ((reg == Regularisation::Elastic) && (nparams >= 2) && (nparams <= 3)) ||
           ((reg == Regularisation::ThirionsDemons) && (nparams == 5));
}

void ImageRegistration::set_solver(const Regularisation reg, const float* regparams, const unsigned int nparams) {
    if (!this->ImageRegistration::valid_regularisation_parameters(reg, nparams)) {
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
            case Regularisation::ThirionsDemons: {
                // Get the regularisation parameters
                const float sigma_i = regparams[0];
                const float sigma_x = regparams[1];
                const float sigma_diffusion = regparams[2];
                const float sigma_fluid = regparams[3];
                const unsigned int kernelwidth = static_cast<unsigned int>(regparams[4]);

                this->solver[s] = new OpticalFlowThirionsDemons(this->dimin[s], 
                                        sigma_i, sigma_x, 
                                        sigma_diffusion, sigma_fluid,
                                        kernelwidth);
            }
        }
    }
}

ImageRegistration::ImageRegistration(const dim dimin, 
    const int nscales, const int* niter, const int nrefine, 
    const Regularisation reg, const float* regparams, const unsigned int nparams) {
    // Size and dimensions of the input image
    this->dimin = new dim[nscales + 1];
    this->sizein = new int[nscales + 1];
    for (int s = nscales; s >= 0; s--) {
        float scale = pow(2, s);
        this->dimin[s] = dim(dimin.x/scale,
                             dimin.y/scale);
        this->sizein[s] = this->dimin[s].x * this->dimin[s].y;
    }

    // Registration parameters
    this->nrefine = nrefine;
    this->nscales = nscales;
    this->niter = new int[nscales + 1];
    memcpy(this->niter, niter, (nscales+1)*sizeof(int));

    // Solver object
    this->ImageRegistration::set_solver(reg, regparams, nparams);

    // Allocate image and motion 
    this->Iref = new Image*[nscales + 1];
    this->Imov = new Image*[nscales + 1];
    this->motion = new Motion*[nscales + 1];
    for (int s = nscales; s >= 0; s--) {
        this->Iref[s] = new Image(this->dimin[s]);
        this->Imov[s] = new Image(this->dimin[s]);
        this->motion[s] = new Motion(this->dimin[s]);
    }

    // Show loaded registration parameters to terminal
    display_registration_parameters(reg, regparams, nparams);
}

ImageRegistration::~ImageRegistration() {
    delete[] this->dimin;
    delete[] this->sizein;

    for (int s = this->nscales; s >= 0; s--) {
        delete this->solver[s];
        delete this->Iref[s];
        delete this->Imov[s];
        delete this->motion[s];
    }
    delete[] this->solver;
    delete[] this->Iref;
    delete[] this->Imov;
    delete[] this->motion;

    delete[] this->niter;
}

// Getters and setters
void ImageRegistration::set_reference_image(const Image& im) {
    // Set the image at the largest resolution level
    *(this->Iref[0]) = im;

    // For the other levels, downsample:
    for (int s = this->nscales; s >= 1; s--) {
        this->Iref[s]->downSample(*this->Iref[0]);
    }
}

void ImageRegistration::set_moving_image(const Image& im) {
    // Set the image at the largest resolution level
    *(this->Imov[0]) = im;

    // For the other levels, downsample:
    for (int s = this->nscales; s >= 1; s--) {
        this->Imov[s]->downSample(*this->Imov[0]);
    }
}

Motion* ImageRegistration::get_estimated_motion() const {
    return this->motion[0];
}

// Copy the estimated motion 
void ImageRegistration::copy_estimated_motion(Motion& mo) const {
    mo = *this->motion[0];
}

// Estimate motion
void ImageRegistration::estimate_motion_at_current_resolution(Motion* motion, 
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

void ImageRegistration::estimate_motion() {    
    // Multiresolution pyramid
    for (int s = this->nscales; s >= 0; s--) {
        // Downsample motion
        if ((s > 0) && (s < this->nscales)) {
            this->motion[s]->downSample(*this->motion[0]);
        }

        // Estimate motion at current resolution level
        this->estimate_motion_at_current_resolution(this->motion[s],
                                                    this->Iref[s], this->Imov[s],
                                                    this->solver[s],
                                                    this->niter[s],
                                                    this->dimin[s], this->sizein[s]);

        // Upscale to next level in the pyramid
        if (s > 0) {
            this->motion[0]->upSample(*this->motion[s]);
        }
    }

    // Done
    return;
}

