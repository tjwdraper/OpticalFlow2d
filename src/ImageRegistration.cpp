#include <src/ImageRegistration.h>
#include <src/OpticalFlowDiffusion.h>
#include <src/OpticalFlowCurvature.h>
#include <src/OpticalFlowElastic.h>
#include <src/Logger.h>

#include <mex.h>
#include <cstring>

void ImageRegistration::display_registration_parameters(const Regularisation reg, const float alpha) const {
    mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
    mexPrintf("Optical flow image registration started... (2D C++ implementation)...\n");
    mexPrintf("Registration parameters:\n");
    mexPrintf("dimensions:\t\t\t\t(%d %d)\n", this->dimin[0].x, this->dimin[0].y);
    mexPrintf("niter:\t\t\t\t\t(%d", this->niter[0]);
    for (int s = 1; s < this->nscales+1; s++) {
        mexPrintf(" %d", this->niter[s]);
    }
    mexPrintf(")\n");
    mexPrintf("nscales:\t\t\t\t%d\n", this->nscales);
    mexPrintf("nrefine:\t\t\t\t%d\n", this->nrefine);
    switch(reg) {
        case Regularisation::Diffusion: mexPrintf("regularisation:\tdiffusion\n"); break;
        case Regularisation::Curvature: mexPrintf("regularisation:\tcurvature\n"); break;
        case Regularisation::Elastic:   mexPrintf("regularisation:\telastic\n"); break;
    }
    mexPrintf("alpha:\t\t\t\t\t%.2f\n", alpha);
    mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");
}

ImageRegistration::ImageRegistration(const dim dimin, const int nscales, const int* niter, const int nrefine, const Regularisation reg, const float alpha) {
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
    this->solver = new OpticalFlow*[nscales + 1];
    for (int s = nscales; s >= 0; s--) {
        switch(reg) {
            case Regularisation::Diffusion: this->solver[s] = new OpticalFlowDiffusion(this->dimin[s], alpha); break;
            case Regularisation::Curvature: this->solver[s] = new OpticalFlowCurvature(this->dimin[s], alpha); break;
            case Regularisation::Elastic:   this->solver[s] = new OpticalFlowElastic(this->dimin[s], alpha); break;
        }
    }

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
    display_registration_parameters(reg, alpha);
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

