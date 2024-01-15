#include <src/ImageRegistration.h>

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
        case Regularisation::Diffusion:           mexPrintf("regularisation:\t\t\t\tDiffusion\n");            break;
        case Regularisation::Curvature:           mexPrintf("regularisation:\t\t\t\tCurvature\n");            break;
        case Regularisation::Elastic:             mexPrintf("regularisation:\t\t\t\tElastic\n");              break;
        case Regularisation::ThirionsDemons:      mexPrintf("regularisation:\t\t\t\tThirions Demons\n");      break;
        case Regularisation::DiffeomorphicDemons: mexPrintf("regularisation:\t\t\t\tDiffeomorphic Demons\n"); break;
        case Regularisation::Fluid:               mexPrintf("regularisation:\t\t\t\tFluid\n");                break;
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

    // Done
    return;
}

ImageRegistration::ImageRegistration(const dim dimin, 
    const int nscales, const int* niter, const int nrefine, 
    const Regularisation reg, const float* regparams, const unsigned int nparams,
    const Verbose verbose) {
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

    // Allocate image and motion 
    this->Iref = new Image*[nscales + 1];
    this->Imov = new Image*[nscales + 1];
    this->motion = new Motion*[nscales + 1];
    for (int s = nscales; s >= 0; s--) {
        this->Iref[s] = new Image(this->dimin[s]);
        this->Imov[s] = new Image(this->dimin[s]);
        this->motion[s] = new Motion(this->dimin[s]);
    }

    // Display registration settings
    this->ImageRegistration::display_registration_parameters(reg, regparams, nparams);

    // Set the verbose option
    this->verbose = verbose;
}

ImageRegistration::~ImageRegistration() {
    delete[] this->dimin;
    delete[] this->sizein;

    for (int s = this->nscales; s >= 0; s--) {
        delete this->Iref[s];
        delete this->Imov[s];
        delete this->motion[s];
    }
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

