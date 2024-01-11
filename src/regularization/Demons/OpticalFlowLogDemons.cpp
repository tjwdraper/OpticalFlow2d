#include <src/regularization/Demons/OpticalFlowLogDemons.h>

// Constructors and deconstructors
OpticalFlowLogDemons::OpticalFlowLogDemons(const dim dimin, 
    const float sigma_i, const float sigma_x,
    const float sigma_diffusion, const float sigma_fluid,
    const unsigned int kernelwidth) : Demons(dimin, 
        sigma_i, sigma_x,
        sigma_diffusion, sigma_fluid,
        kernelwidth) {}

OpticalFlowLogDemons::~OpticalFlowLogDemons() {}

// Scaling and squaring algorithm
void OpticalFlowLogDemons::expfield(Motion *motion) const {
    // Get the scale
    int N = static_cast<int>( std::ceil(1 + std::log2(motion->maxabs())) );
    N = std::max(N, 0);

    if (N > 20) {
        mexPrintf("%d\n", N);
        mexErrMsgTxt("Error: the number of scales is too large\n");
        return;
    }
    else {
        mexPrintf("Maxabs of correspondence: %.3f\tNumber of levels in expfield: %d\n", motion->maxabs(), N);
    }
    
    // Resize the input field
    double resizefactor = std::pow(2, -N);
    vector2d *u = motion->get_motion();
    for (unsigned int i = 0; i < this->sizein; i++) {
        u[i] *= resizefactor;
    }

    // Square
    Motion *Mtmp = new Motion(this->dimin);
    for (unsigned int n = 0; n < N; n++) {
        *Mtmp = *motion;

        motion->accumulate(*Mtmp);
    }
    delete Mtmp;
}

// Overload method from base class
void OpticalFlowLogDemons::get_update(Motion *motion, const Image* Iref, const Image* Imov) {
    // Warp the input image with current estimate of the motion field
    *this->Iwar = *Imov;
    this->Iwar->warp2d(*motion);

    // Get the image gradients
    this->IterativeSolver::set_derivatives(Iref, this->Iwar);

    // Execute Demons iteration - calculate the correspondence update
    this->Demons::demons_iteration(motion);

    // Smoothen the correpondence update
    this->Demons::convolute(this->correspondence, this->kernel_fluid);

    // Update the motion field (Diffeomorphic demons demons)
    this->OpticalFlowLogDemons::expfield(this->correspondence);
    motion->accumulate(*this->correspondence);

    // Smoothen the motion field
    this->Demons::convolute(motion, this->kernel_diffusion);
}