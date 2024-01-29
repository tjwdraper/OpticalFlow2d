#include <src/regularization/Demons/DemonsThirions.h>

// Constructors and deconstructors
DemonsThirions::DemonsThirions(const dim dimin, 
    const float sigma_i, const float sigma_x,
    const float sigma_diffusion, const float sigma_fluid,
    const unsigned int kernelwidth,
    const MotionAccumulation motion_accumulation_method) : Demons(dimin, 
        sigma_i, sigma_x,
        sigma_diffusion, sigma_fluid,
        kernelwidth) {
    this->motion_accumulation_method = motion_accumulation_method;
}

DemonsThirions::~DemonsThirions() {}

// Overload method from base class
void DemonsThirions::get_update(Motion *motion, const Image* Iref, const Image* Imov) {
    // Warp the input image with current estimate of the motion field
    *this->Iwar = *Imov;
    this->Iwar->warp2d(*motion);

    // Get the image gradients
    this->IterativeSolver::set_derivatives(Iref, this->Iwar);

    // Execute Demons iteration - calculate the correspondence update
    this->Demons::demons_iteration(motion);

    // Smoothen the correpondence update
    this->correspondence->convolute(*this->kernel_fluid);

    // Update the motion field (Additive demons or Composite demons)
    if (this->motion_accumulation_method == MotionAccumulation::Composition) {
        motion->accumulate(*this->correspondence);
    }
    else if (this->motion_accumulation_method == MotionAccumulation::Addition) {
        *motion += *this->correspondence;
    }

    // Smoothen the motion field
    motion->convolute(*this->kernel_diffusion);
}