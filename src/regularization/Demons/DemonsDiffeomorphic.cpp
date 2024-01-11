#include <src/regularization/Demons/DemonsDiffeomorphic.h>

// Constructors and deconstructors
DemonsDiffeomorphic::DemonsDiffeomorphic(const dim dimin, 
    const float sigma_i, const float sigma_x,
    const float sigma_diffusion, const float sigma_fluid,
    const unsigned int kernelwidth) : Demons(dimin, 
        sigma_i, sigma_x,
        sigma_diffusion, sigma_fluid,
        kernelwidth) {}

DemonsDiffeomorphic::~DemonsDiffeomorphic() {}

// Overload method from base class
void DemonsDiffeomorphic::get_update(Motion *motion, const Image* Iref, const Image* Imov) {
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
    this->correspondence->exp();
    //this->DemonsDiffeomorphic::expfield(this->correspondence);
    motion->accumulate(*this->correspondence);

    // Smoothen the motion field
    this->Demons::convolute(motion, this->kernel_diffusion);
}