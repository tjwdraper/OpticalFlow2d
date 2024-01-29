#include <src/regularization/Demons/Demons.h>

// Constructors and deconstructors
Demons::Demons(const dim dimin, 
            const float sigma_i, const float sigma_x, 
            const float sigma_diffusion, const float sigma_fluid,
            const unsigned int kernelwidth) : IterativeSolver(dimin) {
    // Auxiliary fields
    this->Iwar = new Image(this->dimin);
    this->correspondence = new Motion(this->dimin);

    // Demons parameters
    this->sigma_i = sigma_i;
    this->sigma_x = sigma_x;
    this->sigma_diffusion = sigma_diffusion;
    this->sigma_fluid = sigma_fluid;

    // Create convolution kernels and use Gaussian filtering
    this->kernel_diffusion = new Kernel(kernelwidth);
    this->kernel_fluid     = new Kernel(kernelwidth);

    this->kernel_diffusion->set_gaussian(this->sigma_diffusion);
    this->kernel_fluid->set_gaussian(this->sigma_fluid);
}

Demons::~Demons() {
    delete this->Iwar;
    delete this->correspondence;
    delete this->kernel_diffusion;
    delete this->kernel_fluid;
}

// Do one iteration with the Demons algorithm
void Demons::demons_iteration(Motion *motion) {
    // Get the field dimensions
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the motion, correspondence and force data
    vector2d *u = this->correspondence->get_motion();
    vector2d *s = motion->get_motion();

    // Get a copy of the pointer to the image gradient data
    float *It = this->It->get_image();
    vector2d *dI = this->gradI->get_motion();

    // Regularisation parameters
    const float& sigma_xsq = this->sigma_x * this->sigma_x;
    const float& sigma_isq = this->sigma_i * this->sigma_i;

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u[idx] = dI[idx] * It[idx] / (dI[idx].x * dI[idx].x + dI[idx].y * dI[idx].y + It[idx]*It[idx] * sigma_isq / sigma_xsq) * -1;
        
        }
    }

    // Done
    return;
}