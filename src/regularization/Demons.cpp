#include <src/regularization/Demons.h>

// Create Gaussian kernels
void Demons::create_gaussian_kernel(double *kernel, const float sigma) const {
    // Get kernel dimensions
    const dim& dimkernel = this->dimkernel;
    const dim& stepkernel = this->stepkernel;
    const int& sizekernel = this->sizekernel;

    // Get the center of the kernel
    int cx = (dimkernel.x - 1) / 2;
    int cy = (dimkernel.y - 1) / 2; 

    unsigned int idx;
    double weight = 0;
    for (int i = 0; i < dimkernel.x; i++) {
        for (int j = 0; j < dimkernel.y; j++) {
            idx = i * stepkernel.x + j * stepkernel.y;

            kernel[idx] = exp(- ((i-cx)*(i-cx) + (j-cy)*(j-cy)) / (2*sigma*sigma));
            weight += kernel[idx];
        }
    }

    // Normalize the weights
    for (int i = 0; i < sizekernel; i++) {
        kernel[i] /= weight;
    }

    // Done
    return;
}

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

    // Convolution kernels
    this->dimkernel = dim(kernelwidth, kernelwidth);
    this->stepkernel = dim(1, this->dimkernel.x);
    this->sizekernel = this->dimkernel.x * this->dimkernel.y;

    // Allocate memory for the convolution kernels
    this->kernel_diffusion = new double[this->sizekernel];
    this->kernel_fluid = new double[this->sizekernel];

    // Set values
    this->Demons::create_gaussian_kernel(this->kernel_diffusion, this->sigma_diffusion);
    this->Demons::create_gaussian_kernel(this->kernel_fluid, this->sigma_fluid);
}

Demons::~Demons() {
    delete this->Iwar;
    delete this->correspondence;
    delete[] this->kernel_diffusion;
    delete[] this->kernel_fluid;
}

// Smooth motion field with Gaussian convolution
void Demons::convolute(Motion *motion, const double *kernel) const {
    // Get the dimensions of the field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get the dimensions of the kernel
    const dim& dimkernel = this->dimkernel;
    const dim& stepkernel = this->stepkernel;

    const int cx = (dimkernel.x - 1) / 2;
    const int cy = (dimkernel.y - 1) / 2;

    // Make a copy of the motion field
    Motion tmp(*motion);

    // Get a copy to the pointer to the motion data
    vector2d *u = motion->get_motion();
    vector2d *t = tmp.get_motion();

    // Iterate over voxels
    int idx, idxkernel;
    for (int i = 0; i < dimin.x; i++) {
        for (int j = 0; j < dimin.y; j++) {
            // Get absolute index
            idx = i * step.x + j * step.y;

            // Iterate over kernel
            vector2d val;
            double weight = 0.0f;
            for (int ii = -cx; ii <= cx; ii++) {
                for (int jj = -cy; jj <= cy; jj++) {
                    // Check if we are within bounds
                    if ((i + ii) * step.x + (j + jj) * step.y < 0 ||
                        (i + ii) * step.x + (j + jj) * step.y >= sizein) {
                        continue;
                    } 
                    else {
                        // Get absolute index in the kernel
                        idxkernel = (ii + cx) * stepkernel.x + (jj + cy) * stepkernel.y;

                        // Add contribution of the kernel element
                        val += t[idx + ii * step.x + jj * step.y] * kernel[idxkernel];
                        weight += kernel[idxkernel];
                    }
                }
            }

            // Set value in motion field
            if (weight != 0) {
                u[idx] = val / weight;
            }
        }
    }

    // Done
    return;
}

// Do one iteration with the Demons algorithm
void Demons::demons_iteration(Motion *motion) {
    // Get the field dimensions
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the motion, correspondence and force data
    vector2d *u = this->correspondence->get_motion();
    vector2d *s = motion->get_motion();
    //vector2d *f = force->get_motion();

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