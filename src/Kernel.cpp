#include <src/Kernel.h>
#include <math.h>

// Constructors and deconstructors
Kernel::Kernel(const dim dimkernel) {
    this->dimkernel = dimkernel;
    this->stepkernel = dim(1, this->dimkernel.x);
    this->sizekernel = this->dimkernel.x * this->dimkernel.y;

    this->kernel = new double[this->sizekernel];
}

Kernel::Kernel(const unsigned int kernelwidth) {
    this->dimkernel = dim(kernelwidth, kernelwidth);
    this->stepkernel = dim(1, this->dimkernel.x);
    this->sizekernel = this->dimkernel.x * this->dimkernel.y;

    this->kernel = new double[this->sizekernel];
    
    //Kernel(dim(kernelwidth, kernelwidth));
}

Kernel::~Kernel() {
    delete[] this->kernel;
}

// Getters and setters
dim Kernel::get_dimensions() const {
    return this->dimkernel;
}

dim Kernel::get_step() const {
    return this->stepkernel;
}

unsigned int Kernel::get_size() const {
    return this->sizekernel;
}

double* Kernel::get_kernel() const {
    return this->kernel;
}

//
void Kernel::set_gaussian(const float sigma) {
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

            this->kernel[idx] = exp(- ((i-cx)*(i-cx) + (j-cy)*(j-cy)) / (2*sigma*sigma));
            weight += this->kernel[idx];
        }
    }

    // Normalize the weights
    for (int i = 0; i < sizekernel; i++) {
        this->kernel[i] /= weight;
    }

    // Done
    return;
}
        
void Kernel::set_average() {
    for (unsigned int i = 0; i < this->sizekernel; i++) {
        this->kernel[i] = 1.0f / (float) this->sizekernel;
    }

    // Done
    return;
}