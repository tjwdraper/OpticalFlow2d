#include <src/regularization/IterativeSolver.h>
#include <src/gradients.h>

// Constructors and deconstructors
IterativeSolver::IterativeSolver(const dim dimin) {
    // Get the dimensions and size of the images
    this->dimin  = dimin;
    this->sizein = this->dimin.x * this->dimin.y;
    this->step   = dim(1, this->dimin.x);

    // Allocate memory for image gradients
    this->gradI = new Motion(this->dimin);
    this->It = new Image(this->dimin);
}

IterativeSolver::~IterativeSolver() {
    delete this->gradI;
    delete this->It;
}

// Image derivatives
void IterativeSolver::spatial_derivative(Motion* grad_image, const Image* image) const {
    // Get the dimensions and the step size of the image
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    float *I     = image->get_image();
    vector2d *dI = grad_image->get_motion();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            dI[idx] = vector2d(gradients::partial_x(I, idx, i, dimin),
                               gradients::partial_y(I, idx, j, dimin));
        }
    }

    // Done
    return;
}

void IterativeSolver::temporal_derivative(Image* It, const Image* Iref, const Image* Imov) const {
    *It = *Imov - *Iref;

    // Done
    return;
}

void IterativeSolver::set_derivatives(const Image* Iref, const Image* Imov) const {
    this->IterativeSolver::spatial_derivative(this->gradI, Imov);
    this->IterativeSolver::temporal_derivative(this->It, Iref, Imov);
}