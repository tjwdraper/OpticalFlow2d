#include <include/regularization/IterativeSolver.h>
#include <include/gradients.h>

// Constructors and deconstructors
IterativeSolver::IterativeSolver(const dim dimin, const double alpha) {
    // Get the dimensions and size of the images
    this->dimin  = dimin;
    this->sizein = this->dimin.x * this->dimin.y;
    this->step   = dim(1, this->dimin.x);

    // Allocate memory for image gradients
    this->gradI = new Motion(this->dimin);
    this->It = new Image(this->dimin);
    this->force = new Motion(this->dimin);
    this->qdiffoperator = new Motion(this->dimin);
    this->alpha = alpha;
}

IterativeSolver::~IterativeSolver() {
    delete this->gradI;
    delete this->It;
    delete this->force;
    delete this->qdiffoperator;
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

// Construct the force from the image gradients and motion estimate
void IterativeSolver::get_force(Motion* force, const Motion* motion) const {
    // Get the dimensions and the step size of the image
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    vector2d *f     = force->get_motion();
    vector2d *u     = motion->get_motion();
    vector2d *dI = this->gradI->get_motion();
    float *It    = this->It->get_image();


    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            f[idx] = dI[idx] * (It[idx] + u[idx].x * dI[idx].x + u[idx].y * dI[idx].y) ;
        }
    }

    // Done
    return;
}

// Define the quasi differential operator of this method
void IterativeSolver::get_quasi_differential_operator(const Motion* motion) {
    // Get the dimensions and step size of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    vector2d *qlap  = this->qdiffoperator->get_motion();
    vector2d *mo    = motion->get_motion();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            qlap[idx] = gradients::qlaplacian(mo, idx, i, j, dimin);
        }
    }

    // Done
    return;
}

// Get the update using the iterative method
void IterativeSolver::get_update(Motion *motion, const Image* Iref, const Image* Imov) {
    // Get the laplacian map (without the central contribution)
    this->get_quasi_differential_operator(motion);

    // Get the force using the quasi-laplacian
    this->get_force(this->force, this->qdiffoperator);

    // Use this map, the images and the motion field to get the next iteration
    this->optical_flow_iteration(motion);

    // Done
    return;
}

void IterativeSolver::optical_flow_iteration(Motion *motion) {
    // Get the dimensions and step size of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    vector2d *u     = motion->get_motion();
    vector2d *qdiff = this->qdiffoperator->get_motion();
    vector2d *dI    = this->gradI->get_motion();
    float *It       = this->It->get_image();
    vector2d *f     = this->force->get_motion();

    // Get the regularisation parameter
    const float alphasq = alpha * alpha;

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u[idx] = qdiff[idx] - f[idx] / (alphasq + dI[idx].x*dI[idx].x + dI[idx].y*dI[idx].y);
        }
    }

    // Done
    return;
}