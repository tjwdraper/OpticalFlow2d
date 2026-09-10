#include <src/regularization/OpticalFlow/OpticalFlow.h>
#include <src/gradients.h>


// Constructors and deconstructors
OpticalFlow::OpticalFlow(const dim dimin, const double alpha) : IterativeSolver(dimin) {
    this->force = new Motion(this->dimin);
    this->qdiffoperator = new Motion(this->dimin);
    this->alpha = alpha;
}

OpticalFlow::~OpticalFlow() {
    delete this->force;
    delete this->qdiffoperator;
}

// Construct the force from the image gradients and motion estimate
void OpticalFlow::get_force(Motion* force, const Motion* motion) const {
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
void OpticalFlow::get_quasi_differential_operator(const Motion* motion) {
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
void OpticalFlow::get_update(Motion *motion, const Image* Iref, const Image* Imov) {
    // Get the laplacian map (without the central contribution)
    this->get_quasi_differential_operator(motion);

    // Get the force using the quasi-laplacian
    this->get_force(this->force, this->qdiffoperator);

    // Use this map, the images and the motion field to get the next iteration
    this->optical_flow_iteration(motion);

    // Done
    return;
}

void OpticalFlow::optical_flow_iteration(Motion *motion) {
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