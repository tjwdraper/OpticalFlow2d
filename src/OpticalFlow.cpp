#include <src/OpticalFlow.h>

#include <src/gradients.h>

// Constructors and deconstructors
OpticalFlow::OpticalFlow(const dim dimin, const float alpha) : ImageRegistrationSolver(dimin, alpha) {
    // Allocate memory for the laplacian field
    this->qlaplacian = new Motion(this->dimin);
}

OpticalFlow::~OpticalFlow() {
    delete this->qlaplacian;
}


// Do one iteration
void OpticalFlow::get_quasi_laplacian(const Motion* motion) {
    // Get the dimensions and step size of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    vector2d *qlap  = this->qlaplacian->get_motion();
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

void OpticalFlow::horn_schunck_iteration(Motion *motion) {
    // Get the dimensions and step size of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    vector2d *mo    = motion->get_motion();
    vector2d *qlap  = this->qlaplacian->get_motion();
    vector2d *dI    = this->gradI->get_motion();
    float *It       = this->It->get_image();

    // Get the regularisation parameter
    const float alphasq = alpha * alpha;

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            mo[idx] = qlap[idx] - dI[idx] * ( (It[idx] + dI[idx].x * qlap[idx].x + dI[idx].y * qlap[idx].y) / (alphasq + dI[idx].x*dI[idx].x + dI[idx].y*dI[idx].y) );
        }
    }

    // Done
    return;
}

void OpticalFlow::get_update(Motion *motion) {
    // Get the laplacian map (without the central contribution)
    this->get_quasi_laplacian(motion);

    // Use this map, the images and the motion field to get the next iteration
    this->horn_schunck_iteration(motion);

    // Done
    return;
}