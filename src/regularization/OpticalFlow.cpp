#include <src/regularization/OpticalFlow.h>
#include <src/gradients.h>


// Constructors and deconstructors
OpticalFlow::OpticalFlow(const dim dimin) : IterativeSolver(dimin) {
    this->force = new Motion(this->dimin);
}

OpticalFlow::~OpticalFlow() {
    delete this->force;
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