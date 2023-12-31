#include <src/regularization/OpticalFlow.h>
#include <src/gradients.h>


// Constructors and deconstructors
OpticalFlow::OpticalFlow(const dim dimin) {
    // Get the dimensions and size of the images
    this->dimin  = dimin;
    this->sizein = this->dimin.x * this->dimin.y;
    this->step   = dim(1, this->dimin.x);

    // Allocate memory for the spatial and temporal image gradients
    this->gradI = new Motion(this->dimin);
    this->It = new Image(this->dimin);
}

OpticalFlow::~OpticalFlow() {
    delete this->gradI;
    delete this->It;
}

// Get the spatial and temporal image gradients
void OpticalFlow::get_image_gradients(const Image* Iref, const Image* Imov) {
    // Get the dimensions and the step size of the image
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get a copy of the pointer to the data of the vector fields
    float *R     = Iref->get_image();
    float *T     = Imov->get_image();
    vector2d *dI = this->gradI->get_motion();
    float *It    = this->It->get_image();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            dI[idx] = vector2d(gradients::partial_x(T, idx, i, dimin),
                               gradients::partial_y(T, idx, j, dimin));

            It[idx] = T[idx] - R[idx];
        }
    }

    // Done
    return;
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