#include <src/OpticalFlowDiffusion.h>

#include <src/gradients.h>

// Constructors and deconstructors
OpticalFlowDiffusion::OpticalFlowDiffusion(const dim dimin, const float alpha) {
    // Get the dimensions and size of the images
    this->dimin  = dimin;
    this->sizein = this->dimin.x * this->dimin.y;
    this->step   = dim(1, this->dimin.x);

    // Get the registration parameters
    this->alpha = alpha;

    // Allocate memory for the laplacian field
    this->qlaplacian = new Motion(this->dimin);

    // Allocate memory for the spatial and temporal image gradients
    this->gradI = new Motion(this->dimin);
    this->It = new Image(this->dimin);
}

OpticalFlowDiffusion::~OpticalFlowDiffusion() {
    delete this->qlaplacian;

    delete this->gradI;
    delete this->It;
}

// Get the spatial and temporal image gradients
void OpticalFlowDiffusion::get_image_gradients(const Image* Iref, const Image* Imov) {
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

// Do one iteration
void OpticalFlowDiffusion::get_quasi_laplacian(const Motion* motion) {
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

void OpticalFlowDiffusion::horn_schunck_iteration(Motion *motion) {
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

void OpticalFlowDiffusion::get_update(Motion *motion) {
    // Get the laplacian map (without the central contribution)
    this->get_quasi_laplacian(motion);

    // Use this map, the images and the motion field to get the next iteration
    this->horn_schunck_iteration(motion);

    // Done
    return;
}