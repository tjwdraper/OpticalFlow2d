#include <src/OpticalFlowDiffusion.h>

#include <src/gradients.h>

// Constructors and deconstructors
OpticalFlowDiffusion::OpticalFlowDiffusion(const dim dimin, const float alpha) : OpticalFlow(dimin, alpha) {}

OpticalFlowDiffusion::~OpticalFlowDiffusion() {}

// Define the quasi differential operator of this method
void OpticalFlowDiffusion::get_quasi_differential_operator(const Motion* motion) {
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