#include <src/OpticalFlowCurvature.h>
#include <src/gradients.h>

#define PI 3.14159265

void OpticalFlowCurvature::set_eigenvalues() {
    // Get regularisation parameters
    const float& alpha = this->alpha;
    const float& tau = this->tau;

    // Get the dimensions and step size etc.
    const dim& dimin = this->dimin;
    const dim& step = this->step_rm;
    const unsigned int sizein = this->sizein;

    // Iterate over the Fourier spectrum
    unsigned int idx;
    for (unsigned int p = 0; p < dimin.x; p++) {
        for (unsigned int q = 0; q < dimin.y; q++) {
            // Get the index in the eigenvalue matrix
            idx = p * step.x + q * step.y;

            // Get the eigenvalue of the curvature operator
            this->eigenvalues[idx] = 1.0f / (1.0f + tau * alpha * pow(-4 + 2*cos(p*PI/dimin.x) + 2*cos(q*PI/dimin.y), 2)); 
        }
    }

    // Done
    return;
}

// Constructors and deconstructors
OpticalFlowCurvature::OpticalFlowCurvature(const dim dimin, const float alpha) : OpticalFlow(dimin, alpha) {
    // Get the image dimensions
    this->step_cm = dim(1, this->dimin.x);
    this->step_rm = dim(this->dimin.y, 1);

    // Allocate memory for the auxiliary fields
    this->force = new Motion(this->dimin);

    this->rhs_x = new double[this->sizein];
    this->rhs_y = new double[this->sizein];

    this->eigenvalues = new double[this->sizein];

    // Set the eigenvalues
    this->OpticalFlowCurvature::set_eigenvalues();

    // FFTW plans
    this->pf_x = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_x, this->rhs_x, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    this->pf_y = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_y, this->rhs_y, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    this->pb_x = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_x, this->rhs_x, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    this->pb_y = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_y, this->rhs_y, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
}

OpticalFlowCurvature::~OpticalFlowCurvature() {
    delete this->force;

    delete this->rhs_x;
    delete this->rhs_y;

    delete this->eigenvalues;

    fftw_destroy_plan(this->pf_x);
    fftw_destroy_plan(this->pf_y);
    fftw_destroy_plan(this->pb_x);
    fftw_destroy_plan(this->pb_y);
}

void OpticalFlowCurvature::construct_rhs(const Motion* motion) {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step_rm = this->step_rm;
    const dim& step_cm = this->step_cm;

    // Get regularisation parameters
    const float& tau = this->tau;

    // Get copy of pointer to the motion and force field data
    vector2d *u = motion->get_motion();
    vector2d *f = this->force->get_motion();

    // Iterate over voxels
    unsigned int idx_rm, idx_cm;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx_rm = i * step_rm.x + j * step_rm.y;
            idx_cm = i * step_cm.x + j * step_cm.y;

            this->rhs_x[idx_rm] = u[idx_cm].x - tau * f[idx_cm].x;
            this->rhs_y[idx_rm] = u[idx_cm].y - tau * f[idx_cm].y;
        }
    }

    // Done
    return;
}

void OpticalFlowCurvature::construct_motion(Motion *motion) const {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step_rm = this->step_rm;
    const dim& step_cm = this->step_cm;
    const unsigned int& sizein = this->sizein;

    // Get copy of pointer to the motion and force field data
    vector2d *u = motion->get_motion();

    // Iterate over voxels
    unsigned int idx_rm, idx_cm;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx_rm = i * step_rm.x + j * step_rm.y;
            idx_cm = i * step_cm.x + j * step_cm.y;

            u[idx_cm] = vector2d(this->rhs_x[idx_rm],
                                 this->rhs_y[idx_rm]) / (4.0f*sizein);
        }
    }

    // Done
    return;

}

void OpticalFlowCurvature::multiply_eigenvalues() {
    // Get the number of voxels
    const unsigned int& sizein = this->sizein;

    // Iterate over all voxels
    double eigenvalue;
    for (unsigned int i = 0; i < sizein; i++) {
        eigenvalue = this->eigenvalues[i];

        this->rhs_x[i] *= eigenvalue;
        this->rhs_y[i] *= eigenvalue;
    }

    // Done
    return;
}

// Overload function from base class
void OpticalFlowCurvature::get_update(Motion *motion) {
    // Get the force
    this->OpticalFlow::get_force(this->force, motion);

    // Construct the rhs before DCT
    this->OpticalFlowCurvature::construct_rhs(motion);

    // Do the DCT on both components of the rhs
    fftw_execute_r2r(this->pf_x, this->rhs_x, this->rhs_x);
    fftw_execute_r2r(this->pf_y, this->rhs_y, this->rhs_y);

    // Multiply DCT components with eigenvalues of the curvature operator
    this->OpticalFlowCurvature::multiply_eigenvalues();

    // Do the inverse DCT
    fftw_execute_r2r(this->pb_x, this->rhs_x, this->rhs_x);
    fftw_execute_r2r(this->pb_y, this->rhs_y, this->rhs_y);

    // Reconstruct the motion field
    this->OpticalFlowCurvature::construct_motion(motion);

    // Done
    return;
}
