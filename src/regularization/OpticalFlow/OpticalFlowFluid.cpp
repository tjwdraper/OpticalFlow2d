#include <src/regularization/OpticalFlow/OpticalFlowFluid.h>
#include <src/Logger.h>
#include <src/gradients.h>

#define PI 3.14159265

void OpticalFlowFluid::set_eigenvalues() {
    // Get regularisation parameters
    const float& tau = this->tau;
    const float mu = this->mu * tau;
    const float lambda = this->lambda * tau;

    // Get the dimensions and step size etc.
    const dim& dimin = this->dimin;
    const dim& step = this->step_rm;
    const unsigned int sizein = this->sizein;

    // Iterate over the Fourier spectrum
    unsigned int idx;

    float T1, T2;
    float invsig1, invsig2;
    float eig_laplacian;

    for (unsigned int p = 0; p < dimin.x; p++) {
        for (unsigned int q = 0; q < dimin.y; q++) {
            // Get the index in the eigenvalue matrix
            idx = p * step.x + q * step.y;

            // Get the eigenvalue of the laplacian
            eig_laplacian = -4 + 2*cos(PI*p/dimin.x) + 2*cos(PI*q/dimin.y);

            // Get the components of the off-diagonal contribution
            T1 = sin(PI*p/dimin.x);
            T2 = sin(PI*q/dimin.y);

            // Get the components of the diagonal matrix
            invsig1 = 1 - (2*mu+lambda)*eig_laplacian - (mu+lambda)*T1;
            invsig2 = 1 - (2*mu+lambda)*eig_laplacian - (mu+lambda)*T2;

            float denom = 1.0f + (mu+lambda)*(invsig1 * T1 * T1 + invsig2 * T2 * T2);

            // Get the eigenvalue of the curvature operator
            this->eigenvalues[idx + 0*sizein] = invsig1 - (mu+lambda) * invsig1 * T1 * invsig1 * T1 / denom;
            this->eigenvalues[idx + 1*sizein] = invsig2 - (mu+lambda) * invsig2 * T2 * invsig2 * T2 / denom;
            this->eigenvalues[idx + 2*sizein] = -(mu+lambda) * invsig1 * T1 * invsig2 * T2;
        }
    }

    // Done
    return;
}

OpticalFlowFluid::OpticalFlowFluid(const dim dimin, const float mu, const float lambda, const float tau) : OpticalFlow(dimin) {
    // Set regularisation and model parameters
    this->mu = mu;
    this->lambda = lambda;
    this->tau = tau;

    // Get the image dimensions
    this->step_cm = dim(1, this->dimin.x);
    this->step_rm = dim(this->dimin.y, 1);

    // Allocate memory for the auxiliary fields
    this->rhs_x = new double[this->sizein];
    this->rhs_y = new double[this->sizein];

    this->eigenvalues = new double[3*this->sizein];

    // Set the eigenvalues
    this->OpticalFlowFluid::set_eigenvalues();

    // FFTW plans
    this->pf_x = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_x, this->rhs_x, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    this->pf_y = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_y, this->rhs_y, FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    this->pb_x = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_x, this->rhs_x, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    this->pb_y = fftw_plan_r2r_2d(this->dimin.x, this->dimin.y, this->rhs_y, this->rhs_y, FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);

    // Allocate memory for the velocity field
    this->velocity = new Motion(this->dimin);
    this->increment = new Motion(this->dimin);
}

OpticalFlowFluid::~OpticalFlowFluid() {
    delete[] this->rhs_x;
    delete[] this->rhs_y;
    delete[] this->eigenvalues;
    
    fftw_destroy_plan(this->pf_x);
    fftw_destroy_plan(this->pf_y);
    fftw_destroy_plan(this->pb_x);
    fftw_destroy_plan(this->pb_y);

    delete this->velocity;
    delete this->increment;
}

void OpticalFlowFluid::construct_rhs(const Motion* motion) {
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

void OpticalFlowFluid::construct_motion(Motion *motion) const {
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

void OpticalFlowFluid::multiply_eigenvalues() {
    // Get the number of voxels
    const unsigned int& sizein = this->sizein;

    // Iterate over all voxels
    double Axx, Axy, Ayy, freq_p, freq_q;
    for (unsigned int i = 0; i < sizein; i++) {
        Axx = this->eigenvalues[i + 0 * sizein];
        Axy = this->eigenvalues[i + 1 * sizein];
        Ayy = this->eigenvalues[i + 2 * sizein];

        freq_p = this->rhs_x[i];
        freq_q = this->rhs_y[i];

        this->rhs_x[i] = Axx * freq_p + Axy * freq_q;
        this->rhs_y[i] = Axy * freq_p + Ayy * freq_q;
    }

    // Done
    return;
}

void OpticalFlowFluid::get_increment(const Motion* motion) {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step = this->step_cm;

    // Get copy of pointer to the motion and force field data
    vector2d *mo  = motion->get_motion();
    vector2d* vel = this->velocity->get_motion();
    vector2d* R = this->increment->get_motion();

    // Iterate over voxels
    unsigned int idx;
    vector2d dudx, dudy;
    vector2d u, v;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u = mo[idx];
            v = vel[idx];

            dudx = gradients::partial_x(mo, idx, i, dimin);
            dudy = gradients::partial_y(mo, idx, j, dimin);

            R[idx] = v - dudx * v.x - dudy * v.y;
        }
    }

    // Done
    return;
}

void OpticalFlowFluid::estimate_timestep() {
    this->timestep = this->dumax / this->increment->maxabs();
}

void OpticalFlowFluid::integrate(Motion *motion) const {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step = this->step_cm;

    // Time step
    const float& timestep = this->timestep;

    // Get copy of pointer to the motion and force field data
    vector2d *u = motion->get_motion();
    vector2d *R = this->increment->get_motion();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u[idx] += R[idx] * timestep;
        }
    }

    // Done
    return;
}

void OpticalFlowFluid::get_update(Motion* motion, const Image* Iref, const Image* Imov) {
    // Get the force
    this->OpticalFlow::get_force(this->force, motion);

    // Get the velocity field
    this->OpticalFlowFluid::construct_rhs(this->velocity);

    fftw_execute_r2r(this->pf_x, this->rhs_x, this->rhs_x);
    fftw_execute_r2r(this->pf_y, this->rhs_y, this->rhs_y);

    this->OpticalFlowFluid::multiply_eigenvalues();

    fftw_execute_r2r(this->pb_x, this->rhs_x, this->rhs_x);
    fftw_execute_r2r(this->pb_y, this->rhs_y, this->rhs_y);

    this->OpticalFlowFluid::construct_motion(this->velocity);

    // Integrate the material derivative equation to get the next iteration of the motion field
    this->OpticalFlowFluid::get_increment(motion);

    this->OpticalFlowFluid::estimate_timestep();

    this->OpticalFlowFluid::integrate(motion);
}