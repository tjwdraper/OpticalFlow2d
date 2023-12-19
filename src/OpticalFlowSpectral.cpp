#include <src/OpticalFlowSpectral.h>
#include <src/gradients.h>

#define PI 3.14159265

// Set the eigenvalue matrix
void OpticalFlowSpectral::set_eigenvalues() {
    const float alphasq = this->alpha * this->alpha;

    // Get the dimensions and step size etc.
    const dim& dimin = this->dimin;
    const dim& dimcoefs = this->dimcoefs;
    const dim& stepcoefs = this->stepcoefs;
    const unsigned int sizein = this->sizein;
    const unsigned int sizecoefs = this->sizecoefs;

    // Iterate over Fourier spectru,
    for (unsigned int i = 0; i < dimcoefs.x; i++) {
        for (unsigned int j = 0; j < dimcoefs.y; j++) {
            double p = (double) (i + 0.5f) / (double) dimin.x - 0.5f;
            double q = (double) (j + 0.5f) / (double) dimin.y - 0.5f;

            // Get the norm of the wavenumber
            double psq = p*p + q*q;

            // FFT shift the index
            unsigned int i_shift = (i < dimcoefs.x/2 ? i + dimcoefs.x/2 : i - dimcoefs.x/2);
            unsigned int j_shift = dimcoefs.y - j - 1;

            // Get the index in the eigenvalue matrix
            unsigned int idx = i_shift * stepcoefs.x + j_shift * stepcoefs.y;

            // Prevent division by zero
            if ((p == 0) && (q == 0)) {
                this->eigenvalues[idx] = -1.0f;
                continue;
            }

            // Set eigenvalue value
            this->eigenvalues[idx] = -1.0f / (4.0f * PI * PI * alphasq * psq * sizein);
        }
    }

    // Done
    return;
}

// Constructors and deconstructors
OpticalFlowSpectral::OpticalFlowSpectral(const dim dimin, const float alpha) : ImageRegistrationSolver(dimin, alpha) {
    // Get image dimensions
    this->step_cm = dim(1, this->dimin.x);
    this->step_rm = dim(this->dimin.y, 1);

    // Get the dimensions of the FFT
    this->dimcoefs = dim(this->dimin.x, this->dimin.y/2+1);
    this->sizecoefs = this->dimcoefs.x * this->dimcoefs.y;
    this->stepcoefs = dim(this->dimcoefs.y, 1);

    // Allocate memory for the components of the force field
    this->comp_x = new double[this->sizein];
    this->comp_y = new double[this->sizein];

    // Get the eigenvalue matrix
    this->eigenvalues = new double[this->sizecoefs];
    
    this->set_eigenvalues();

    // FFTW coefficients
    this->coef_x = (fftw_complex *) fftw_malloc(this->sizecoefs*sizeof(fftw_complex));
    this->coef_y = (fftw_complex *) fftw_malloc(this->sizecoefs*sizeof(fftw_complex));

    // Set the FFTW plans
    this->pf_x = fftw_plan_dft_r2c_2d(this->dimin.x, this->dimin.y, this->comp_x, this->coef_x, FFTW_MEASURE);
    this->pf_y = fftw_plan_dft_r2c_2d(this->dimin.x, this->dimin.y, this->comp_y, this->coef_y, FFTW_MEASURE);

    this->pb_x = fftw_plan_dft_c2r_2d(this->dimin.x, this->dimin.y, this->coef_x, this->comp_x, FFTW_MEASURE);
    this->pb_y = fftw_plan_dft_c2r_2d(this->dimin.x, this->dimin.y, this->coef_y, this->comp_y, FFTW_MEASURE);
}

OpticalFlowSpectral::~OpticalFlowSpectral() {
    delete[] this->comp_x;
    delete[] this->comp_y;

    delete[] this->eigenvalues;

    fftw_free(this->coef_x);
    fftw_free(this->coef_y); 

    fftw_destroy_plan(this->pf_x);
    fftw_destroy_plan(this->pf_y);
    fftw_destroy_plan(this->pb_x);
    fftw_destroy_plan(this->pb_y);
}

void OpticalFlowSpectral::decompose(const Motion* mo) {
    // Get the dimensions and step sizes
    const dim& dimin = this->dimin;
    const dim& step_rm = this->step_rm;
    const dim& step_cm = this->step_cm;

    // Get a copy to the pointer to the data of motion
    vector2d *motion = mo->get_motion();

    // Get a copt to the pointer to the data of It and gradI
    vector2d *gradI = this->gradI->get_motion();
    float *It = this->It->get_image();

    // Iterate over voxels
    unsigned int idx_rm;
    unsigned int idx_cm;

    float dIdx, dIdy, dIdt;
    vector2d u;
    float prefac;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx_rm = i * step_rm.x + j * step_rm.y;
            idx_cm = i * step_cm.x + j * step_cm.y;

            // Get the prefactor
            dIdx = gradI[idx_cm].x;
            dIdy = gradI[idx_cm].y;
            dIdt = It[idx_cm];

            u = motion[idx_cm];

            prefac = dIdt + u.x * dIdx + u.y * dIdy;

            this->comp_x[idx_rm] = (double) prefac * dIdx;
            this->comp_y[idx_rm] = (double) prefac * dIdy;
        }
    }

    // Done
    return;
}

void OpticalFlowSpectral::compose(Motion* motion) const {
    // Get the dimensions and step sizes
    const dim& dimin = this->dimin;
    const dim& step_rm = this->step_rm;
    const dim& step_cm = this->step_cm;

    // Get a copy to the pointer to the data of motion
    vector2d *datain = motion->get_motion();

    // Iterate over voxels
    unsigned int idx_rm;
    unsigned int idx_cm;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx_rm = i * step_rm.x + j * step_rm.y;
            idx_cm = i * step_cm.x + j * step_cm.y;

            datain[idx_cm] = vector2d(this->comp_x[idx_rm],
                                      this->comp_y[idx_rm]);
        }
    }

    // Done
    return;
}

void OpticalFlowSpectral::multiply_eigenvalues() {
    // Get the step size
    const dim& dimcoefs = this->dimcoefs;
    const dim& step = this->stepcoefs;

    // Iterate over wavenumbers in Fourier space
    unsigned int idx;
    for (unsigned int p = 0; p < dimcoefs.x; p++) {
        for (unsigned int q = 0; q < dimcoefs.y; q++) {
            // Get the linear index
            idx = p * step.x + q * step.y;

            // Get the eigenvalue 
            double eigenval = this->eigenvalues[idx];

            // Multiply coefficients with corresponding eigenvalue matrix
            this->coef_x[idx][0] *= eigenval;
            this->coef_x[idx][1] *= eigenval;
            this->coef_y[idx][0] *= eigenval;
            this->coef_y[idx][1] *= eigenval;
        }
    }

    // Done
    return;
}

void OpticalFlowSpectral::get_update(Motion *motion) {
    // Get the components of the force field
    this->decompose(motion);

    // FFT the individual components
    fftw_execute(this->pf_x);
    fftw_execute(this->pf_y);

    // Multiply with eigenvalues
    multiply_eigenvalues();

    // Inverse FFT of the individual components
    fftw_execute(this->pb_x);
    fftw_execute(this->pb_y);

    // Compose the motion field from components
    this->compose(motion);

    // Done
    return;
}