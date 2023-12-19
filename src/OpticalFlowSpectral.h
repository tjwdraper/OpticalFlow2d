#ifndef _OPTICAL_FLOW_SPECTRAL_H_
#define _OPTICAL_FLOW_SPECTRAL_H_

#include <fftw3.h>

#include <src/ImageRegistrationSolver.h>

class OpticalFlowSpectral : public ImageRegistrationSolver {
    public:
        // Constructors and deconstructors
        OpticalFlowSpectral(const dim dimin, const float alpha);
        ~OpticalFlowSpectral();

        // Overload function from base class
        void get_update(Motion *motion);

    private:
        void decompose(const Motion* motion);
        void compose(Motion* motion) const;

        void multiply_eigenvalues();

        void set_eigenvalues();

        // Image dimensions
        dim step_rm;
        dim step_cm;

        // FFT dimensions
        dim dimcoefs;
        unsigned int sizecoefs;
        dim stepcoefs;

        // Auxiliary fields for FFT
        double *comp_x;
        double *comp_y;

        // Eigenvalue matrix of the Laplacian differential operator
        double *eigenvalues;

        // FFT coefficients
        fftw_complex *coef_x;
        fftw_complex *coef_y;

        // FFTW plan for forward and inverse FFT
        fftw_plan pf_x;
        fftw_plan pf_y;
        fftw_plan pb_x;
        fftw_plan pb_y;
};

#endif