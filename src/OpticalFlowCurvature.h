#ifndef _OPTICAL_FLOW_CURVATURE_H_
#define _OPTICAL_FLOW_CURVATURE_H_

#include <src/OpticalFlow.h>
#include <src/Motion.h>
#include <fftw3.h>

class OpticalFlowCurvature : public OpticalFlow {
    public:
        OpticalFlowCurvature(const dim dimin, const float alpha);
        ~OpticalFlowCurvature();

        // Overload method from base class
        void get_update(Motion *motion);

    private:
        void construct_rhs(const Motion *motion);

        void construct_motion(Motion *motion) const;

        void multiply_eigenvalues();

        void set_eigenvalues();

        // Image dimensions in row- and column-major ordering
        dim step_rm;
        dim step_cm;

        // FFT plan for the forward and backward DCT
        fftw_plan pf_x, pf_y;
        fftw_plan pb_x, pb_y;

        // Auxiliary fields
        Motion* force;

        double *rhs_x, *rhs_y;
        double *coef_x, *coef_y;

        // Eigenvalue matrix of the biharmonic operator
        double *eigenvalues;

        // Step size in time-marching algorithm
        const float tau = 100;
};

#endif