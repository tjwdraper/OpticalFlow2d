#ifndef _OPTICAL_FLOW_DIFFUSION_H_
#define _OPTICAL_FLOW_DIFFUSION_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>
#include <src/OpticalFlow.h>

class OpticalFlowDiffusion : public OpticalFlow {
    public:
        // Constructors and deconstructors
        OpticalFlowDiffusion(const dim dimin, const float alpha);
        ~OpticalFlowDiffusion();

        // Overload method from base class
        void get_update(Motion *motion);

    private:
        // Get the FD approximation of the motion, wwithout the "central" contribution
        void get_quasi_laplacian(const Motion* motion);

        // Do one iteration of the Horn-Schunck method (= Optical Flow Diffusion)
        void horn_schunck_iteration(Motion* motion);

        // Auxiliary field to store the quasi-laplacian, as calculated from the above method
        Motion *qlaplacian;
};

#endif