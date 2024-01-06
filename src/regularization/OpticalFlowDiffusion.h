#ifndef _OPTICAL_FLOW_DIFFUSION_H_
#define _OPTICAL_FLOW_DIFFUSION_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>
#include <src/regularization/OpticalFlow.h>

class OpticalFlowDiffusion : public OpticalFlow {
    public:
        // Constructors and deconstructors
        OpticalFlowDiffusion(const dim dimin, const float alpha);
        ~OpticalFlowDiffusion();

        // Overload method from base class
        void get_update(Motion *motion);

    private:    
        // Do one iteration of the Horn-Schunck method (= Optical Flow Diffusion)
        void optical_flow_iteration(Motion* motion);

        // Get the FD approximation of the motion, wwithout the "central" contribution
        void get_quasi_differential_operator(const Motion* motion);

        // Quasi differential operator
        Motion *qdiffoperator;

        // Regularisation parameter
        float alpha;
};

#endif