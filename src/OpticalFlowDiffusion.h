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

    private:        
        // Overload method from base class
        void get_quasi_differential_operator(const Motion* motion);
};

#endif