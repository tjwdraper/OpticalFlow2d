#ifndef _OPTICAL_FLOW_CURVATURE_H_
#define _OPTICAL_FLOW_CURVATURE_H_

#include <src/OpticalFlow.h>
#include <src/Motion.h>

class OpticalFlowCurvature : public OpticalFlow {
    public:
        OpticalFlowCurvature(const dim dimin, const float alpha);
        ~OpticalFlowCurvature();

    private:
        // Overload method from base class
        void get_quasi_differential_operator(const Motion* motion);
};

#endif