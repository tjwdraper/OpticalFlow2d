#ifndef _OPTICAL_FLOW_ELASTIC_H_
#define _OPTICAL_FLOW_ELASTIC_H_

#include <src/OpticalFlow.h>
#include <src/Motion.h>

class OpticalFlowElastic : public OpticalFlow {
    public:
        OpticalFlowElastic(const dim dimin, const float alpha);
        ~OpticalFlowElastic();

        // Overload method from base class
        void get_update(Motion* motion);

    private:
        void SOR_iteration(Motion* motion) const;

        Motion* force;

        // Relaxation parameter
        const float omega = 0.66f;
};

#endif