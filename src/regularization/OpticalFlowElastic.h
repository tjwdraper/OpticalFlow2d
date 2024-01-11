#ifndef _OPTICAL_FLOW_ELASTIC_H_
#define _OPTICAL_FLOW_ELASTIC_H_

#include <src/regularization/OpticalFlow.h>
#include <src/Motion.h>

class OpticalFlowElastic : public OpticalFlow {
    public:
        OpticalFlowElastic(const dim dimin, const float mu, const float lambda, const float omega = 0.66f);
        ~OpticalFlowElastic();

        // Overload method from base class
        void get_update(Motion* motion, const Image* Iref = NULL, const Image* Imov = NULL);

    private:
        void SOR_iteration(Motion* motion) const;

        // Regularisation and relaxation parameters
        float mu;
        float lambda;
        float omega;
};

#endif