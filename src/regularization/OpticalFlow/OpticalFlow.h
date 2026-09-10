#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <src/regularization/IterativeSolver.h>
#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlow : public IterativeSolver {
    public:
        // Constructors and deconstructors
        OpticalFlow(const dim dimin, const double alpha);
        ~OpticalFlow();

        // Construct the force from the image gradients and motion estimate
        void get_force(Motion* force, const Motion* motion) const;

        // Do one iteration
        void get_update(Motion *motion, const Image* Iref = NULL, const Image* Imov = NULL) {};

    protected:
        // Spatial and temporal image gradients
        Motion *force;  
        
        // Do one iteration of the Horn-Schunck method (= Optical Flow Diffusion)
        void optical_flow_iteration(Motion* motion);

        // Get the FD approximation of the motion, wwithout the "central" contribution
        void get_quasi_differential_operator(const Motion* motion);

        // Quasi differential operator
        Motion *qdiffoperator;

        // Regularisation parameters
        double alpha;
};

#endif