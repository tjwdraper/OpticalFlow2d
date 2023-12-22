#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlow {
    public:
        // Constructors and deconstructors
        OpticalFlow(const dim dimin, const float alpha);
        ~OpticalFlow();

        // Get the image gradients
        void get_image_gradients(const Image* Iref, const Image* Imov);

        // Do one iteration
        void get_update(Motion *motion);

    protected:
        // Do one iteration of the Horn-Schunck method (= Optical Flow Diffusion)
        void optical_flow_iteration(Motion* motion);

        // Get the FD approximation of the motion, wwithout the "central" contribution
        virtual void get_quasi_differential_operator(const Motion* motion) {};

        dim dimin;
        unsigned int sizein;
        dim step;

        // Spatial and temporal image gradients
        Motion *gradI;
        Image *It;

        // Quasi differential operator
        Motion *qdiffoperator;

        float alpha;
};

#endif