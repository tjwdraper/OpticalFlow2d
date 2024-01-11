#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <src/regularization/IterativeSolver.h>
#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlow : public IterativeSolver {
    public:
        // Constructors and deconstructors
        OpticalFlow(const dim dimin);
        ~OpticalFlow();

        // Get the image gradients
        void get_image_gradients(const Image* Iref, const Image* Imov);

        // Construct the force from the image gradients and motion estimate
        void get_force(Motion* force, const Motion* motion) const;

        // Do one iteration
        virtual void get_update(Motion *motion) {};

    protected:
        // Spatial and temporal image gradients
        Motion *gradI;
        Image *It;
};

#endif