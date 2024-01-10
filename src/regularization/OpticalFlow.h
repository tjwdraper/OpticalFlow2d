#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlow {
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

        virtual void get_update(Motion *motion, const Image* Iref, const Image* Imov) {};

    protected:
        dim dimin;
        unsigned int sizein;
        dim step;

        // Spatial and temporal image gradients
        Motion *gradI;
        Image *It;
};

#endif