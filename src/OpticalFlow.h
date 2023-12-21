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
        virtual void get_update(Motion *motion) {};

    protected:
        dim dimin;
        unsigned int sizein;
        dim step;

        // Spatial and temporal image gradients
        Motion *gradI;
        Image *It;

        float alpha;
};

#endif