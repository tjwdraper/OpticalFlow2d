#ifndef _IMAGE_REGISTRATION_SOLVER_H_
#define _IMAGE_REGISTRATION_SOLVER_H_

#include <src/coord2d.h>
#include <src/Motion.h>
#include <src/Image.h>

class ImageRegistrationSolver {
    public:
        // Constructors and deconstructors
        ImageRegistrationSolver() {};
        ImageRegistrationSolver(const dim dimin, const float alpha);
        ~ImageRegistrationSolver();

        // Get the image gradients
        void set_image_gradients(const Image* Iref, const Image* Imov);

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