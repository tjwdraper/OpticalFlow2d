#ifndef _OPTICAL_FLOW_SOLVER_H_
#define _OPTICAL_FLOW_SOLVER_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class OpticalFlowSolver {
    public:
        // Constructors and deconstructors
        OpticalFlowSolver(const dim dimin, const float alpha);
        ~OpticalFlowSolver();

        // Get the image gradients
        void get_image_gradients(const Image* Iref, const Image* Imov);

        // Do one iteration
        void get_update(Motion *motion);

    private:
        void get_quasi_laplacian(const Motion* motion);

        void horn_schunck_iteration(Motion* motion);

        dim dimin;
        unsigned int sizein;
        dim step;

        // Spatial and temporal image gradients
        Motion *gradI;
        Image *It;

        float alpha;
        Motion *qlaplacian;
};

#endif