#ifndef _ITERATIVE_SOLVER_H_
#define _ITERATIVE_SOLVER_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class IterativeSolver {
    public:
        // Constructors and deconstructors
        IterativeSolver(const dim dimin);
        ~IterativeSolver();

        // Calculate image gradients
        void spatial_derivative(Motion* grad_image, const Image *image) const;
        void temporal_derivative(Image* It, const Image *Iref, const Image* Imov) const;

    protected:
        dim dimin;
        dim step;
        unsigned int sizein;

};

#endif