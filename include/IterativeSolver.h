#ifndef _ITERATIVE_SOLVER_H_
#define _ITERATIVE_SOLVER_H_

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>

class IterativeSolver {
    public:
        // Constructors and deconstructors
        IterativeSolver(const dim dimin, const double alpha);
        ~IterativeSolver();

        // Calculate image gradients
        void spatial_derivative(Motion* grad_image, const Image *image) const;
        void temporal_derivative(Image* It, const Image *Iref, const Image* Imov) const;
        void set_derivatives(const Image* Iref, const Image* Imov) const;
        void get_force(Motion* force, const Motion* motion) const;
        
        // Do one update in iterative scheme
        void get_update(Motion *motion, const Image* Iref = NULL, const Image* Imov = NULL);

        // One iteration of the iterative scheme
        void get_update(Motion *motion, const Image* Iref = NULL, const Image* Imov = NULL);

    private:
        dim dimin;
        dim step;
        unsigned int sizein;

        Motion *gradI;
        Image *It;
        Motion* force;
        
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