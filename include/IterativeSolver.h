#ifndef _ITERATIVE_SOLVER_H_
#define _ITERATIVE_SOLVER_H_

#include "include/coord2d.hpp"
#include "include/Field.hpp"

class IterativeSolver {
    public:
        // Constructors and deconstructors
        IterativeSolver(const dim dimin, const double alpha, const std::size_t niter, const double eps);
        ~IterativeSolver();

        // Getters and setters
        double get_alpha() const;

        // Estimate motion from Horn-Schunck model
        void estimate_optical_flow(opticalflow::Motion& motion, const opticalflow::Image& Iref, const opticalflow::Image& Imov);

    private:

        // Calculate image gradients
        // void spatial_derivative(Motion* grad_image, const Image *image) const;
        // void temporal_derivative(Image* It, const Image *Iref, const Image* Imov) const;
        // void set_derivatives(const Image* Iref, const Image* Imov) const;
        // void get_force(Motion* force, const Motion* motion) const;
        
        dim _dimin;
        dim _step;
        std::size_t _sizein;

        opticalflow::Motion* _horn_schunck_average;

        opticalflow::Motion *_spatial_gradient_image;
        opticalflow::Image *_temporal_derivative_image;
        // opticalflow::Motion* force;
        
        // // Do one iteration of the Horn-Schunck method (= Optical Flow Diffusion)
        // void optical_flow_iteration(opticalflow::Motion* motion);

        // // Get the FD approximation of the motion, wwithout the "central" contribution
        // void get_quasi_differential_operator(const opticalflow::Motion* motion);

        // // Quasi differential operator
        // opticalflow::Motion *qdiffoperator;

        // Regularisation parameters
        double _alpha;
        double _eps;
        std::size_t _niter;
};

#endif