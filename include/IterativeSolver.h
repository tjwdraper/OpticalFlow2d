#ifndef _ITERATIVE_SOLVER_H_
#define _ITERATIVE_SOLVER_H_

#include "coord2d.hpp"
#include "Field.hpp"

class IterativeSolver {
    public:
        // Constructors and deconstructors
        IterativeSolver(const dim dimin, 
                        const std::size_t resolution_level, 
                        const double alpha, 
                        const double beta, 
                        const std::size_t niter, 
                        const double eps);
        ~IterativeSolver();

        // Getters and setters
        double get_alpha() const;
        double get_beta() const;

        // Estimate motion from Horn-Schunck model
        void estimate_optical_flow(opticalflow::Motion& motion, 
                                   const opticalflow::Image& Iref, 
                                   const opticalflow::Image& Imov);
        void estimate_optical_flow(opticalflow::Motion& motion, 
                                   opticalflow::Image& c, 
                                   const opticalflow::Image& Iref, 
                                   const opticalflow::Image& Imov);

    private:        
        dim _dimin;
        dim _step;
        std::size_t _sizein;

        opticalflow::Motion* _horn_schunck_average;
        opticalflow::Image* _c_average;

        opticalflow::Motion *_spatial_gradient_image;
        opticalflow::Image *_temporal_derivative_image;

        // Model parameters
        double _alpha;
        double _beta;
        double _eps;
        std::size_t _niter;
        std::size_t _resolution_level;
};

#endif