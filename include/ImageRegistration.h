#ifndef _IMAGE_REGISTRATION_H_
#define _IMAGE_REGISTRATION_H_

// #include <mex.h>

#include "include/coord2d.hpp"
#include "include/Field.hpp"
// #include "include/SolverOptions.h"

#include "include/IterativeSolver.h"

class ImageRegistration {
    public:
        // Constructors and deconstructors
        ImageRegistration(const dim dimin, 
                          const std::size_t nscales, 
                          const std::size_t* niter,
                          const double alpha,
                          const double eps,
                          const std::size_t nrefine);
        // ImageRegistration(const mxArray* config);
        ~ImageRegistration();

        // Getters and setters
        void set_reference_image(const opticalflow::Image& im);
        void set_moving_image(const opticalflow::Image& im);
        const opticalflow::Motion& get_estimated_motion() const;

        // Estimate motion
        void estimate_optical_flow();

    private:
        // void display_registration_parameters() const;
        std::size_t _nscales;
        std::size_t _nrefine;
        IterativeSolver** _solver;

        opticalflow::Image** _Iref;
        opticalflow::Image** _Imov;

        opticalflow::Motion** _motion;
};

#endif