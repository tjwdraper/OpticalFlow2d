#ifndef _IMAGE_REGISTRATION_H_
#define _IMAGE_REGISTRATION_H_

#include "include/coord2d.hpp"
#include "include/Field.hpp"
#include "include/ConfigurationOptions.hpp"

#include "include/IterativeSolver.h"

class ImageRegistration {
    public:
        // Constructors and deconstructors
        ImageRegistration(const dim dimin, 
                          const ModelOption option,
                          const std::size_t nscales, 
                          const std::size_t* niter,
                          const double alpha,
                          const double beta,
                          const double eps,
                          const std::size_t nrefine,
                          const double resampling_factor);
        ~ImageRegistration();

        // Getters and setters
        void set_reference_image(const opticalflow::Image& im);
        void set_moving_image(const opticalflow::Image& im);
        const opticalflow::Motion& get_estimated_motion() const;
        const opticalflow::Image& get_estimated_c() const;

        // Estimate motion
        void estimate_optical_flow();

    private:
        // void display_registration_parameters() const;
        std::size_t _nscales;
        std::size_t _nrefine;
        ModelOption _option;
        double _resampling_factor;

        IterativeSolver** _solver;

        opticalflow::Image** _Iref;
        opticalflow::Image** _Imov;

        opticalflow::Motion** _motion;
        opticalflow::Image** _c;
};

#endif