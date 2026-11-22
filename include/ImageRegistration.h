#ifndef _IMAGE_REGISTRATION_H_
#define _IMAGE_REGISTRATION_H_

#include "coord2d.hpp"
#include "Field.hpp"
#include "ConfigurationOptions.hpp"

#include "IterativeSolver.h"

#include <memory>

class ImageRegistration {
    private:
        static std::unique_ptr<ImageRegistration> _instance;

        // Remove default constructor, create constructor for registration parameters
        ImageRegistration() = delete;
        ImageRegistration(
            const dim dimin,
            const ModelOption option,
            const std::size_t nscales,
            const std::size_t* niter,
            const double alpha,
            const double beta,
            const double eps,
            const std::size_t nrefine,
            const double resampling_factor
        );

        // Private members
        std::size_t _nscales;
        std::size_t _nrefine;
        ModelOption _option;
        double _resampling_factor;

        IterativeSolver** _solver;

        opticalflow::Image** _Iref;
        opticalflow::Image** _Imov;

        opticalflow::Motion** _motion;
        opticalflow::Image** _c;

    public:
        // Create an instance
        static void createInstance(const dim dimin,
                                   const ModelOption option,
                                   const std::size_t nscales,
                                   const std::size_t* niter,
                                   const double alpha,
                                   const double beta,
                                   const double eps,
                                   const std::size_t nrefine,
                                   const double resampling_factor) {
            if (_instance)
                throw std::logic_error("Error: ImageRegistration is already initialized.");

            // Create a new instance of ImageRegistration. Note: we cannot use std::make_unique 
            // here because the constructor is private.
            _instance.reset(
                new ImageRegistration(
                    dimin,
                    option,
                    nscales,
                    niter,
                    alpha,
                    beta,
                    eps,
                    nrefine,
                    resampling_factor
                )
            );
        }

        static ImageRegistration& getInstance() {
            if (!_instance)
                throw std::logic_error("Error: ImageRegistration is not initialized.");
            return *_instance;
        }

        // Deconstructor
        ~ImageRegistration();

        // Check if ImageRegistration has been initialized
        static bool isInitialized() {
            return _instance != nullptr;
        }

        // Getters and setters
        void set_reference_image(const opticalflow::Image& im);
        void set_moving_image(const opticalflow::Image& im);
        const opticalflow::Motion& get_estimated_motion() const;
        const opticalflow::Image& get_estimated_c() const;

        // Estimate motion
        void estimate_optical_flow();
};

#endif