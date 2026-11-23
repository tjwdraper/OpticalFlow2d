#ifndef _IMAGE_REGISTRATION_H_
#define _IMAGE_REGISTRATION_H_

#include "coord2d.hpp"
#include "Field.hpp"
#include "RegistrationConfiguration.hpp"
#include "IterativeSolver.h"

#include <memory>
#include <vector>

class ImageRegistration {
    private:
        static std::unique_ptr<ImageRegistration> _instance;

        // Remove default constructor, create constructor for registration parameters
        ImageRegistration() = delete;
        ImageRegistration(const RegistrationConfiguration& config);

        // Private members
        std::size_t _nscales;
        std::size_t _nrefine;
        ModelOption _option;
        double _resampling_factor;

        std::vector<std::unique_ptr<IterativeSolver>> _solver;

        std::vector<std::unique_ptr<opticalflow::Image>> _Iref;
        std::vector<std::unique_ptr<opticalflow::Image>> _Imov;
        std::vector<std::unique_ptr<opticalflow::Motion>> _motion;
        std::vector<std::unique_ptr<opticalflow::Image>> _c;

    public:
        // Instance management
        static void createInstance(const RegistrationConfiguration& config);
        static ImageRegistration& getInstance(); 
        static void shutdownInstance();

        // Deconstructor
        ~ImageRegistration();

        // Check if ImageRegistration has been initialized
        static bool isInitialized(); 

        // Getters and setters
        void set_reference_image(const opticalflow::Image& im);
        void set_moving_image(const opticalflow::Image& im);
        const opticalflow::Motion& get_estimated_motion() const;
        const opticalflow::Image& get_estimated_c() const;

        // Estimate motion
        void estimate_optical_flow();
};

#endif