#include "ImageRegistration.h"
#include "conv2d.hpp"
#include "interp2d.hpp"

#include <cstring>

///////////////////////////////////////////////////////////////////////////////////////////////////
// Private methods
///////////////////////////////////////////////////////////////////////////////////////////////////

// Set static member variable
std::unique_ptr<ImageRegistration> ImageRegistration::_instance = nullptr;

// Constructor
ImageRegistration::ImageRegistration(const RegistrationConfiguration& config) {
    // Registration parameters
    _nscales = config.nscales;
    _nrefine = config.nrefine;
    _option = config.model_option;
    _resampling_factor = config.resampling_factor;

    // Allocate image and motion 
    _Iref.resize(_nscales + 1);
    _Imov.resize(_nscales + 1);
    _motion.resize(_nscales + 1);
    _c.resize(_nscales + 1);
    _solver.resize(_nscales + 1);

    for (int s = static_cast<int>(config.nscales); s >= 0; s--) {
        // Get the solver parameters for the current scale
        const SolverParameters& solver_params = config.solver_parameters[s];

        // Get image dimensions at current resolution level
        const dim dimin = solver_params.dimin;

        // Allocate memory for images and motion fields at current resolution level
        _Iref[s] = std::make_unique<opticalflow::Image>(solver_params.dimin);
        _Imov[s] = std::make_unique<opticalflow::Image>(solver_params.dimin);
        _motion[s] = std::make_unique<opticalflow::Motion>(solver_params.dimin);
        _c[s] = std::make_unique<opticalflow::Image>(solver_params.dimin);
        _solver[s] = std::make_unique<IterativeSolver>(solver_params);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Private methods
///////////////////////////////////////////////////////////////////////////////////////////////////

// Deconstructor
ImageRegistration::~ImageRegistration() {
    _instance.reset();

    for (int s = static_cast<int>(_nscales); s>=0; s--) {
        _Iref[s].reset();
        _Imov[s].reset();
        _motion[s].reset();
        _c[s].reset();
        _solver[s].reset();
    }
}

// Instance management
void ImageRegistration::createInstance(const RegistrationConfiguration& config) {
    // Check if an instance already exists
    if (_instance)
        throw std::logic_error("Error: ImageRegistration is already initialized.");

    // Create a new instance of ImageRegistration. Note: we cannot use std::make_unique 
    // here because the constructor is private.
    _instance.reset(new ImageRegistration(config));
}

ImageRegistration& ImageRegistration::getInstance() {
    if (!_instance)
        throw std::logic_error("Error: ImageRegistration is not initialized.");
    return *_instance;
}

void ImageRegistration::shutdownInstance() {
    if (!_instance)
        throw std::logic_error("Error: ImageRegistration is not initialized.");
    _instance.reset();
}

bool ImageRegistration::isInitialized() {
    return _instance != nullptr;
}

// Getters and setters
void ImageRegistration::set_reference_image(const opticalflow::Image& image) {
    // Set finest pyramid level
    *_Iref[0] = image;

    // Smooth finest level
    gaussian_conv2d filter(0.8);
    filter.convolute(*_Iref[0]);

    // Anti-aliasing filter for downsampling
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (_resampling_factor * _resampling_factor) - 1.0);
    gaussian_conv2d filter_aa(sigma_aa);

    // Build Gaussian pyramid
    for (std::size_t s = 1; s <= _nscales; ++s) {
        opticalflow::Image I_aa(*_Iref[s - 1]);

        filter_aa.convolute(I_aa);
        interp2d::resize(*_Iref[s], I_aa);
    }
}

void ImageRegistration::set_moving_image(const opticalflow::Image& image) {
    // Set finest pyramid level
    *_Imov[0] = image;

    // Smooth finest level
    gaussian_conv2d filter(0.8);
    filter.convolute(*_Imov[0]);

    // Anti-aliasing filter for downsampling
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (_resampling_factor * _resampling_factor) - 1.0);
    gaussian_conv2d filter_aa(sigma_aa);

    // Build Gaussian pyramid
    for (std::size_t s = 1; s <= _nscales; ++s) {
        opticalflow::Image I_aa(*_Imov[s - 1]);

        filter_aa.convolute(I_aa);
        interp2d::resize(*_Imov[s], I_aa);
    }
}

const opticalflow::Motion& ImageRegistration::get_estimated_motion() const {
    return *_motion[0];
}

const opticalflow::Image& ImageRegistration::get_estimated_c() const {
    return *_c[0];
}

// Estimate motion
void ImageRegistration::estimate_optical_flow() {    
    // Multiresolution pyramid
    for (int s = static_cast<int>(_nscales); s >= 0; s--) {
        // Dereference variables at current level
        opticalflow::Motion& motion_s = *_motion[s];
        opticalflow::Image& c_s = *_c[s];
        const opticalflow::Image& Iref_s = *_Iref[s];
        const opticalflow::Image& Imov_s = *_Imov[s];
        IterativeSolver& solver_s = *_solver[s];

        // Upsample from previous resolution
        if (s < _nscales) {
            interp2d::resize(motion_s, *_motion[s+1]); // Resample estimated DVF from previous resolution level
            if (_option == ModelOption::CORNELIUS_KANADE)
                interp2d::resize(c_s, *_c[s+1]);
        }
        else if (_option == ModelOption::CORNELIUS_KANADE)
            c_s.fill(0.0);

        // Estimate motion at current resolution level
        if (_option == ModelOption::HORN_SCHUNCK)
            solver_s.estimate_optical_flow(motion_s, Iref_s, Imov_s);
        else if (_option == ModelOption::CORNELIUS_KANADE)
            solver_s.estimate_optical_flow(motion_s, c_s, Iref_s, Imov_s);

        // Refinement
        for (std::size_t r = 0; r < _nrefine; ++r) {
            // Initialize auxiliary variables
            opticalflow::Image Imov_aux_s(Imov_s.get_dimensions());
            opticalflow::Motion motion_aux_s(Imov_s.get_dimensions());
            motion_aux_s.fill(vector2d(0.0));

            // Warp the moving image with the so-far estimated motion field (which warps Imov -> Imov_aux \approx Iref)...
            interp2d::warp2d(Imov_aux_s, Imov_s, motion_s);

            // ...use deformed image to estimate an deformation field from Imov_aux -> Iref...
            if (_option == ModelOption::HORN_SCHUNCK)
                solver_s.estimate_optical_flow(motion_aux_s, Iref_s, Imov_aux_s);
            else if (_option == ModelOption::CORNELIUS_KANADE)
                solver_s.estimate_optical_flow(motion_aux_s, c_s, Iref_s, Imov_aux_s);

            // ...accumulate (= motion field composition) to get the estimated deformation field from Imov to Iref
            interp2d::accumulate(motion_s, motion_aux_s);
        }
    }

    // Done
    return;
}

