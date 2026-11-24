#include "ImageRegistration.h"
#include "IterativeSolver.hpp"
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

        if (config.model_option == ModelOption::HORN_SCHUNCK) {    
            _solver[s] = std::make_unique<IterativeSolver>(
                std::make_unique<HornSchunckStrategy>(solver_params)
            );
        }
        else if (config.model_option == ModelOption::CORNELIUS_KANADE) {
            _solver[s] = std::make_unique<IterativeSolver>(
                std::make_unique<CorneliusKanadeStrategy>(solver_params)
            );

        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public methods
///////////////////////////////////////////////////////////////////////////////////////////////////

// Deconstructor
ImageRegistration::~ImageRegistration() {
    _instance.reset();

    for (int s = static_cast<int>(_nscales); s>=0; s--) {
        _Iref[s].reset();
        _Imov[s].reset();
        _motion[s].reset();
        // _c[s].reset();
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
    const opticalflow::gaussianConv2d filter(0.8);
    filter.convolute(*_Iref[0]);

    // Anti-aliasing filter for downsampling
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (_resampling_factor * _resampling_factor) - 1.0);
    const opticalflow::gaussianConv2d filter_aa(sigma_aa);

    // Build Gaussian pyramid
    for (std::size_t s = 1; s <= _nscales; ++s) {
        opticalflow::Image I_aa(*_Iref[s - 1]);

        filter_aa.convolute(I_aa);
        opticalflow::interp2d::resize(*_Iref[s], I_aa);
    }
}

void ImageRegistration::set_moving_image(const opticalflow::Image& image) {
    // Set finest pyramid level
    *_Imov[0] = image;

    // Smooth finest level
    const opticalflow::gaussianConv2d filter(0.8);
    filter.convolute(*_Imov[0]);

    // Anti-aliasing filter for downsampling
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (_resampling_factor * _resampling_factor) - 1.0);
    const opticalflow::gaussianConv2d filter_aa(sigma_aa);

    // Build Gaussian pyramid
    for (std::size_t s = 1; s <= _nscales; ++s) {
        opticalflow::Image I_aa(*_Imov[s - 1]);

        filter_aa.convolute(I_aa);
        opticalflow::interp2d::resize(*_Imov[s], I_aa);
    }
}

const opticalflow::Motion& ImageRegistration::get_estimated_motion() const {
    return *_motion[0];
}

opticalflow::Image ImageRegistration::get_residue() const {
    // Dereference images and motion field
    const opticalflow::Image& Iref = *_Iref[0];
    const opticalflow::Image& Imov = *_Imov[0];
    const opticalflow::Motion& motion = *_motion[0];

    // Get the image dimensions
    const dim dimin = Iref.get_dimensions();

    // Compute image gradient of moving image
    opticalflow::Motion image_gradient(dimin);
    opticalflow::gradients::gradient(image_gradient, Imov);

    // Allocate memory for output
    opticalflow::Image c(dimin);

    // Compute the residue c
    for (std::size_t idx = 0; idx < Iref.get_size(); ++idx) {
        // Select deformation vector
        const vector2d u = motion.at(idx);
        const vector2d dI = image_gradient.at(idx);

        // Compute residue at voxel
        c.set(Iref.at(idx) - Imov.at(idx) - dot(u, dI), idx);
    }

    return c;
}

// Estimate motion
void ImageRegistration::estimate_optical_flow() {    

    // Multiresolution pyramid
    for (int s = static_cast<int>(_nscales); s >= 0; s--) {

        // Dereference variables at current level
        opticalflow::Motion& motion = *_motion[s];
        const opticalflow::Image& Iref = *_Iref[s];
        const opticalflow::Image& Imov = *_Imov[s];
        IterativeSolver& solver = *_solver[s];

        // Get current image dimensions
        const dim dimin = solver.parameters().dimin;

        // Upsample from previous resolution
        if (s < _nscales)
            opticalflow::interp2d::resize(motion, *_motion[s+1]); // Resample estimated DVF from previous resolution level

        // Estimate motion at current resolution level
        solver.estimate_optical_flow(motion, Iref, Imov);

        // If no refine (i.e. _nrefine == 0), continue to next iteration
        if (_nrefine == 0) 
            continue;

        // Refinement
        opticalflow::Image ImovPropagated(dimin);
        opticalflow::Motion motionPropagated(dimin);
        for (std::size_t r = 0; r < _nrefine; ++r) {
            // Initialize auxiliary variables
            motionPropagated.fill(0.0);

            // Warp the moving image with the so-far estimated motion field (which warps Imov -> Imov_aux \approx Iref)...
            opticalflow::interp2d::warp2d(ImovPropagated, Imov, motion);

            // ...use deformed image to estimate an deformation field from Imov_aux -> Iref...
            solver.estimate_optical_flow(motionPropagated, Iref, ImovPropagated);

            // ...accumulate (= motion field composition) to get the estimated deformation field from Imov to Iref
            opticalflow::interp2d::accumulate(motion, motionPropagated);
        }
    }

    // Done
    return;
}

