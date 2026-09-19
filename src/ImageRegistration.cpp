#include "include/ImageRegistration.h"
#include "include/conv2d.hpp"
#include "include/interp2d.hpp"
#include <cstring>


ImageRegistration::ImageRegistration(
    const dim dimin, 
    const ModelOption option,
    const std::size_t nscales, 
    const std::size_t* niter,
    const double alpha,
    const double beta,
    const double eps,
    const std::size_t nrefine,
    const double resampling_factor) {
    // Registration parameters
    _nscales = nscales;
    _nrefine = nrefine;
    _option = option;
    _resampling_factor = resampling_factor;

    // Allocate image and motion 
    _Iref = new opticalflow::Image*[nscales + 1];
    _Imov = new opticalflow::Image*[nscales + 1];
    _motion = new opticalflow::Motion*[nscales + 1];
    _c = new opticalflow::Image*[nscales + 1];
    _solver = new IterativeSolver*[nscales + 1];
    for (int s = static_cast<int>(nscales); s >= 0; s--) {
        double scale = pow(_resampling_factor, s);
        const dim dim_s = dim(
            static_cast<std::size_t> (dimin.x*scale),
            static_cast<std::size_t> (dimin.y*scale)
        );

        _Iref[s] = new opticalflow::Image(dim_s);
        _Imov[s] = new opticalflow::Image(dim_s);
        _motion[s] = new opticalflow::Motion(dim_s);
        _c[s] = new opticalflow::Image(dim_s);
        _solver[s] = new IterativeSolver(dim_s, s, alpha, beta, niter[s], eps);
    }
}

ImageRegistration::~ImageRegistration() {
    for (int s = static_cast<int>(_nscales); s>=0; s--) {
        delete _Iref[s];
        delete _Imov[s];
        delete _motion[s];
        delete _c[s];
        delete _solver[s];
    }
    delete[] _Iref;
    delete[] _Imov;
    delete[] _motion;
    delete[] _c;
    delete[] _solver;
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

