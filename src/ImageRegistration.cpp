#include "include/ImageRegistration.h"
#include "include/conv2d.hpp"
#include "include/interp2d.hpp"
// #include "include/mxParser.hpp"

// #include <mex.h>
#include <cstring>
// #include <include/Logger.h>

// void ImageRegistration::display_registration_parameters() const {
//     mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n");
//     mexPrintf("Optical flow image registration started... (2D C++ implementation)...\n");
//     mexPrintf("Registration parameters:\n");

//     // Image dimensions and multiresolution parameters
//     mexPrintf("dimensions:\t\t\t\t(%d %d)\n", _dimin[0].x, _dimin[0].y);
//     mexPrintf("niter:\t\t\t\t\t(%d", _niter[0]);
//     for (int s = 1; s < _nscales+1; s++) {
//         mexPrintf(" %d", _niter[s]);
//     }
//     mexPrintf(")\n");
//     mexPrintf("nscales:\t\t\t\t%d\n", _nscales);
//     mexPrintf("nrefine:\t\t\t\t%d\n", _nrefine);
//     mexPrintf("alpha:\t\t\t\t%.3f\n", _solver[0]->get_alpha());

//     mexPrintf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n");

//     // Done
//     return;
// }

// void ImageRegistration::estimate_motion_at_current_resolution(
//     Motion& motion, 
//     const Image& Iref, 
//     const Image& Imov,
//     IterativeSolver& solver) {
    
//     // Create auxiliary motion field
//     Image *Iaux = new Image(dimin);

//     // Create auxiliary motion field
//     Motion *motion_est = new Motion(dimin);

//     for (int refine = 0; refine < _nrefine; refine++) {
//         // Reset Iaux to input image
//         *Iaux = *Imov;

//         // Warp moving image with accumulated motion field
//         Iaux->warp2d(*motion);

//         // Create a Logger object
//         Logger log(dimin, niter, _verbose);

//         // Calculating the image gradients only has to be done once
//         solver->set_derivatives(Iref, Iaux);

//         // Iterate over resolution levels
//         for (int iter = 0; iter < niter; iter++) {
//             // Calculate the update step
//             solver->get_update(motion_est);

//             // Calculate the difference between iterations
//             log.update_error(motion_est);

//             // Converge check
//             if ((log.get_error_at_current_iteration() < 0.001f) &&
//                 (iter > 1)) {
//                 break;
//             }
//         }

//         // Accumulate motion field
//         motion->accumulate(*motion_est);

//         // Reset auxiliary field
//         motion_est->reset();

//     }

//     // Free up the mem
//     delete motion_est;
//     delete Iaux;
    
//     // Done
//     return;
// }

ImageRegistration::ImageRegistration(
    const dim dimin, 
    const ModelOption option,
    const std::size_t nscales, 
    const std::size_t* niter,
    const double alpha,
    const double beta,
    const double eps,
    const std::size_t nrefine) {
    // Registration parameters
    _nscales = nscales;
    _nrefine = nrefine;

    // Allocate image and motion 
    _Iref = new opticalflow::Image*[nscales + 1];
    _Imov = new opticalflow::Image*[nscales + 1];
    _motion = new opticalflow::Motion*[nscales + 1];
    _c = new opticalflow::Image*[nscales + 1];
    _solver = new IterativeSolver*[nscales + 1];
    for (int s = static_cast<int>(nscales); s >= 0; s--) {
        double scale = pow(2.0, s);
        const dim dim_s = dim(
            static_cast<std::size_t> (dimin.x/scale),
            static_cast<std::size_t> (dimin.y/scale)
        );

        _Iref[s] = new opticalflow::Image(dim_s);
        _Imov[s] = new opticalflow::Image(dim_s);
        _motion[s] = new opticalflow::Motion(dim_s);
        _c[s] = new opticalflow::Image(dim_s);
        _solver[s] = new IterativeSolver(dim_s, s, alpha, beta, niter[s], eps);
    }

    // Display registration settings
    // ImageRegistration::display_registration_parameters();

}

// ImageRegistration::ImageRegistration(const mxArray *config) {
//     // Registration parameters
//     mxParser::parse_nscales(_nscales, config);
//     mxParser::parse_nrefine(_nrefine, config);
    
//     dim dimin;
//     mxParser::parse_size_image(dimin, config);

//     double alpha;
//     mxParser::parse_alpha(alpha, config);

//     std::size_t* niter = new size_t[_nscales+1];
//     mxParser::parse_niter(niter, config);

//     double eps;
//     mxParser::parse_convergence_threshold(eps, config);

//     // Allocate images, motion field and solver
//     _Iref = new opticalflow::Image*[_nscales+1];
//     _Imov = new opticalflow::Image*[_nscales+1];
//     _motion = new opticalflow::Motion*[_nscales+1];
//     _solver = new IterativeSolver*[_nscales+1];
//     for (int s = static_cast<int>(_nscales); s >= 0; s--) {
//         double scale = pow(2.0, s);
//         const dim dim_s = dim(
//             static_cast<std::size_t> (dimin.x/scale),
//             static_cast<std::size_t> (dimin.y/scale)
//         );

//         _Iref[s] = new opticalflow::Image(dim_s);
//         _Imov[s] = new opticalflow::Image(dim_s);
//         _motion[s] = new opticalflow::Motion(dim_s);
//         _solver[s] = new IterativeSolver(dim_s, s, alpha, niter[s], eps);
//     }

//     // Free memory
//     delete[] niter;
// }

ImageRegistration::~ImageRegistration() {
    for (int s = static_cast<int>(_nscales); s>=0; s--) {
        delete _Iref[s];
        delete _Imov[s];
        delete _motion[s];
        delete _c[s];
        delete _solver[s];
    }
    delete[] _solver;
    delete[] _Iref;
    delete[] _Imov;
    delete[] _motion;
    delete[] _c;
}

// Getters and setters
void ImageRegistration::set_reference_image(const opticalflow::Image& image) {
    // Set finest pyramid level
    *_Iref[0] = image;

    // Smooth finest level
    gaussian_conv2d filter(0.8);
    filter.convolute(*_Iref[0]);

    // Anti-aliasing filter for downsampling
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (0.5 * 0.5) - 1.0);
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
    const double sigma_aa = 0.6 * std::sqrt(1.0 / (0.5 * 0.5) - 1.0);
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
        if (s < _nscales)
            interp2d::resize(motion_s, *_motion[s+1]); // Resample estimated DVF from previous resolution level
            if (_option == ModelOption::CORNELIUS_KANADE)
                interp2d::resize(c_s, *_c[s+1]);

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

