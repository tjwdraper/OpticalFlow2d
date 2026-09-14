#include "include/IterativeSolver.h"
#include "include/gradients.hpp"

// Constructors and deconstructors
IterativeSolver::IterativeSolver(const dim dimin, const double alpha, const std::size_t niter, const double eps) {
    // Get the dimensions and size of the images
    _dimin  = dimin;
    _step   = dim(1, dimin.x);
    _sizein = dimin.x * dimin.y;

    // Allocate memory for image gradients
    _spatial_gradient_image = new opticalflow::Motion(_dimin);
    _temporal_derivative_image = new opticalflow::Image(_dimin);
    _horn_schunck_average = new opticalflow::Motion(dimin);
    _alpha = alpha;
    _eps = eps;
    _niter = niter;
}

IterativeSolver::~IterativeSolver() {
    delete _horn_schunck_average;
    delete _spatial_gradient_image;
    delete _temporal_derivative_image;
}

double IterativeSolver::get_alpha() const {
    return _alpha;
}

// Estimate motion from Horn-Schunck model
void IterativeSolver::estimate_optical_flow(
    opticalflow::Motion& motion, 
    const opticalflow::Image& Iref, 
    const opticalflow::Image& Imov) {
    // Auxiliary variable
    opticalflow::Motion motion_new(motion.get_dimensions());

    // Dereference some variables
    opticalflow::Motion& horn_schunck_average = *_horn_schunck_average;
    opticalflow::Motion& spatial_gradient_image = *_spatial_gradient_image;
    opticalflow::Image& temporal_derivative_image = *_temporal_derivative_image;

    // Calculate spatial and temporal derivative
    gradients::gradient(spatial_gradient_image, Imov);
    temporal_derivative_image = Imov - Iref;

    // Regularization parameters
    double alphasq = _alpha * _alpha;

    for (std::size_t iter = 0; iter < _niter; ++iter) {
        gradients::horn_schunck_average(horn_schunck_average, motion);

        for (std::size_t idx = 0; idx < motion.get_size(); ++idx) {
            // Get values
            vector2d hs_avg = horn_schunck_average.get_val(idx);
            vector2d dI = spatial_gradient_image.get_val(idx);
            double It = temporal_derivative_image.get_val(idx);

            // Calculate prefactor
            double s = (dot(hs_avg, dI) + It) / (alphasq + normsq(dI));

            // Horn-Schunck iteration
            motion_new.set_val(hs_avg - s * dI, idx);
        }

        // Convergence criteria
        double du = opticalflow::motion::norm(motion - motion_new);
        double m = opticalflow::motion::norm(motion_new);
        if (du < _eps * std::max(1.0, m))
            break;

        // Move
        motion = std::move(motion_new);

    }
}

