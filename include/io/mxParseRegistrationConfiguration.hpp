////////////////////////////////////////////////////////////////////////////////////////////////////
// This file defines a helper function to parse the registration configuration from a Matlab struct
// stored in a mxArray pointer. The function reads the fields of the struct and populates
// a RegistrationConfiguration object with the corresponding values. It also performs validation
// checks on the parsed values to ensure they are within acceptable ranges.
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef MX_PARSE_REGISTRATION_CONFIGURATION_HPP
#define MX_PARSE_REGISTRATION_CONFIGURATION_HPP

#include <vector>

#include "mex.h"

#include "RegistrationConfiguration.hpp"

inline RegistrationConfiguration mxParseRegistrationConfiguration(const mxArray* input) {
    RegistrationConfiguration config;

    // Parse model and multiscale parameters
    config.verbose_option = mxParser::parse_string<VerboseOption>(input, "verbose_option", mapper_verbose_option, VerboseOption::VERBOSE);
    config.model_option = mxParser::parse_string<ModelOption>(input, "optical_flow_option", mapper_model_option, ModelOption::HORN_SCHUNCK);
    config.nscales = mxParser::parse_array_size(input, "niter") - 1;
    config.nrefine = mxParser::parse_scalar<std::size_t>(input, "nrefine", std::optional<std::size_t>(0));
    
    // Parse the resampling factor
    double resampling_factor = mxParser::parse_scalar<double>(input, "resampling_factor", std::optional<double>(0.5));
    if (resampling_factor <= 0.0 || resampling_factor >= 1.0)
        mexErrMsgIdAndTxt("mxParser:ValueError", "resampling_factor must be in the range (0, 1).");
    config.resampling_factor = resampling_factor;

    // Parse registration parameters for each scale
    dim dimin = mxParser::parse_dim(input, "size_image");

    double alpha = mxParser::parse_scalar<double>(input, "alpha", std::optional<double>(0.4));
    if (alpha < 0.0)
        mexErrMsgIdAndTxt("mxParser:ValueError", "alpha has to be positive.");

    double beta = mxParser::parse_scalar<double>(input, "beta", std::optional<double>(0.2));
    if (beta < 0.0)
        mexErrMsgIdAndTxt("mxParser:ValueError", "beta has to be positive.");

    double eps = mxParser::parse_scalar<double>(input, "eps", std::optional<double>(1e-3));
    
    std::vector<std::size_t> niter(config.nscales + 1);
    int32_T* niter_arr = mxParser::parse_array_ptr<int32_T>(input, "niter");
    for (std::size_t i = 0; i < config.nscales + 1; ++i) {
        if (niter_arr[i] < 0)
            mexErrMsgIdAndTxt("mxParser:ValueError", "field %s contains negative values.", "niter");
        niter[i] = static_cast<std::size_t>(niter_arr[i]);
    }

    // Set registration parameters per level
    config.solver_parameters.resize(config.nscales + 1);
    for (int s = static_cast<int>(config.nscales); s >= 0; --s) {
        // Resample the image dimensions for the current scale
        double scale = pow(config.resampling_factor, s);
        const dim dim_s = dim(
            static_cast<std::size_t> (dimin.x * scale),
            static_cast<std::size_t> (dimin.y * scale)
        );

        // Create SolverParameters object
        SolverParameters solver_params;

        solver_params.dimin = dim_s;
        solver_params.step = dim(1, dim_s.x);
        solver_params.size = dim_s.x * dim_s.y;
        
        solver_params.alpha = alpha;
        solver_params.beta = beta;
        solver_params.eps = eps;
        solver_params.niter = niter[s];

        // Set the solver parameters for the current scale
        config.solver_parameters[s] = solver_params;
    }
    
    // Done
    return config;
}

#endif