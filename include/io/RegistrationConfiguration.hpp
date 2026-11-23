////////////////////////////////////////////////////////////////////////////////////////////////////
// This file defines two structures:
// 1. SolverParameters: contains the parameters for the IterativeSolver class, which are used to 
//      estimate the optical flow at any given resolution scale.
// 2. RegistrationConfiguration: contains the parameters for the ImageRegistration class, which
//      are used to configure the multiscale image registration process.
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CONFIGURATION_OPTIONS_HPP_
#define _CONFIGURATION_OPTIONS_HPP_

#include <string>
#include <map>
#include <vector>

#include "coord2d.hpp"
#include "mxParser.hpp"

enum class VerboseOption {SILENT, DISABLE_WARNING, VERBOSE};
enum class ModelOption {HORN_SCHUNCK, CORNELIUS_KANADE};

inline const std::map<std::string, VerboseOption> mapper_verbose_option {
    {"silent", VerboseOption::SILENT},
    {"disable-warnings", VerboseOption::DISABLE_WARNING},
    {"verbose", VerboseOption::VERBOSE}
};

inline const std::map<std::string, ModelOption> mapper_model_option {
    {"horn-schunck", ModelOption::HORN_SCHUNCK},
    {"cornelius-kanade", ModelOption::CORNELIUS_KANADE}
};

struct SolverParameters {
    dim dimin; 

    double alpha;
    double beta;
    double eps;
    std::size_t niter;
};

struct RegistrationConfiguration {
    VerboseOption verbose_option;
    ModelOption model_option;

    std::size_t nscales;
    std::size_t nrefine;
    std::vector<SolverParameters> solver_parameters;

    double resampling_factor;

    dim size_image() {
        if (solver_parameters.empty())
            throw std::logic_error("Error: solver_parameters is empty.");
        return solver_parameters[0].dimin;
    }
};

#endif