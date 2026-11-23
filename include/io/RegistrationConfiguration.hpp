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
};

#endif