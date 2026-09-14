#ifndef _CONFIGURATION_OPTIONS_HPP_
#define _CONFIGURATION_OPTIONS_HPP_

#include <string>
#include <map>

enum class VerboseOption {SILENT, DISABLE_WARNING, VERBOSE};

extern const std::map<std::string, VerboseOption> mapper_verbose_option {
    {"silent", VerboseOption::SILENT},
    {"disable_warnings", VerboseOption::DISABLE_WARNING},
    {"verbose", VerboseOption::VERBOSE}
};

#endif