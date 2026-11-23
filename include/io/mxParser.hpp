///////////////////////////////////////////////////////////////////////////////////////////////////
// This file defines helper function to read and parse model parameters from Matlabs workspace, 
// particularly Matlab structs stored a mxArray pointers.
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MX_PARSER_HPP_
#define _MX_PARSER_HPP_

#include <string>
#include <mex.h>
#include <type_traits>
#include <optional>
#include <algorithm>
#include <cstddef>
#include <map>

#include "coord2d.hpp"

namespace mxParser {
    template <typename T>
    T parse_scalar(const mxArray* config, const char* fieldname, const std::optional<T> default_value = std::nullopt) {
        const mxArray* field = mxGetField(config, 0, fieldname);
        if (!field && !default_value.has_value()) {
            mexErrMsgIdAndTxt("mxParser:InputError", "field %s not found.", fieldname);
        }
        else if (!field && default_value.has_value()) {
            mexWarnMsgIdAndTxt("mxParser:InputWarning", "field %s not found. set to default value.", fieldname);
            return default_value.value();
        }

        if (!mxIsScalar(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s must be a scalar.", fieldname);
        }

        if (mxIsComplex(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s must contain real values.", fieldname);
        }
        return static_cast<T>(mxGetScalar(field));
    }

    template <typename T>
    T parse_string(const mxArray* config, const char* fieldname, const std::map<std::string, T>& mapper, const std::optional<T> default_value = std::nullopt) {
        const mxArray* field = mxGetField(config, 0, fieldname);
        if (!field && !default_value.has_value()) {
            mexErrMsgIdAndTxt("mxParser:InputError", "field %s not found.", fieldname);
        }
        else if (!field && default_value.has_value()) {
            mexWarnMsgIdAndTxt("mxParser:InputWarning", "field %s not found. set to default value.", fieldname);  
            return default_value.value();
        }

        if (!mxIsChar(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s is not a valid character array (did you try \" instead of \'?).", fieldname);
        }

        char* field_c = mxArrayToString(field);
        std::string key(field_c);
        mxFree(field_c);

        auto it = mapper.find(key);
        if (it == mapper.end()) {
            mexErrMsgIdAndTxt("mxParser:InputError", "invalid field option given for field name %s.", fieldname);
        }

        return it->second;
    }

    inline dim parse_dim(const mxArray* config, const char* fieldname, const std::optional<dim> default_value = std::nullopt) {
        const mxArray* field = mxGetField(config, 0, fieldname);
        if (!field && !default_value.has_value()) {
            mexErrMsgIdAndTxt("mxParser:InputError", "field %s not found.", fieldname);
        }
        else if (!field && default_value.has_value()) {
            mexWarnMsgIdAndTxt("mxParser:InputWarning", "field %s not found. set to default value.", fieldname);    
            return default_value.value();
        }

        if (!mxIsInt32(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s is not a valid int32 array.", fieldname);
        }

        if (mxIsComplex(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s must contain real values.", fieldname);
        }

        if (mxGetNumberOfElements(field) != 2) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s is not of length two (2).", fieldname);
        }

        int32_T* dim_arr = static_cast<int32_T*>(mxGetData(field));
        return dim(dim_arr[0], dim_arr[1]);
    }

    template <typename T>
    T* parse_array_ptr(const mxArray* config, const char* fieldname) {
        const mxArray* field = mxGetField(config, 0, fieldname);
        if (!field) {
            mexErrMsgIdAndTxt("mxParser:InputError", "field %s not found.", fieldname);        
        }
        if (!mxIsNumeric(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s does not contain numeric data.", fieldname);
        }
        if (mxIsComplex(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s must contain real values.", fieldname);
        }

        if constexpr (std::is_same_v<T, int32_T>) {
            if (mxGetClassID(field) != mxINT32_CLASS) {
                mexErrMsgIdAndTxt("mxParser:TypeError", "the array contained in field %s must be type int32.", fieldname);
            }
            return static_cast<int32_T*>(mxGetData(field));
        }
        else if constexpr (std::is_same_v<T, double>) {
            if (mxGetClassID(field) != mxDOUBLE_CLASS) {
                mexErrMsgIdAndTxt("mxParser:TypeError", "the array contained in field %s must be type double.", fieldname);
            }
            return static_cast<double*>(mxGetData(field));
        }
        else {
            mexErrMsgIdAndTxt("mxParser:TypeError", "the array contained in field %s must be either of type int32 or double.", fieldname);
        }
    }

    inline std::size_t parse_array_size(const mxArray* config, const char* fieldname) {
        const mxArray* field = mxGetField(config, 0, fieldname);
        if (!field) {
            mexErrMsgIdAndTxt("mxParser:InputError", "field %s not found.", fieldname);        
        }
        if (!mxIsNumeric(field)) {
            mexErrMsgIdAndTxt("mxParser:TypeError", "field %s does not contain numeric data.", fieldname);
        }
        return static_cast<std::size_t>(mxGetNumberOfElements(field));
    }

}
#endif