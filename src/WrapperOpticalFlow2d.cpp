#include <cstring>
#include <stdexcept>
#include <mex.h>

#include "coord2d.hpp"
#include "Field.hpp"
#include "ImageRegistration.h"
#include "interp2d.hpp"
#include "mxParseRegistrationConfiguration.hpp"

#include <memory>

static dim dimin;
static std::vector<mwSize> dim_image_mw(2); // 2d-vector for image dimensions
static std::vector<mwSize> dim_motion_mw(3); // 3d-vector for motion dimensions (2 channels for x and y)

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Set registration parameters
    if ((nlhs == 0) && (nrhs == 1) && (!ImageRegistration::isInitialized())) {
        // Read the registration configuration from the input mxArray
        RegistrationConfiguration config = mxParseRegistrationConfiguration(prhs[0]);

        // Create the ImageRegistration instance with the parsed configuration
        ImageRegistration::createInstance(config);

        // Set the output dimension for image and motion field
        dimin = config.size_image();

        // Allocate memory for the output dimensions
        dim_image_mw = {dimin.x, dimin.y};
        dim_motion_mw = {dimin.x, dimin.y, 2};
    }

    // Load the images and estimate motion through image registration
    else if ((nlhs == 0) && (nrhs == 2) && (ImageRegistration::isInitialized())) {
        opticalflow::Image Iref(dimin);
        opticalflow::Image Imov(dimin);

        ImageRegistration& myImageRegistration = ImageRegistration::getInstance();

        double *tmp;
        // Load the reference and moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::load_image(tmp, Iref);
        myImageRegistration.set_reference_image(Iref);

        tmp = mxGetPr(prhs[1]);
        opticalflow::image::load_image(tmp, Imov);
        myImageRegistration.set_moving_image(Imov);

        // Do the registration
        myImageRegistration.estimate_optical_flow();
    }

    // Return the motion field
    else if ((nlhs == 1) && (nrhs == 0) && (ImageRegistration::isInitialized())) {
        ImageRegistration& myImageRegistration = ImageRegistration::getInstance();

        const opticalflow::Motion& motion = myImageRegistration.get_estimated_motion();

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw.data(), mxDOUBLE_CLASS, mxREAL);
        double *tmp = mxGetPr(plhs[0]);

        // Fill the data with the motion field
        opticalflow::motion::save_motion(tmp, motion);
    }

    else if ((nlhs == 2) && (nrhs == 0) && (ImageRegistration::isInitialized())) {
        ImageRegistration& myImageRegistration = ImageRegistration::getInstance();

        const opticalflow::Motion& motion = myImageRegistration.get_estimated_motion();
        const opticalflow::Image& c = myImageRegistration.get_estimated_c();

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw.data(), mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericArray(2, dim_image_mw.data(), mxDOUBLE_CLASS, mxREAL);

        double *tmp = mxGetPr(plhs[0]);
        opticalflow::motion::save_motion(tmp, motion);
        tmp = mxGetPr(plhs[1]);
        opticalflow::image::save_image(tmp, c);
    }

    // Warp the input image with the estimated motion
    else if((nlhs == 1) && (nrhs == 1) && (ImageRegistration::isInitialized())) {
        ImageRegistration& myImageRegistration = ImageRegistration::getInstance();

        opticalflow::Image Imov(dimin);
        opticalflow::Image Ireg(dimin);
        const opticalflow::Motion& motion = myImageRegistration.get_estimated_motion();

        double *tmp;
        // Load the moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::load_image(tmp, Imov);

        // Warp the image with the motion field
        interp2d::warp2d(Ireg, Imov, motion);

        // Create output array and poiter to data
        plhs[0] = mxCreateNumericArray(2, dim_image_mw.data(), mxDOUBLE_CLASS, mxREAL);
        tmp = mxGetPr(plhs[0]);

        // Fill the data with the registered image
        opticalflow::image::save_image(tmp, Ireg);
    }

    // Close the library
    else if ((nlhs == 0) && (nrhs == 0) && (ImageRegistration::isInitialized())) {
        ImageRegistration::shutdownInstance();

        dim_image_mw.clear();
        dim_motion_mw.clear();
    }

    else {
        mexErrMsgTxt("Error: invalid number of input and output variables gives.\n");
    }

    // Done
    return;
}