#include <cstring>
#include <stdexcept>
#include <mex.h>

#include "coord2d.hpp"
#include "Field.hpp"
#include "ImageRegistration.h"
#include "interp2d.hpp"
#include "mxParser.hpp"


static ImageRegistration *myImageRegistration = nullptr;
static mwSize *dim_image_mw;
static mwSize *dim_motion_mw;
static dim dimin;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Set registration parameters
    if ((nlhs == 0) && (nrhs == 1) && (myImageRegistration == nullptr)) {
        mxParser::parse_size_image(dimin, prhs[0]);

        // Registration parameters
        ModelOption option;
        mxParser::parse_opticalflow_option(option, prhs[0]);

        std::size_t nscales, nrefine;
        mxParser::parse_nscales(nscales, prhs[0]);
        mxParser::parse_nrefine(nrefine, prhs[0]);

        double alpha;
        mxParser::parse_alpha(alpha, prhs[0]);


        double beta;
        mxParser::parse_beta(beta, prhs[0]);

        std::size_t* niter = new size_t[nscales+1];
        mxParser::parse_niter(niter, prhs[0]);

        double eps;
        mxParser::parse_convergence_threshold(eps, prhs[0]);

        double resampling_factor;
        mxParser::parse_resampling_factor(resampling_factor, prhs[0]);

        myImageRegistration = new ImageRegistration(dimin, option, nscales, niter, alpha, beta, eps, nrefine, resampling_factor);

        // Set the output dimension for image and motion field
        dim_image_mw = new mwSize[2];
        dim_image_mw[0] = dimin.x;
        dim_image_mw[1] = dimin.y;

        dim_motion_mw = new mwSize[3];
        dim_motion_mw[0] = dimin.x;
        dim_motion_mw[1] = dimin.y;
        dim_motion_mw[2] = 2;

        // Free memory
        delete[] niter;
    }

    // Load the images and estimate motion through image registration
    else if ((nlhs == 0) && (nrhs == 2) && (myImageRegistration != nullptr)) {
        opticalflow::Image Iref(dimin);
        opticalflow::Image Imov(dimin);

        double *tmp;
        // Load the reference and moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::load_image(tmp, Iref);
        myImageRegistration->set_reference_image(Iref);

        tmp = mxGetPr(prhs[1]);
        opticalflow::image::load_image(tmp, Imov);
        myImageRegistration->set_moving_image(Imov);

        // Do the registration
        myImageRegistration->estimate_optical_flow();
    }

    // Return the motion field
    else if ((nlhs == 1) && (nrhs == 0) && (myImageRegistration != nullptr)) {
        const opticalflow::Motion& motion = myImageRegistration->get_estimated_motion();

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw, mxDOUBLE_CLASS, mxREAL);
        double *tmp = mxGetPr(plhs[0]);

        // Fill the data with the motion field
        opticalflow::motion::save_motion(tmp, motion);
    }

    else if ((nlhs == 2) && (nrhs == 0) && (myImageRegistration != nullptr)) {
        const opticalflow::Motion& motion = myImageRegistration->get_estimated_motion();
        const opticalflow::Image& c = myImageRegistration->get_estimated_c();

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw, mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericArray(2, dim_image_mw, mxDOUBLE_CLASS, mxREAL);

        double *tmp = mxGetPr(plhs[0]);
        opticalflow::motion::save_motion(tmp, motion);
        tmp = mxGetPr(plhs[1]);
        opticalflow::image::save_image(tmp, c);
    }

    // Warp the input image with the estimated motion
    else if((nlhs == 1) && (nrhs == 1) && (myImageRegistration != nullptr)) {
        opticalflow::Image Imov(dimin);
        opticalflow::Image Ireg(dimin);
        const opticalflow::Motion& motion = myImageRegistration->get_estimated_motion();

        double *tmp;
        // Load the moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::load_image(tmp, Imov);

        // Warp the image with the motion field
        interp2d::warp2d(Ireg, Imov, motion);

        // Create output array and postd::size_ter to data
        plhs[0] = mxCreateNumericArray(2, dim_image_mw, mxDOUBLE_CLASS, mxREAL);
        tmp = mxGetPr(plhs[0]);

        // Fill the data with the registered image
        opticalflow::image::save_image(tmp, Ireg);
    }

    // Close the library
    else if ((nlhs == 0) && (nrhs == 0) && (myImageRegistration != nullptr)) {
        // Deallocate and free all the mem
        delete myImageRegistration;
        myImageRegistration = nullptr;

        delete[] dim_image_mw;
        delete[] dim_motion_mw;
    }

    else {
        mexErrMsgTxt("Error: invalid number of input and output variables gives.\n");
    }

    // Done
    return;
}