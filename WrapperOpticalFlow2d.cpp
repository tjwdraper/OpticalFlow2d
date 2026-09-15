#include <cstring>
#include <stdexcept>
#include <mex.h>

#include "include/coord2d.hpp"
#include "include/Field.hpp"
#include "include/ImageRegistration.h"
#include "include/interp2d.hpp"
#include "include/mxParser.hpp"


static ImageRegistration *myImageRegistration = nullptr;
static mwSize *dim_image_mw;
static mwSize *dim_motion_mw;
static dim dimin;

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Set registration parameters
    if ((nlhs == 0) && (nrhs == 1) && (myImageRegistration == nullptr)) {
        mxParser::parse_size_image(dimin, prhs[0]);
        myImageRegistration = new ImageRegistration(prhs[0]);


        // // Get the dimensions and the size of the images
        // double *tmp;
        // tmp = mxGetPr(prhs[0]);
        // std::size_t dimx = (std::size_t) tmp[0];
        // std::size_t dimy = (std::size_t) tmp[1];
        // dimin = dim(dimx, dimy);

        // // Get the registration parameters
        // tmp = mxGetPr(prhs[2]);
        // std::size_t nscales = (std::size_t) tmp[0];
        // tmp = mxGetPr(prhs[1]);
        // std::size_t *niter = new std::size_t[nscales + 1];
        // for (std::size_t s = 0; s < nscales + 1; s++) {
        //     niter[s] = (std::size_t) tmp[s];
        // }

        // tmp = mxGetPr(prhs[3]);
        // double alpha = (double) tmp[0];

        // tmp = mxGetPr(prhs[4]);
        // double eps = (double) tmp[0];

        // tmp = mxGetPr(prhs[5]);
        // std::size_t nrefine = (std::size_t) tmp[0];

        // // Pass parameters to ImageRegistration object
        // myImageRegistration = new ImageRegistration(dimin, nscales, niter, alpha, eps, nrefine);

        // Set the output dimension for image and motion field
        dim_image_mw = new mwSize[2];
        dim_image_mw[0] = dimin.x;
        dim_image_mw[1] = dimin.y;

        dim_motion_mw = new mwSize[3];
        dim_motion_mw[0] = dimin.x;
        dim_motion_mw[1] = dimin.y;
        dim_motion_mw[2] = 2;
    }

    // Load the images and estimate motion through image registration
    else if ((nlhs == 0) && (nrhs == 2) && (myImageRegistration != nullptr)) {
        opticalflow::Image Iref(dimin);
        opticalflow::Image Imov(dimin);

        double *tmp;
        // Load the reference and moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::mex_load_image(tmp, Iref);
        myImageRegistration->set_reference_image(Iref);

        tmp = mxGetPr(prhs[1]);
        opticalflow::image::mex_load_image(tmp, Imov);
        myImageRegistration->set_moving_image(Imov);

        // Do the registration
        myImageRegistration->estimate_optical_flow();
    }

    // Return the motion field
    else if ((nlhs == 1) && (nrhs == 0) && (myImageRegistration != nullptr)) {
        const opticalflow::Motion& motion = myImageRegistration->get_estimated_motion();

        // Create output array and postd::size_ter to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw, mxDOUBLE_CLASS, mxREAL);
        double *tmp = mxGetPr(plhs[0]);

        // Fill the data with the motion field
        opticalflow::motion::mex_save_motion(tmp, motion);
    }

    // Warp the input image with the estimated motion
    else if((nlhs == 1) && (nrhs == 1) && (myImageRegistration != nullptr)) {
        opticalflow::Image Imov(dimin);
        opticalflow::Image Ireg(dimin);
        const opticalflow::Motion& motion = myImageRegistration->get_estimated_motion();

        double *tmp;
        // Load the moving image
        tmp = mxGetPr(prhs[0]);
        opticalflow::image::mex_load_image(tmp, Imov);

        // Warp the image with the motion field
        interp2d::warp2d(Ireg, Imov, motion);

        // Create output array and postd::size_ter to data
        plhs[0] = mxCreateNumericArray(2, dim_image_mw, mxDOUBLE_CLASS, mxREAL);
        tmp = mxGetPr(plhs[0]);

        // Fill the data with the registered image
        opticalflow::image::mex_save_image(tmp, Ireg);
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