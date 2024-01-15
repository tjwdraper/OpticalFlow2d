#include <cstring>
#include <stdexcept>
#include <mex.h>

#include <src/coord2d.h>
#include <src/Image.h>
#include <src/Motion.h>
#include <src/ImageRegistrationOpticalFlow.h>
#include <src/ImageRegistrationDemons.h>
#include <src/ImageRegistrationFluid.h>
#include <src/SolverOptions.h>

static ImageRegistration *myImageRegistration = NULL;
static mwSize *dim_image_mw;
static mwSize *dim_motion_mw;
static dim dimin;

void
mexFunction (int nlhs, mxArray *plhs[],
             int nrhs, const mxArray *prhs[])
{
    // Set registration parameters
    if ((nlhs == 0) && (nrhs == 8) && (myImageRegistration == NULL)) {
        // Get the dimensions and the size of the images
        double *tmp;
        tmp = mxGetPr(prhs[0]);
        int dimx = (int) tmp[0];
        int dimy = (int) tmp[1];
        dimin = dim(dimx, dimy);

        // Get the registration parameters
        tmp = mxGetPr(prhs[2]);
        int nscales = (int) tmp[0];
        tmp = mxGetPr(prhs[1]);
        int *niter = new int[nscales + 1];
        for (int s = 0; s < nscales + 1; s++) {
            niter[s] = (int) tmp[s];
        }
        tmp = mxGetPr(prhs[3]);
        Regularisation reg = static_cast<Regularisation>( (int) tmp[0]);

        tmp = mxGetPr(prhs[5]);
        unsigned int nparams = (unsigned int) tmp[0];
        tmp = mxGetPr(prhs[4]);
        float *regparams = new float[nparams];
        for (int p = 0; p < nparams; p++) {
            regparams[p] = (float) tmp[p];
        }

        tmp = mxGetPr(prhs[6]);
        int nrefine = (int) tmp[0];

        tmp = mxGetPr(prhs[7]);
        Verbose verb = static_cast<Verbose>( (int) tmp[0]);

        // Pass parameters to ImageRegistration object
        if ((reg == Regularisation::Diffusion) ||
            (reg == Regularisation::Curvature) ||
            (reg == Regularisation::Elastic)) {
            myImageRegistration = new ImageRegistrationOpticalFlow(dimin, nscales, niter, nrefine, reg, regparams, nparams, verb);
        }
        else if ((reg == Regularisation::ThirionsDemons) ||
                 (reg == Regularisation::DiffeomorphicDemons)) {
            myImageRegistration = new ImageRegistrationDemons(dimin, nscales, niter, nrefine, reg, regparams, nparams, verb);
        }
        else if (reg == Regularisation::Fluid) {
            myImageRegistration = new ImageRegistrationFluid(dimin, nscales, niter, nrefine, reg, regparams, nparams, verb);
        }
        else {
            mexErrMsgTxt("Error: invalid regularisation given\n");
        }


        // Set the output dimension for image and motion field
        dim_image_mw = new mwSize[2];
        dim_image_mw[0] = dimx; dim_image_mw[1] = dimy;
        dim_motion_mw = new mwSize[3];
        dim_motion_mw[0] = dimx; dim_motion_mw[1] = dimy; dim_motion_mw[2] = 2;

        // Free up the niter and regparams array
        delete[] niter;
        delete[] regparams;
    }

    // Load the images and estimate motion through image registration
    else if ((nlhs == 0) && (nrhs == 2) && (myImageRegistration != NULL)) {
        Image Iref(dimin);
        Image Imov(dimin);

        double *tmp;
        // Load the reference and moving image
        tmp = mxGetPr(prhs[0]);
        Iref.set_image(tmp);
        myImageRegistration->set_reference_image(Iref);

        tmp = mxGetPr(prhs[1]);
        Imov.set_image(tmp);
        myImageRegistration->set_moving_image(Imov);

        // Do the registration
        myImageRegistration->estimate_motion();
    }

    // Return the motion field
    else if ((nlhs == 1) && (nrhs == 0) && (myImageRegistration != NULL)) {
        Motion motion(dimin);

        // Copy estimated motion from device to host
        myImageRegistration->copy_estimated_motion(motion);

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(3, dim_motion_mw, mxDOUBLE_CLASS, mxREAL);
        double *tmp = mxGetPr(plhs[0]);

        // Fill the data with the motion field
        motion.copy_motion_to_input(tmp);
    }

    // Warp the input image with the estimated motion
    else if((nlhs == 1) && (nrhs == 1) && (myImageRegistration != NULL)) {
        Image Imov(dimin);

        double *tmp;
        // Load the moving image
        tmp = mxGetPr(prhs[0]);
        Imov.set_image(tmp);

        // Warp the image with the motion field
        Imov.warp2d(*(myImageRegistration->get_estimated_motion()));

        // Create output array and pointer to data
        plhs[0] = mxCreateNumericArray(2, dim_image_mw, mxDOUBLE_CLASS, mxREAL);
        tmp = mxGetPr(plhs[0]);

        // Fill the data with the registered image
        Imov.copy_image_to_input(tmp);
    }

    // Close the library
    else if ((nlhs == 0) && (nrhs == 0) && (myImageRegistration != NULL)) {
        // Deallocate and free all the mem
        delete myImageRegistration;
        myImageRegistration = NULL;

        delete[] dim_image_mw;
        delete[] dim_motion_mw;
    }

    else {
        mexErrMsgTxt("Error: invalid number of input and output variables gives.\n");
    }

    // Done
    return;
}