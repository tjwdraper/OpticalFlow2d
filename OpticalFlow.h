#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <coord2d.h>
#include <Image.h>
#include <Motion.h>
#include <OpticalFlowSolver.h>

class OpticalFlow {
    public:
        // Constructors and deconstructors
        OpticalFlow(const dim dimin, const int nscales, const int* niter, const float alpha);
        ~OpticalFlow();

        // Getters and setters
        void set_reference_image(const Image& im);
        void set_moving_image(const Image& im);
        Motion* get_estimated_motion() const;

        // Copy the estimated motion
        void copy_estimated_motion(Motion& mo) const;

        // Estimate motion
        void estimate_motion();

    private:
        void display_registration_parameters(const float alpha) const;

        void estimate_motion_at_current_resolution(Motion* motion, 
                                                    const Image *Iref, Image *Imov,
                                                    OpticalFlowSolver *solver, 
                                                    const int niter,
                                                    const dim dimin, const int sizein);

        dim *dimin;
        int *sizein;
        int nscales;
        int *niter;

        OpticalFlowSolver **solver;

        Image **Iref;
        Image **Imov;

        Motion **motion;
};

#endif