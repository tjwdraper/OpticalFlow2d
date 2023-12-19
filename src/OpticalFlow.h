#ifndef _OPTICAL_FLOW_H_
#define _OPTICAL_FLOW_H_

#include <src/ImageRegistrationSolver.h>

class OpticalFlow : public ImageRegistrationSolver {
    public:
        // Constructors and deconstructors
        OpticalFlow(const dim dimin, const float alpha);
        ~OpticalFlow();

        // Overload function from base class
        void get_update(Motion *motion);

    private:
        void get_quasi_laplacian(const Motion* motion);

        void horn_schunck_iteration(Motion* motion);

        // Memory for quasi-laplacian
        Motion *qlaplacian;
};

#endif