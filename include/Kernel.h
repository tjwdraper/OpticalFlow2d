#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <src/coord2d.h>

class Kernel {
    public:
        // Constructors and deconstructors
        Kernel(const unsigned int kernelwidth);
        Kernel(const dim dimkernel);
        ~Kernel();

        // Getters and setters
        dim get_dimensions() const;
        dim get_step() const;
        unsigned int get_size() const;
        double *get_kernel() const;

        // Set values of the kernel
        void set_gaussian(const float sigma);
        void set_average();

    private:
        dim dimkernel;
        dim stepkernel;
        unsigned int sizekernel;
        double* kernel;
};

#endif