#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <src/Field.h>
#include <src/Motion.h>
#include <src/Kernel.h>

class Image : public Field<float> {
    public:
        // Constructors and deconstructors
        Image(const dim dimin);
        Image(const Image& im);
        ~Image();

        // Getters and setters
        void set_image(const double* im);
        float* get_image() const;

        // 
        void copy_image_to_input(double* im) const;

        // Upsample and downsample
        void upSample(const Image& im);
        void downSample(const Image& im);

        // Get some information from the image
        float sum() const;
        float max() const;
        float min() const;

        // Normalize the image
        void normalize();

        // Warp image with motion field
        void warp2d(const Motion& mo);

        // Convolute with kernel
        void convolute(const Kernel& kernel);

        // Jacobian of a motion field
        void jacobian(const Motion& mo);

        // Overload operator=
        Image& operator=(const Image& im);
        
        // Overload operators from base class
        Image operator+(const Image& im) const;
        Image& operator+=(const Image& im);
        Image operator-(const Image& im) const;
        Image& operator-=(const Image& im);

        Image& operator*=(const float& val);
};

#endif