#ifndef _IMAGE_H_
#define _IMAGE_H_

#include <src/Field.h>
#include <src/Motion.h>

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
};

#endif