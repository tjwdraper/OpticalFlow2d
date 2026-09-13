#ifndef _GRADIENTS_H_
#define _GRADIENTS_H_

#include "include/coord2d.hpp"
#include "include/Field.hpp"

namespace gradients {
    template <typename T>
    // First order partial derivatives
    __inline__ T partial_x(T* field, const unsigned int idx, const unsigned int i, const dim& dimin) {
        if (i == 0) {
            return field[idx+1] - field[idx]; 
        }
        else if (i == dimin.x - 1) {
            return field[idx] - field[idx - 1];
        }
        else {
            return (field[idx+1] - field[idx-1])/2.0f;
        }
    }

    template <typename T>
    __inline__ T partial_y(T *field, const unsigned int idx, const unsigned int j, const dim& dimin) {
        if (j == 0) {
            return field[idx + dimin.x] - field[idx];
        }
        else if (j == dimin.y - 1) {
            return field[idx] - field[idx - dimin.x];
        }
        else {
            return (field[idx + dimin.x] - field[idx-dimin.x])/2.0f;
        }
    }

    // Second order partial derivatives
    template <typename T>
    __inline__ T partial_xx(T *field, const unsigned int idx, const unsigned int i, const dim& dimin) {
        if (i == 0) {
            return field[idx]*2 - field[idx+1]*5 + field[idx+2]*4 - field[idx+3];
        }
        else if (i == dimin.x-1) {
            return field[idx-3]*-1 + field[idx-2]*4 - field[idx-1]*5 + 2 * field[idx];
        }
        else {
            return field[idx+1] - field[idx]*2 + field[idx-1];
        }
    }

    template <typename T>
    __inline__ T partial_yy(T *field, const unsigned int idx, const unsigned int j, const dim& dimin) {
        if (j == 0) {
            return field[idx]*2 - field[idx+1*dimin.x]*5 + field[idx+2*dimin.x]*4 - field[idx+3*dimin.x];
        }
        else if (j == dimin.y-1) {
            return field[idx-3*dimin.x]*-1 + field[idx-2*dimin.x]*4 - field[idx-1*dimin.x]*5 + 2 * field[idx];
        }
        else {
            return field[idx+dimin.x] - field[idx]*2 + field[idx-dimin.x];
        }
    }

    template <typename T>
    __inline__ T partial_xy(T *field, const unsigned int idx, const unsigned int i, const unsigned int j, const dim& dimin) {
        if ((i == 0) || (j == 0) || (i == dimin.x-1) || (j == dimin.y-1)) {
            return T(0.0f);
        }
        else {
            return (field[idx + 1 + dimin.x] - field[idx + 1 - dimin.x] - field[idx -1 + dimin.x] + field[idx - 1 - dimin.x]) / 4.0f;
        }
    }

    template <typename T>
    __inline__ T qlaplacian(T *field, const unsigned int idx, const unsigned int i, const unsigned int j, const dim& dimin) {
        if ((i == 0) || (i == dimin.x-1) ||
            (j == 0) || (j == dimin.y-1)) {
            return T(0.0f);
        }
        else {
            return (field[idx - 1] + field[idx + 1] + field[idx - dimin.x] + field[idx + dimin.x])/4.0f;
        }
    }
    void jacobian(opticalflow::Image& image, const opticalflow::Motion& motion) {
        // Check that input dimensions are OK
        if (image.get_dimensions() != motion.get_dimensions())
            throw std::runtime_error("Error in Image::warp2d(const Motion& mo): input dimensions have to be the same as target");

        // Get the dimensions of the image
        const dim dimin = image.get_dimensions();
        const dim step = image.get_step();

        // Store the input in the object
        std::size_t idx;
        vector2d dudx, dudy;
        for (std::size_t i = 0; i < dimin.x; i++) {
            for (std::size_t j = 0; j < dimin.y; j++) {
                idx = i * step.x + j * step.y;

                dudx = gradients::partial_x(motion.get_field(), idx, i, dimin);
                dudy = gradients::partial_y(motion.get_field(), idx, j, dimin);

                image.set_val(
                    (1.0 + dudx.x) * (1.0 + dudy.y) - dudx.y * dudy.x,
                    idx
                );
            }
        }

        // Done
        return;

    }
}

#endif