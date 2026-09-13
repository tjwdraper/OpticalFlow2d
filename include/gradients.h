#ifndef _GRADIENTS_H_
#define _GRADIENTS_H_

#include "include/coord2d.hpp"
#include "include/Field.hpp"

namespace gradients {
    // First order partial derivatives
    template <typename T>
    inline T partial_x(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.x < 2)
            throw std::runtime_error("In T gradients::partial_x(const Field<T>&, const std::size_t, const std::size_t), x-dimension must be at least 2.");

        if (i == 0)
            return field.get_val(i+1,j) - field.get_val(i,j); 
        else if (i == dimin.x-1)
            return field.get_val(i,j) - field.get_val(i-1, j);
        else
            return (field.get_val(i+1,j) - field.get_val(i-1,j)) / 2.0; 
    }

    template <typename T>
    inline T partial_y(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.y < 2)
            throw std::runtime_error("In T gradients::partial_y(const Field<T>&, const std::size_t, const std::size_t), y-dimension must be at least 2.");

        if (j == 0)
            return field.get_val(i,j+1) - field.get_val(i,j); 
        else if (j == dimin.y-1)
            return field.get_val(i,j) - field.get_val(i, j-1);
        else
            return (field.get_val(i,j+1) - field.get_val(i,j-1)) / 2.0; 
    }

    template <typename T>
    inline T partial_xx(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.x < 4)
            throw std::runtime_error("In T gradients::partial_xx(const Field<T>&, const std::size_t, const std::size_t), x-dimension must be at least 4")

        if (i == 0)
            return 2.0*field.get_val(i,j) - 5.0*field.get_val(i+1,j) + 4.0*field.get_val(i+2,j) - field.get_val(i+3,j);
        else if (i == dimin.x-1)
            return 2.0*field.get_val(i,j) - 5.0*field.get_val(i-1,j) + 4.0*field.get_val(i-2,j) - field.get_val(i-3,j);
        else
            return field.get_val(i+1,j) - 2.0*field.get_val(i,j) + field.get_val(i-1,j);
    }

    template <typename T>
    inline T partial_yy(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.y < 4)
            throw std::runtime_error("In T gradients::partial_yy(const Field<T>&, const std::size_t, const std::size_t), y-dimension must be at least 4");

        if (j == 0)
            return 2.0*field.get_val(i,j) - 5.0*field.get_val(i,j+1) + 4.0*field.get_val(i,j+2) - field.get_val(i,j+3);
        else if (j == dimin.y-1)
            return 2.0*field.get_val(i,j) - 5.0*field.get_val(i,j-1) + 4.0*field.get_val(i,j-2) - field.get_val(i,j-3);
        else
            return field.get_val(i,j+1) - 2.0*field.get_val(i,j) + field.get_val(i,j-1);
    }

    template <typename T>
    inline T partial_xy(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.x < 1 || dimin.y < 1)
            throw std::runtime_error("In T gradients::partial_xy(const Field<T>&, const std::size_t, const std::size_t), x and y dimensions must be at least 1");

        if (i == 0 || j == 0 || i == dimin.x-1 || j == dimin.y-1)
            return T{};

        return (field.get_val(i+1,j+1) - field.get_val(i+1,j-1) - field.get_val(i-1,j+1) + field.get_val(i-1,j-1)) / 4.0;
    }

    template <typename T>
    inline T horn_schunck_average(const Field<T>& field, const std::size_t i, const std::size_t j) {
        const dim dimin = field.get_dimensions();
        if (dimin.x < 2 || dimin.y < 2)
            throw  std::runtime_error("In T gradients::horn_schunck_average(const Field<T>&, const std::size_t, const std::size_t), field dimensions must be at least 2x2.");

        T val{};

        if (i == 0) {
            val += field.get_val(i,j);
            val += field.get_val(i+1,j);
        }
        else if (i == dimin.x-1) {
            val += field.get_val(i-1,j);
            val += field.get_val(i,j);
        }
        else {
            val += field.get_val(i-1,j);
            val += field.get_val(i+1,j);
        }

        if (j == 0) {
            val += field.get_val(i,j);
            val += field.get_val(i,j+1);
        }
        else if (j == dimin.y-1) {
            val += field.get_val(i,j-1);
            val += field.get_val(i,j);
        }
        else {
            val += field.get_val(i,j-1);
            val += field.get_val(i,j+1);
        }

        return val / 4.0;

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

                dudx = gradients::partial_x(motion, idx, i, dimin);
                dudy = gradients::partial_y(motion, idx, j, dimin);

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