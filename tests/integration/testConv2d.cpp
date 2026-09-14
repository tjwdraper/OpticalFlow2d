#include "include/conv2d.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

using opticalflow::Image;


// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

TEST(Conv2dTest, ConstructsWithValidDimensions)
{
    const dim d(3, 5);

    conv2d filter(d);

    EXPECT_EQ(filter.get_dimensions().x, 3);
    EXPECT_EQ(filter.get_dimensions().y, 5);
    EXPECT_EQ(filter.get_size(), 15);
}

TEST(Conv2dTest, RejectsZeroDimensions)
{
    EXPECT_THROW(
        conv2d(dim(0, 3)),
        std::runtime_error
    );

    EXPECT_THROW(
        conv2d(dim(3, 0)),
        std::runtime_error
    );
}

TEST(Conv2dTest, RejectsEvenDimensions)
{
    EXPECT_THROW(
        conv2d(dim(2, 3)),
        std::runtime_error
    );

    EXPECT_THROW(
        conv2d(dim(3, 2)),
        std::runtime_error
    );

    EXPECT_THROW(
        conv2d(dim(4, 4)),
        std::runtime_error
    );
}


// -----------------------------------------------------------------------------
// Weights
// -----------------------------------------------------------------------------

TEST(Conv2dTest, WeightsCanBeSetAndRetrieved)
{
    conv2d filter(dim(3, 3));

    filter.set_weight(1.0, 0, 0);
    filter.set_weight(2.0, 0, 1);
    filter.set_weight(3.0, 0, 2);

    filter.set_weight(4.0, 1, 0);
    filter.set_weight(5.0, 1, 1);
    filter.set_weight(6.0, 1, 2);

    filter.set_weight(7.0, 2, 0);
    filter.set_weight(8.0, 2, 1);
    filter.set_weight(9.0, 2, 2);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(
                filter.get_weight(i, j),
                static_cast<double>(3 * i + j + 1)
            );
        }
    }
}

TEST(Conv2dTest, GetWeightRejectsOutOfBounds)
{
    conv2d filter(dim(3, 5));

    EXPECT_THROW(
        filter.get_weight(3, 0),
        std::runtime_error
    );

    EXPECT_THROW(
        filter.get_weight(0, 5),
        std::runtime_error
    );
}

TEST(Conv2dTest, SetWeightRejectsOutOfBounds)
{
    conv2d filter(dim(3, 5));

    EXPECT_THROW(
        filter.set_weight(1.0, 3, 0),
        std::runtime_error
    );

    EXPECT_THROW(
        filter.set_weight(1.0, 0, 5),
        std::runtime_error
    );
}


// -----------------------------------------------------------------------------
// Convolution
// -----------------------------------------------------------------------------

TEST(Conv2dTest, IdentityKernelLeavesImageUnchanged)
{
    Image image(dim(5, 7));

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 7; ++j) {
            image.set_val(
                static_cast<double>(10 * i + j),
                i,
                j
            );
        }
    }

    conv2d filter(dim(3, 3));

    // Identity kernel
    filter.set_weight(0.0, 0, 0);
    filter.set_weight(0.0, 0, 1);
    filter.set_weight(0.0, 0, 2);

    filter.set_weight(0.0, 1, 0);
    filter.set_weight(1.0, 1, 1);
    filter.set_weight(0.0, 1, 2);

    filter.set_weight(0.0, 2, 0);
    filter.set_weight(0.0, 2, 1);
    filter.set_weight(0.0, 2, 2);

    filter.convolute(image);

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 7; ++j) {
            EXPECT_DOUBLE_EQ(
                image.get_val(i, j),
                static_cast<double>(10 * i + j)
            );
        }
    }
}


TEST(Conv2dTest, Convolves3x3KernelWithInteriorPixels)
{
    Image image(dim(5, 5));

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            image.set_val(
                static_cast<double>(5 * i + j + 1),
                i,
                j
            );
        }
    }

    conv2d filter(dim(3, 3));

    // Kernel:
    //
    // 0 1 0
    // 1 0 1
    // 0 1 0
    //
    filter.set_weight(0.0, 0, 0);
    filter.set_weight(1.0, 0, 1);
    filter.set_weight(0.0, 0, 2);

    filter.set_weight(1.0, 1, 0);
    filter.set_weight(0.0, 1, 1);
    filter.set_weight(1.0, 1, 2);

    filter.set_weight(0.0, 2, 0);
    filter.set_weight(1.0, 2, 1);
    filter.set_weight(0.0, 2, 2);

    filter.convolute(image);

    // Original image:
    //
    //  1  2  3  4  5
    //  6  7  8  9 10
    // 11 12 13 14 15
    // 16 17 18 19 20
    // 21 22 23 24 25
    //
    // At the center:
    //
    //      8
    //   12    14
    //     18
    //
    // = 52

    EXPECT_DOUBLE_EQ(image.get_val(2, 2), 52.0);

    // Another interior pixel:
    //
    //      13
    //   17     19
    //      23
    //
    // = 72

    EXPECT_DOUBLE_EQ(image.get_val(3, 3), 76.0);
}


TEST(Conv2dTest, BoundaryUsesClampedValues)
{
    Image image(dim(3, 3));

    // 1 2 3
    // 4 5 6
    // 7 8 9
    image.set_val(1.0, 0, 0);
    image.set_val(2.0, 0, 1);
    image.set_val(3.0, 0, 2);

    image.set_val(4.0, 1, 0);
    image.set_val(5.0, 1, 1);
    image.set_val(6.0, 1, 2);

    image.set_val(7.0, 2, 0);
    image.set_val(8.0, 2, 1);
    image.set_val(9.0, 2, 2);

    conv2d filter(dim(3, 3));

    // Average kernel
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            filter.set_weight(1.0 / 9.0, i, j);
        }
    }

    filter.convolute(image);

    // With clamped boundaries:
    //
    // Top-left neighborhood becomes
    //
    // 1 1 2
    // 1 1 2
    // 4 4 5
    //
    // sum = 21
    //
    // average = 21 / 9

    EXPECT_NEAR(
        image.get_val(0, 0),
        21.0 / 9.0,
        1e-12
    );

    // Center has the ordinary average:
    //
    // 1 2 3
    // 4 5 6
    // 7 8 9
    //
    // = 45 / 9 = 5

    EXPECT_DOUBLE_EQ(
        image.get_val(1, 1),
        5.0
    );
}


TEST(Conv2dTest, NonSquareKernelWorks)
{
    Image image(dim(5, 7));

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 7; ++j) {
            image.set_val(
                static_cast<double>(j),
                i,
                j
            );
        }
    }

    conv2d filter(dim(3, 5));

    // Only use the middle row:
    //
    // 0 0 0 0 0
    // 1 1 1 1 1
    // 0 0 0 0 0
    //
    for (std::size_t j = 0; j < 5; ++j) {
        filter.set_weight(0.0, 0, j);
        filter.set_weight(1.0, 1, j);
        filter.set_weight(0.0, 2, j);
    }

    filter.convolute(image);

    // At the center column j=3:
    //
    // 1 + 2 + 3 + 4 + 5 = 15

    EXPECT_DOUBLE_EQ(
        image.get_val(2, 3),
        15.0
    );
}


// -----------------------------------------------------------------------------
// Asymmetric kernel
// -----------------------------------------------------------------------------

TEST(Conv2dTest, AsymmetricKernelUsesExpectedConvolutionConvention)
{
    Image image(dim(3, 5));

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            image.set_val(
                static_cast<double>(j),
                i,
                j
            );
        }
    }

    conv2d filter(dim(1, 3));

    // Kernel:
    //
    // [1 2 3]
    //
    filter.set_weight(1.0, 0, 0);
    filter.set_weight(2.0, 0, 1);
    filter.set_weight(3.0, 0, 2);

    filter.convolute(image);

    // Your implementation evaluates:
    //
    // input(j-1)*1 + input(j)*2 + input(j+1)*3
    //
    // at j=2:
    //
    // 1*1 + 2*2 + 3*3 = 14

    EXPECT_DOUBLE_EQ(
        image.get_val(1, 2),
        14.0
    );
}


// -----------------------------------------------------------------------------
// Type support
// -----------------------------------------------------------------------------

TEST(Conv2dTest, ConvolutionWorksWithFieldOfVector2d)
{
    opticalflow::Motion motion(dim(3, 3));

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            motion.set_val(
                vector2d(
                    static_cast<double>(i),
                    static_cast<double>(j)
                ),
                i,
                j
            );
        }
    }

    conv2d filter(dim(3, 3));

    // Identity kernel
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            filter.set_weight(0.0, i, j);

    filter.set_weight(1.0, 1, 1);

    filter.convolute(motion);

    EXPECT_DOUBLE_EQ(motion.get_val(1, 1).x, 1.0);
    EXPECT_DOUBLE_EQ(motion.get_val(1, 1).y, 1.0);
}