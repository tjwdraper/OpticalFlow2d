#include "gtest/gtest.h"
#include "include/conv2d.hpp"
#include "include/Field.hpp"

#include <cmath>

TEST(Conv2dIntegrationTest, AverageFilter3x3)
{
    // Input:
    //
    // 1 2 3
    // 4 5 6
    // 7 8 9
    //
    dim dimin(3, 3);
    opticalflow::Image image(dimin);

    for (std::size_t i = 0; i < dimin.x; ++i) {
        for (std::size_t j = 0; j < dimin.y; ++j) {
            image.set_val(
                static_cast<double>(i * dimin.y + j + 1),
                i,
                j
            );
        }
    }

    // Create a 3x3 average convolution.
    average_conv2d filter(dim(3, 3));

    // Check the kernel.
    EXPECT_DOUBLE_EQ(filter.get_weight_i(0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(filter.get_weight_i(1), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(filter.get_weight_i(2), 1.0 / 3.0);

    EXPECT_DOUBLE_EQ(filter.get_weight_j(0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(filter.get_weight_j(1), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(filter.get_weight_j(2), 1.0 / 3.0);

    // Apply the convolution.
    filter.convolute(image);

    // Expected result with clamped boundaries:
    //
    // 7/3   3   11/3
    // 11/3  5   19/3
    // 17/3  7   23/3
    //
    EXPECT_NEAR(image.get_val(0, 0), 7.0 / 3.0, 1e-12);
    EXPECT_NEAR(image.get_val(0, 1), 3.0,       1e-12);
    EXPECT_NEAR(image.get_val(0, 2), 11.0 / 3.0, 1e-12);

    EXPECT_NEAR(image.get_val(1, 0), 13.0 / 3.0, 1e-12);
    EXPECT_NEAR(image.get_val(1, 1), 5.0,        1e-12);
    EXPECT_NEAR(image.get_val(1, 2), 17.0 / 3.0, 1e-12);

    EXPECT_NEAR(image.get_val(2, 0), 19.0 / 3.0, 1e-12);
    EXPECT_NEAR(image.get_val(2, 1), 7.0,        1e-12);
    EXPECT_NEAR(image.get_val(2, 2), 23.0 / 3.0, 1e-12);
}


TEST(Conv2dIntegrationTest, AverageFilterPreservesConstantImage)
{
    dim dimin(5, 5);
    opticalflow::Image image(dimin);

    // Fill image with 10.
    for (std::size_t idx = 0; idx < image.get_size(); ++idx)
        image.set_val(10.0, idx);

    average_conv2d filter(dim(3, 3));

    filter.convolute(image);

    // A normalized averaging filter should preserve
    // a constant image, including at the boundaries.
    for (std::size_t idx = 0; idx < image.get_size(); ++idx)
        EXPECT_DOUBLE_EQ(image.get_val(idx), 10.0);
}


TEST(Conv2dIntegrationTest, GaussianFilterIsNormalized)
{
    dim dimin(5, 5);
    vector2d sigma(1.0, 1.0);

    gaussian_conv2d filter(dimin, sigma);

    double sum_i = 0.0;
    double sum_j = 0.0;

    for (std::size_t i = 0; i < dimin.x; ++i)
        sum_i += filter.get_weight_i(i);

    for (std::size_t j = 0; j < dimin.y; ++j)
        sum_j += filter.get_weight_j(j);

    EXPECT_NEAR(sum_i, 1.0, 1e-12);
    EXPECT_NEAR(sum_j, 1.0, 1e-12);

    // Gaussian should be symmetric.
    EXPECT_DOUBLE_EQ(
        filter.get_weight_i(0),
        filter.get_weight_i(4)
    );

    EXPECT_DOUBLE_EQ(
        filter.get_weight_i(1),
        filter.get_weight_i(3)
    );

    EXPECT_DOUBLE_EQ(
        filter.get_weight_j(0),
        filter.get_weight_j(4)
    );

    EXPECT_DOUBLE_EQ(
        filter.get_weight_j(1),
        filter.get_weight_j(3)
    );
}


TEST(Conv2dIntegrationTest, GaussianFilterPreservesConstantImage)
{
    dim dimin(7, 7);
    opticalflow::Image image(dimin);

    // Fill image with a constant.
    for (std::size_t idx = 0; idx < image.get_size(); ++idx)
        image.set_val(42.0, idx);

    gaussian_conv2d filter(
        dim(5, 5),
        vector2d(1.0, 1.0)
    );

    filter.convolute(image);

    // A normalized Gaussian filter should preserve
    // a constant image.
    for (std::size_t idx = 0; idx < image.get_size(); ++idx)
        EXPECT_NEAR(image.get_val(idx), 42.0, 1e-12);
}
