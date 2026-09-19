#include <gtest/gtest.h>

#include "Field.hpp"
#include "gradients.hpp"

TEST(GradientsTest, PartialX)
{
    const dim d(5, 5);
    opticalflow::Image image(d);

    for (std::size_t i = 0; i < d.x; ++i) {
        for (std::size_t j = 0; j < d.y; ++j) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            image.set_val(x * x + y * y, i, j);
        }
    }

    // Boundary: forward/backward difference
    EXPECT_DOUBLE_EQ(1.0, gradients::partial_x<double>(image, 0, 2));
    EXPECT_DOUBLE_EQ(7.0, gradients::partial_x<double>(image, 4, 2));

    // Interior: centered difference
    EXPECT_DOUBLE_EQ(2.0, gradients::partial_x<double>(image, 1, 2));
    EXPECT_DOUBLE_EQ(4.0, gradients::partial_x<double>(image, 2, 2));
    EXPECT_DOUBLE_EQ(6.0, gradients::partial_x<double>(image, 3, 2));
}


TEST(GradientsTest, PartialY)
{
    const dim d(5, 5);
    opticalflow::Image image(d);

    for (std::size_t i = 0; i < d.x; ++i) {
        for (std::size_t j = 0; j < d.y; ++j) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            image.set_val(x * x + y * y, i, j);
        }
    }

    EXPECT_DOUBLE_EQ(1.0, gradients::partial_y<double>(image, 2, 0));
    EXPECT_DOUBLE_EQ(7.0, gradients::partial_y<double>(image, 2, 4));

    EXPECT_DOUBLE_EQ(2.0, gradients::partial_y<double>(image, 2, 1));
    EXPECT_DOUBLE_EQ(4.0, gradients::partial_y<double>(image, 2, 2));
    EXPECT_DOUBLE_EQ(6.0, gradients::partial_y<double>(image, 2, 3));
}


TEST(GradientsTest, PartialXX)
{
    const dim d(5, 5);
    opticalflow::Image image(d);

    for (std::size_t i = 0; i < d.x; ++i) {
        for (std::size_t j = 0; j < d.y; ++j) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            image.set_val(x * x + y * y, i, j);
        }
    }

    for (std::size_t i = 0; i < d.x; ++i) {
        EXPECT_DOUBLE_EQ(
            2.0,
            gradients::partial_xx<double>(image, i, 2)
        );
    }
}


TEST(GradientsTest, PartialYY)
{
    const dim d(5, 5);
    opticalflow::Image image(d);

    for (std::size_t i = 0; i < d.x; ++i) {
        for (std::size_t j = 0; j < d.y; ++j) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            image.set_val(x * x + y * y, i, j);
        }
    }

    for (std::size_t j = 0; j < d.y; ++j) {
        EXPECT_DOUBLE_EQ(
            2.0,
            gradients::partial_yy<double>(image, 2, j)
        );
    }
}


TEST(GradientsTest, PartialXY)
{
    const dim d(5, 5);
    opticalflow::Image image(d);

    for (std::size_t i = 0; i < d.x; ++i) {
        for (std::size_t j = 0; j < d.y; ++j) {
            const double x = static_cast<double>(i);
            const double y = static_cast<double>(j);
            image.set_val(x * x + y * y, i, j);
        }
    }

    // f_xy = 0 in the interior
    for (std::size_t i = 1; i < d.x - 1; ++i) {
        for (std::size_t j = 1; j < d.y - 1; ++j) {
            EXPECT_DOUBLE_EQ(
                0.0,
                gradients::partial_xy<double>(image, i, j)
            );
        }
    }

    // Your chosen boundary convention
    for (std::size_t i = 0; i < d.x; ++i) {
        EXPECT_DOUBLE_EQ(0.0, gradients::partial_xy<double>(image, i, 0));
        EXPECT_DOUBLE_EQ(0.0, gradients::partial_xy<double>(image, i, d.y - 1));
    }

    for (std::size_t j = 0; j < d.y; ++j) {
        EXPECT_DOUBLE_EQ(0.0, gradients::partial_xy<double>(image, 0, j));
        EXPECT_DOUBLE_EQ(0.0, gradients::partial_xy<double>(image, d.x - 1, j));
    }
}