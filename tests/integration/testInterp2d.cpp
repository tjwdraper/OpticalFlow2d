#include "include/interp2d.hpp"

#include <gtest/gtest.h>

#include <cmath>


TEST(Interp2DTest, BilinearInterpolation) {
    opticalflow::Image image(dim(3, 3));

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

    // Centre of the four pixels:
    //
    // 5 6
    // 8 9
    //
    // Expected = (5 + 6 + 8 + 9) / 4 = 7
    EXPECT_DOUBLE_EQ(
        interp2d::interpolate_at_value<double>(
            image,
            vector2d(1.5, 1.5),
            image.get_dimensions()),
        7.0
    );

    // Exact pixel location should return the pixel value.
    EXPECT_DOUBLE_EQ(
        interp2d::interpolate_at_value<double>(
            image,
            vector2d(1.0, 2.0),
            image.get_dimensions()),
        6.0
    );
}


TEST(Interp2DTest, BilinearInterpolationAtBoundary) {
    opticalflow::Image image(dim(3, 3));

    // 1 2 3
    // 4 5 6
    // 7 8 9
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            image.set_val(
                static_cast<double>(1 + i * 3 + j),
                i, j);

    // With clamped boundaries:
    //
    // (-0.5,-0.5) is equivalent to (0,0)
    //
    // Expected = 1
    EXPECT_DOUBLE_EQ(
        interp2d::interpolate_at_value<double>(
            image,
            vector2d(-0.5, -0.5),
            image.get_dimensions()),
        1.0
    );

    // Likewise beyond the upper boundary.
    EXPECT_DOUBLE_EQ(
        interp2d::interpolate_at_value<double>(
            image,
            vector2d(2.5, 2.5),
            image.get_dimensions()),
        9.0
    );
}


TEST(Interp2DTest, WarpWithZeroMotionPreservesImage) {
    opticalflow::Image image_in(dim(3, 3));
    opticalflow::Image image_out(dim(3, 3));
    opticalflow::Motion motion(dim(3, 3));

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            image_in.set_val(
                static_cast<double>(1 + i * 3 + j),
                i, j);

            motion.set_val(vector2d(0.0, 0.0), i, j);
        }
    }

    interp2d::warp2d(image_out, image_in, motion);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(
                image_out.get_val(i, j),
                image_in.get_val(i, j));
        }
    }
}


TEST(Interp2DTest, WarpWithConstantTranslation) {
    opticalflow::Image image_in(dim(3, 3));
    opticalflow::Image image_out(dim(3, 3));
    opticalflow::Motion motion(dim(3, 3));

    // 1 2 3
    // 4 5 6
    // 7 8 9
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            image_in.set_val(
                static_cast<double>(1 + i * 3 + j),
                i, j);

            // Sample one pixel to the right.
            motion.set_val(vector2d(0.0, 1.0), i, j);
        }
    }

    interp2d::warp2d(image_out, image_in, motion);

    // Because the warp samples:
    //
    // image_out(i,j) = image_in(i,j+1)
    //
    // and the boundary is clamped:
    //
    // 2 3 3
    // 5 6 6
    // 8 9 9
    EXPECT_DOUBLE_EQ(image_out.get_val(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(0, 2), 3.0);

    EXPECT_DOUBLE_EQ(image_out.get_val(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(1, 1), 6.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(1, 2), 6.0);

    EXPECT_DOUBLE_EQ(image_out.get_val(2, 0), 8.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(2, 1), 9.0);
    EXPECT_DOUBLE_EQ(image_out.get_val(2, 2), 9.0);
}


TEST(Interp2DTest, AccumulateZeroMotion) {
    opticalflow::Motion motion(dim(3, 3));
    opticalflow::Motion motion_interp(dim(3, 3));
    opticalflow::Motion motion_acc(dim(3, 3));

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            motion.set_val(vector2d(
                static_cast<double>(i),
                static_cast<double>(j)), i, j);

            motion_interp.set_val(
                vector2d(0.0, 0.0), i, j);
        }
    }

    interp2d::accumulate(motion_acc, motion, motion_interp);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            EXPECT_DOUBLE_EQ(
                motion_acc.get_val(i, j).x,
                motion.get_val(i, j).x);

            EXPECT_DOUBLE_EQ(
                motion_acc.get_val(i, j).y,
                motion.get_val(i, j).y);
        }
    }
}


TEST(Interp2DTest, AccumulateConstantTranslations) {
    opticalflow::Motion motion(dim(3, 3));
    opticalflow::Motion motion_interp(dim(3, 3));
    opticalflow::Motion motion_acc(dim(3, 3));

    // u(x) = (1, 2)
    // v(x) = (3, 4)
    //
    // u(x + v(x)) = (1, 2)
    //
    // Therefore:
    //
    // u_acc(x) = u(x + v(x)) + v(x)
    //           = (4, 6)

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            motion.set_val(
                vector2d(1.0, 2.0), i, j);

            motion_interp.set_val(
                vector2d(3.0, 4.0), i, j);
        }
    }

    interp2d::accumulate(
        motion_acc,
        motion,
        motion_interp);

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            const vector2d val = motion_acc.get_val(i, j);

            EXPECT_DOUBLE_EQ(val.x, 4.0);
            EXPECT_DOUBLE_EQ(val.y, 6.0);
        }
    }
}

TEST(Interp2DTest, ResizeImageUp) {
    opticalflow::Image image_in(dim(2, 2));
    opticalflow::Image image_out(dim(4, 4));

    // 1 2
    // 3 4
    image_in.set_val(1.0, 0, 0);
    image_in.set_val(2.0, 0, 1);
    image_in.set_val(3.0, 1, 0);
    image_in.set_val(4.0, 1, 1);

    interp2d::resize(image_out, image_in);

    // Check the corners.
    //
    // The resize uses pixel-center coordinates and
    // clamped bilinear interpolation.
    EXPECT_NEAR(image_out.get_val(0, 0), 1.0, 1e-12);
    EXPECT_NEAR(image_out.get_val(0, 3), 2.0, 1e-12);
    EXPECT_NEAR(image_out.get_val(3, 0), 3.0, 1e-12);
    EXPECT_NEAR(image_out.get_val(3, 3), 4.0, 1e-12);

    // Interior values should be interpolated.
    EXPECT_NEAR(image_out.get_val(1, 1), 1.75, 1e-12);
    EXPECT_NEAR(image_out.get_val(1, 2), 2.25, 1e-12);
    EXPECT_NEAR(image_out.get_val(2, 1), 2.75, 1e-12);
    EXPECT_NEAR(image_out.get_val(2, 2), 3.25, 1e-12);
}


TEST(Interp2DTest, ResizeImageConstantPreservesValue) {
    opticalflow::Image image_in(dim(3, 3));
    opticalflow::Image image_out(dim(7, 5));

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            image_in.set_val(42.0, i, j);

    interp2d::resize(image_out, image_in);

    for (std::size_t i = 0; i < 7; ++i)
        for (std::size_t j = 0; j < 5; ++j)
            EXPECT_DOUBLE_EQ(
                image_out.get_val(i, j),
                42.0);
}


TEST(Interp2DTest, ResizeMotionScalesDisplacement) {
    opticalflow::Motion motion_in(dim(2, 2));
    opticalflow::Motion motion_out(dim(4, 4));

    // Constant displacement of (2, 4) pixels
    // in the original image.
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            motion_in.set_val(
                vector2d(2.0, 4.0), i, j);

    interp2d::resize(motion_out, motion_in);

    // 2x upscaling means the displacement should
    // also become twice as large.
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            const vector2d val = motion_out.get_val(i, j);

            EXPECT_NEAR(val.x, 4.0, 1e-12);
            EXPECT_NEAR(val.y, 8.0, 1e-12);
        }
    }
}


TEST(Interp2DTest, ResizeZeroMotionPreservesZero) {
    opticalflow::Motion motion_in(dim(3, 3));
    opticalflow::Motion motion_out(dim(6, 6));

    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            motion_in.set_val(
                vector2d(0.0, 0.0), i, j);

    interp2d::resize(motion_out, motion_in);

    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t j = 0; j < 6; ++j) {
            const vector2d val = motion_out.get_val(i, j);

            EXPECT_DOUBLE_EQ(val.x, 0.0);
            EXPECT_DOUBLE_EQ(val.y, 0.0);
        }
    }
}