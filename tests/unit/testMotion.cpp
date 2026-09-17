#include <gtest/gtest.h>
#include <include/Field.hpp>

class MotionTest : public testing::Test {
    protected:
    opticalflow::Motion motion{dim(3, 2)};

    void SetUp() override {
        motion.set_val(vector2d(1.0, 2.0), 0, 0);
        motion.set_val(vector2d(2.0, 3.0), 1, 0);
        motion.set_val(vector2d(3.0, 4.0), 2, 0);

        motion.set_val(vector2d(4.0, 5.0), 0, 1);
        motion.set_val(vector2d(5.0, 6.0), 1, 1);
        motion.set_val(vector2d(6.0, 7.0), 2, 1);
    }
};

TEST_F(MotionTest, Norm) {
    EXPECT_DOUBLE_EQ(opticalflow::motion::norm(motion), std::sqrt(230.0));
}

TEST_F(MotionTest, Max) {
    const vector2d result = opticalflow::motion::max(motion);
    EXPECT_DOUBLE_EQ(result.x, 6.0);
    EXPECT_DOUBLE_EQ(result.y, 7.0);
}

TEST_F(MotionTest, Min) {
    const vector2d result = opticalflow::motion::min(motion);
    EXPECT_DOUBLE_EQ(result.x, 1.0);
    EXPECT_DOUBLE_EQ(result.y, 2.0);
}

TEST_F(MotionTest, MaxSingleValue) {
    opticalflow::Motion single(dim(1, 1));
    single.set_val(vector2d(42.0, 24.0), 0);

    const vector2d result = opticalflow::motion::max(single);

    EXPECT_DOUBLE_EQ(result.x, 42.0);
    EXPECT_DOUBLE_EQ(result.y, 24.0);
}

TEST_F(MotionTest, MinSingleValue) {
    opticalflow::Motion single(dim(1, 1));
    single.set_val(vector2d(42.0, 24.0), 0);

    const vector2d result = opticalflow::motion::min(single);

    EXPECT_DOUBLE_EQ(result.x, 42.0);
    EXPECT_DOUBLE_EQ(result.y, 24.0);
}

TEST_F(MotionTest, MexSaveMotion) {
    double vals[12] = {};

    opticalflow::motion::save_motion(vals, motion);

    // save_motion stores all x components first,
    // followed by all y components.
    //
    // vals:
    // [x0, x1, x2, x3, x4, x5,
    //  y0, y1, y2, y3, y4, y5]

    const double expected[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,

        2.0, 3.0, 4.0,
        5.0, 6.0, 7.0
    };

    for (std::size_t idx = 0; idx < 12; ++idx)
        EXPECT_DOUBLE_EQ(vals[idx], expected[idx]);
}

TEST_F(MotionTest, MexSaveMotionSingleValue) {
    opticalflow::Motion single(dim(1, 1));
    single.set_val(vector2d(42.0, 24.0), 0);

    double vals[2] = {};

    opticalflow::motion::save_motion(vals, single);

    EXPECT_DOUBLE_EQ(vals[0], 42.0);
    EXPECT_DOUBLE_EQ(vals[1], 24.0);
}

TEST_F(MotionTest, MexSaveMotionPreservesComponents) {
    double vals[12] = {};

    opticalflow::motion::save_motion(vals, motion);

    const std::size_t N = motion.get_size();

    for (std::size_t idx = 0; idx < N; ++idx) {
        const vector2d value = motion.get_val(idx);

        EXPECT_DOUBLE_EQ(vals[idx], value.x);
        EXPECT_DOUBLE_EQ(vals[idx + N], value.y);
    }
}
