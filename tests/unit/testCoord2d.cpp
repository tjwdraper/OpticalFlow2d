#include <gtest/gtest.h>
#include <include/coord2d.h>

class Coord2dTest : public testing::Test {
    protected:
        Coord2dTest() {}
        ~Coord2dTest() override {}

        void SetUp() override { res = coord2d<double>(); }
        void TearDown() override {}

    protected:
        coord2d<double> a{1.2, 5.9};
        coord2d<double> b{-0.7, 2.3};
        double s{2.0};
        coord2d<double> res;
};

TEST_F(Coord2dTest, equal) {
    // Arange
    coord2d<double> expected(1.2, 5.9);

    // Act
    res = a;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, addition) {
    // Arange
    coord2d<double> expected(0.5, 8.2);

    // Act
    res = a + b;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, addition_scalar) {
    // Arange
    coord2d<double> expected(3.2, 7.9);

    // Act
    res = s + a;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, subtraction) {
    // Arange
    coord2d<double> expected(1.9, 3.6);

    // Act
    res = a - b;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, subtraction_scalar) {
    // Arange
    coord2d<double> expected(-0.8, 3.9);

    // Act
    res = a - s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, addition_eq) {
    // Arange
    coord2d<double> expected(0.5, 8.2);

    // Act
    res = a;
    res += b;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, addition_eq_scalar) {
    // Arange
    coord2d<double> expected(3.2, 7.9);

    // Act
    res = a;
    res += s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, subtraction_eq) {
    // Arange
    coord2d<double> expected(1.9, 3.6);

    // Act
    res = a;
    res -= b;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, subtraction_eq_scalar) {
    // Arange
    coord2d<double> expected(-0.8, 3.9);

    // Act
    res = a;
    res -= s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, multiplication_scalar) {
    // Arange
    coord2d<double> expected(2.4, 11.8);

    // Act
    res = s * a;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, multiplication_eq_scalar) {
    // Arange
    coord2d<double> expected(2.4, 11.8);

    // Act
    res = a;
    res *= s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, division_scalar) {
    // Arange
    coord2d<double> expected(0.6, 2.95);

    // Act
    res = a / s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

TEST_F(Coord2dTest, division_eq_scalar) {
    // Arange
    coord2d<double> expected(0.6, 2.95);

    // Act
    res = a;
    res /= s;

    // Assert
    ASSERT_DOUBLE_EQ(res.x, expected.x);
    ASSERT_DOUBLE_EQ(res.y, expected.y);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
}