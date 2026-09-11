#include <gtest/gtest.h>
#include <include/coord2d.h>

TEST(coord2dTest, TestEqual) {
    // Arrange
    coord2d a(2.0, 3.0);

    // Act
    coord2d c = a;

    // Assert
    ASSERT_DOUBLE_EQ(c.x, 2.0);
    ASSERT_DOUBLE_EQ(c.y, 3.0);
}

TEST(coord2dTest, TestAddition) {
    // Arrange
    coord2d a(2.0, 3.0);
    coord2d b(4.0, 5.0);

    // Act 
    coord2d c = a + b;

    // ASSERT
    ASSERT_DOUBLE_EQ(c.x, 6.0);
    ASSERT_DOUBLE_EQ(c.y, 8.0);
}

TEST(coord2dTest, TestAdditionScalar) {
    // Arrange
    coord2d a(2.0, 3.0);
    double b(5.0);

    // Act
    coord2d c = a + b;

    // Assert
    ASSERT_DOUBLE_EQ(c.x, 7.0);
    ASSERT_DOUBLE_EQ(c.y, 8.0);
}

TEST(coord2dTest, TestSubtraction) {
    // Arrange
    coord2d a(2.0, 3.0);
    coord2d b(4.0, 6.0);

    // Act 
    coord2d c = a - b;

    // ASSERT
    ASSERT_DOUBLE_EQ(c.x, -2.0);
    ASSERT_DOUBLE_EQ(c.y, -3.0);
}

TEST(coord2dTest, TestSubtractionScalar) {
    // Arange
    coord2d a(2.0, 3.0);
    double b(5.0);

    // Act
    coord2d c = a - b;

    // Assert
    ASSERT_DOUBLE_EQ(c.x, -3.0);
    ASSERT_DOUBLE_EQ(c.y, -2.0);
}

TEST(coord2dTest, TestAdditionEq) {
    // Arange
    coord2d a(2.0, 3.0);
    coord2d b(4.0, 5.0);

    // Act
    a += b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, 6.0);
    ASSERT_DOUBLE_EQ(a.y, 8.0);
}

TEST(coord2dTest, TestAdditionEqScalar) {
    // Arange
    coord2d a(2.0, 3.0);
    double b(5.0);

    // Act
    a += b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, 7.0);
    ASSERT_DOUBLE_EQ(a.y, 8.0);
}

TEST(coord2dTest, TestSubtractionEq) {
    // Arange
    coord2d a(2.0, 3.0);
    coord2d b(4.0, 5.0);

    // Act
    a -= b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, -2.0);
    ASSERT_DOUBLE_EQ(a.y, -2.0);
}

TEST(coord2dTest, TestSubtractionEqScalar) {
    // Arange
    coord2d a(2.0, 3.0);
    double b(5.0);

    // Act
    a -= b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, -3.0);
    ASSERT_DOUBLE_EQ(a.y, -2.0);
}

TEST(coord2dTest, TestMultiplicationScalar) {
    // Arange
    coord2d a(2.0, 3.0);
    double b = 3.5;

    // Act
    coord2d c = b * a;

    // Assert
    ASSERT_DOUBLE_EQ(c.x, 7.0);
    ASSERT_DOUBLE_EQ(c.y, 10.5);
}

TEST(coord2dTest, TestMultiplicationEqScalar) {
    // Arange
    coord2d a(2.0, 3.0);
    double b = 3.5;

    // Act
    a *= b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, 7.0);
    ASSERT_DOUBLE_EQ(a.y, 10.5);
}

TEST(coord2dTest, TestDivisionScalar) {
    // Arange
    coord2d a(4.0, 6.0);
    double b(2.0);

    // Act
    coord2d c = a / b;

    // Assert
    ASSERT_DOUBLE_EQ(c.x, 2.0);
    ASSERT_DOUBLE_EQ(c.y, 3.0);
}

TEST(coord2dTest, TestDivisionEqScalar) {
    // Arange
    coord2d a(4.0, 6.0);
    double b(2.0);

    // Act
    a /= b;

    // Assert
    ASSERT_DOUBLE_EQ(a.x, 2.0);
    ASSERT_DOUBLE_EQ(a.y, 3.0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    RUN_ALL_TESTS();
}