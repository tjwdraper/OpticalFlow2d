#include <gtest/gtest.h>
#include "Field.hpp"

class FieldTest : public testing::Test {
    protected:
        FieldTest() {}
        ~FieldTest() override {}

        void SetUp() override {
            field.set(1.0, 0, 0);
            field.set(2.0, 1, 0);
            field.set(3.0, 2, 0);

            field.set(4.0, 0, 1);
            field.set(5.0, 1, 1);
            field.set(6.0, 2, 1);
        }
        void TearDown() override {}

    protected:
        opticalflow::Image field{dim(3, 2)};
};

TEST_F(FieldTest, get_dimensions) {
    // Arange
    dim expected_dims(3, 2);
    dim expected_step(1, 3);
    std::size_t expected_size(6);

    // Act

    // Assert
    ASSERT_EQ(field.get_dimensions(), expected_dims);
    ASSERT_EQ(field.get_step(), expected_step);
    ASSERT_EQ(field.get_size(), expected_size);
}


TEST_F(FieldTest, constructor) {
    // Arange
    opticalflow::Image fcopy(field);

    // Act

    // Assert
    ASSERT_EQ(fcopy.get_dimensions(), field.get_dimensions());
    ASSERT_EQ(fcopy.get_size(), field.get_size());
    for (std::size_t idx = 0; idx < fcopy.get_size(); ++idx)
        ASSERT_EQ(fcopy.at(idx), field.at(idx));
}

TEST_F(FieldTest, move) {
    // Arange
    opticalflow::Image moved(std::move(field));

    // Act

    // Assert
    for (std::size_t idx = 0; idx < moved.get_size(); ++idx)
        ASSERT_DOUBLE_EQ(moved.at(idx), static_cast<double>(idx+1));
    
    moved.set(100,0);
    ASSERT_DOUBLE_EQ(moved.at(0), 100.0);    
}

TEST_F(FieldTest, at) {
    // Arange
    double expected(4.0);

    // Act

    // Assert
    ASSERT_DOUBLE_EQ(field.at(0,1), expected);
}

TEST_F(FieldTest, at_linear_idx) {
    // Arange
    double expected(4.0);

    // Act

    // Assert
    ASSERT_DOUBLE_EQ(field.at(3), expected);
}

TEST_F(FieldTest, set) {
    // Arange
    double expected(10.0);

    // Act
    field.set(10.0,0,1);

    // Assert
    ASSERT_DOUBLE_EQ(field.at(0,1), expected);
}

TEST_F(FieldTest, set_linear_idx) {
    // Arange
    double expected(10.0);

    // Act
    field.set(10.0,3);

    // Assert
    ASSERT_DOUBLE_EQ(field.at(3), expected);
}

// Operator overloading
TEST_F(FieldTest, assignment) {
    // Arange
    opticalflow::Image other(dim(3, 2));
    for (std::size_t idx = 0; idx < other.get_size(); ++idx) 
        other.set(static_cast<double>(10+idx), idx);

    // Act
    field = other;

    // Assert
    ASSERT_EQ(field.get_dimensions(), other.get_dimensions());
    for (std::size_t idx = 0; idx < field.get_size(); ++idx) {
        EXPECT_DOUBLE_EQ(field.at(idx), other.at(idx));
    }
}

TEST_F(FieldTest, assignment_self) {
    // Arange

    // Act
    field = field;

    // Assert
    for (std::size_t idx = 0; idx < field.get_size(); ++idx)
        EXPECT_DOUBLE_EQ(field.at(idx), static_cast<double>(idx+1));
}

TEST_F(FieldTest, addition)
{
    // Arrange
    opticalflow::Image other(dim(3, 2));
    for (std::size_t idx = 0; idx < other.get_size(); ++idx)
        other.set(10.0, idx);

    // Act
    opticalflow::Image result = field + other;

    // Assert
    EXPECT_DOUBLE_EQ(result.at(0), 11.0);
    EXPECT_DOUBLE_EQ(result.at(1), 12.0);
    EXPECT_DOUBLE_EQ(result.at(2), 13.0);
    EXPECT_DOUBLE_EQ(result.at(3), 14.0);
    EXPECT_DOUBLE_EQ(result.at(4), 15.0);
    EXPECT_DOUBLE_EQ(result.at(5), 16.0);

    // Original field should be unchanged.
    EXPECT_DOUBLE_EQ(field.at(0), 1.0);
    EXPECT_DOUBLE_EQ(field.at(5), 6.0);

    // Other should be unchanged.
    EXPECT_DOUBLE_EQ(other.at(0), 10.0);
}

TEST_F(FieldTest, addition_assignment)
{
    // Arrange
    opticalflow::Image other(dim(3, 2));
    for (std::size_t idx = 0; idx < other.get_size(); ++idx)
        other.set(10.0, idx);

    // Act
    field += other;

    // Assert
    EXPECT_DOUBLE_EQ(field.at(0), 11.0);
    EXPECT_DOUBLE_EQ(field.at(1), 12.0);
    EXPECT_DOUBLE_EQ(field.at(2), 13.0);
    EXPECT_DOUBLE_EQ(field.at(3), 14.0);
    EXPECT_DOUBLE_EQ(field.at(4), 15.0);
    EXPECT_DOUBLE_EQ(field.at(5), 16.0);
}

TEST_F(FieldTest, subtraction)
{
    // Arrange
    opticalflow::Image other(dim(3, 2));
    for (std::size_t idx = 0; idx < other.get_size(); ++idx)
        other.set(10.0, idx);

    // Act
    opticalflow::Image result = field - other;

    // Assert
    EXPECT_DOUBLE_EQ(result.at(0), -9.0);
    EXPECT_DOUBLE_EQ(result.at(1), -8.0);
    EXPECT_DOUBLE_EQ(result.at(2), -7.0);
    EXPECT_DOUBLE_EQ(result.at(3), -6.0);
    EXPECT_DOUBLE_EQ(result.at(4), -5.0);
    EXPECT_DOUBLE_EQ(result.at(5), -4.0);

    // Original unchanged.
    EXPECT_DOUBLE_EQ(field.at(0), 1.0);
    EXPECT_DOUBLE_EQ(field.at(5), 6.0);
}

TEST_F(FieldTest, subtraction_assignment)
{
    // Arange
    opticalflow::Image other(dim(3, 2));
    for (std::size_t idx = 0; idx < other.get_size(); ++idx)
        other.set(10.0, idx);

    // Act
    field -= other;

    // Assert
    EXPECT_DOUBLE_EQ(field.at(0), -9.0);
    EXPECT_DOUBLE_EQ(field.at(5), -4.0);
}

TEST_F(FieldTest, multiplication)
{
    // Arange
    double s(2.0);
    // Act
    opticalflow::Image result = field * s;

    // Assert
    EXPECT_DOUBLE_EQ(result.at(0), 2.0);
    EXPECT_DOUBLE_EQ(result.at(1), 4.0);
    EXPECT_DOUBLE_EQ(result.at(2), 6.0);
    EXPECT_DOUBLE_EQ(result.at(3), 8.0);
    EXPECT_DOUBLE_EQ(result.at(4), 10.0);
    EXPECT_DOUBLE_EQ(result.at(5), 12.0);

    // Original unchanged.
    EXPECT_DOUBLE_EQ(field.at(0), 1.0);
    EXPECT_DOUBLE_EQ(field.at(5), 6.0);
}

TEST_F(FieldTest, multiplication_assignment)
{
    // Arange
    double s(2.0);

    // Act
    field *= s;

    // Assert
    EXPECT_DOUBLE_EQ(field.at(0), 2.0);
    EXPECT_DOUBLE_EQ(field.at(1), 4.0);
    EXPECT_DOUBLE_EQ(field.at(2), 6.0);
    EXPECT_DOUBLE_EQ(field.at(3), 8.0);
    EXPECT_DOUBLE_EQ(field.at(4), 10.0);
    EXPECT_DOUBLE_EQ(field.at(5), 12.0);
}

TEST_F(FieldTest, scalar_multiplication)
{
    // Arange
    double s(2.0);

    opticalflow::Image result1 = field * s;
    opticalflow::Image result2 = s * field;

    for (std::size_t idx = 0; idx < field.get_size(); ++idx) {
        EXPECT_DOUBLE_EQ(result1.at(idx), static_cast<double>(2.0 * (idx + 1)));
        EXPECT_DOUBLE_EQ(result2.at(idx), static_cast<double>(2.0 * (idx + 1)));
    }
}

TEST_F(FieldTest, division)
{
    // Arange
    double s(2.0);

    // Act
    opticalflow::Image result = field / s;

    // Assert
    EXPECT_DOUBLE_EQ(result.at(0), 0.5);
    EXPECT_DOUBLE_EQ(result.at(1), 1.0);
    EXPECT_DOUBLE_EQ(result.at(2), 1.5);
    EXPECT_DOUBLE_EQ(result.at(3), 2.0);
    EXPECT_DOUBLE_EQ(result.at(4), 2.5);
    EXPECT_DOUBLE_EQ(result.at(5), 3.0);

    // Original unchanged.
    EXPECT_DOUBLE_EQ(field.at(0), 1.0);
}

TEST_F(FieldTest, division_assignment)
{
    // Arange
    double s(2.0);

    // Act
    field /= s;

    // Assert
    EXPECT_DOUBLE_EQ(field.at(0), 0.5);
    EXPECT_DOUBLE_EQ(field.at(1), 1.0);
    EXPECT_DOUBLE_EQ(field.at(2), 1.5);
    EXPECT_DOUBLE_EQ(field.at(3), 2.0);
    EXPECT_DOUBLE_EQ(field.at(4), 2.5);    
    EXPECT_DOUBLE_EQ(field.at(5), 3.0);
}



// Exception handling
TEST_F(FieldTest, assignment_different_dimensions) {
    // Arange
    opticalflow::Image other(dim(2, 2));

    // Act

    // Assert
    EXPECT_THROW(field = other, std::runtime_error);
}

TEST_F(FieldTest, addition_different_dimensions_throws)
{
    // Arange
    opticalflow::Image other(dim(2, 2));

    // Act

    // Assert
    EXPECT_THROW(field + other, std::runtime_error);
}

TEST_F(FieldTest, addition_assignment_different_dimensions_throws)
{
    opticalflow::Image other(dim(2, 2));

    EXPECT_THROW(field += other, std::runtime_error);
}

TEST_F(FieldTest, subtraction_different_dimensions_throws)
{
    opticalflow::Image other(dim(2, 2));

    EXPECT_THROW(field - other, std::runtime_error);
}

TEST_F(FieldTest, division_by_zero_throws)
{
    EXPECT_THROW(field / 0.0, std::runtime_error);
}

TEST_F(FieldTest, division_assignment_by_zero_throws)
{
    EXPECT_THROW(field /= 0.0, std::runtime_error);
}
