#include <gtest/gtest.h>
#include <include/Field_new.hpp>

class FieldTest : public testing::Test {
    protected:
        FieldTest() {}
        ~FieldTest() override {}

        void SetUp() override {
            field.set_val(1.0, 0, 0);
            field.set_val(2.0, 1, 0);
            field.set_val(3.0, 2, 0);

            field.set_val(4.0, 0, 1);
            field.set_val(5.0, 1, 1);
            field.set_val(6.0, 2, 1);
        }
        void TearDown() override {}

    protected:
        Field<double> field{dim(3,2)};
};

TEST_F(FieldTest, dimension_check) {
    // Arange
    dim expected(3,2);

    // Act

    // Assert
    ASSERT_EQ(field.get_dimensions(), expected);
}