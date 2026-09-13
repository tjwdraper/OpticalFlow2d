#include <gtest/gtest.h>
#include <include/Field.hpp>

class ImageTest : public testing::Test {
protected:
    opticalflow::Image image{dim(3, 2)};

    void SetUp() override {
        image.set_val(1.0, 0, 0);
        image.set_val(2.0, 1, 0);
        image.set_val(3.0, 2, 0);

        image.set_val(4.0, 0, 1);
        image.set_val(5.0, 1, 1);
        image.set_val(6.0, 2, 1);
    }
};


TEST_F(ImageTest, Sum)
{
    EXPECT_DOUBLE_EQ(opticalflow::image::sum(image), 21.0);
}


TEST_F(ImageTest, Norm)
{
    // sqrt(1^2 + 2^2 + ... + 6^2) = sqrt(91)
    EXPECT_DOUBLE_EQ(
        opticalflow::image::norm(image),
        std::sqrt(91.0)
    );
}


TEST_F(ImageTest, Max)
{
    EXPECT_DOUBLE_EQ(opticalflow::image::max(image), 6.0);
}


TEST_F(ImageTest, Min)
{
    EXPECT_DOUBLE_EQ(opticalflow::image::min(image), 1.0);
}


TEST_F(ImageTest, MaxSingleValue)
{
    opticalflow::Image single(dim(1, 1));
    single.set_val(42.0, 0);

    EXPECT_DOUBLE_EQ(opticalflow::image::max(single), 42.0);
}


TEST_F(ImageTest, MinSingleValue)
{
    opticalflow::Image single(dim(1, 1));
    single.set_val(42.0, 0);

    EXPECT_DOUBLE_EQ(opticalflow::image::min(single), 42.0);
}


TEST_F(ImageTest, Normalize)
{
    opticalflow::image::normalize(image);

    // Original range [1, 6] should become [0, 1].
    EXPECT_DOUBLE_EQ(image.get_val(0), 0.0);
    EXPECT_DOUBLE_EQ(image.get_val(1), 0.2);
    EXPECT_DOUBLE_EQ(image.get_val(2), 0.4);
    EXPECT_DOUBLE_EQ(image.get_val(3), 0.6);
    EXPECT_DOUBLE_EQ(image.get_val(4), 0.8);
    EXPECT_DOUBLE_EQ(image.get_val(5), 1.0);
}


TEST_F(ImageTest, NormalizeProducesRangeZeroToOne)
{
    opticalflow::image::normalize(image);

    EXPECT_DOUBLE_EQ(opticalflow::image::min(image), 0.0);
    EXPECT_DOUBLE_EQ(opticalflow::image::max(image), 1.0);
}


TEST_F(ImageTest, NormalizeConstantImageThrows)
{
    opticalflow::Image constant(dim(3, 2));

    for (std::size_t idx = 0; idx < constant.get_size(); ++idx)
        constant.set_val(5.0, idx);

    EXPECT_THROW(
        opticalflow::image::normalize(constant),
        std::runtime_error
    );
}


TEST_F(ImageTest, NormalizeConstantImageRemainsUnchanged)
{
    opticalflow::Image constant(dim(3, 2));

    for (std::size_t idx = 0; idx < constant.get_size(); ++idx)
        constant.set_val(5.0, idx);

    EXPECT_THROW(
        opticalflow::image::normalize(constant),
        std::runtime_error
    );

    for (std::size_t idx = 0; idx < constant.get_size(); ++idx)
        EXPECT_DOUBLE_EQ(constant.get_val(idx), 5.0);
}

TEST_F(ImageTest, MexLoadImage)
{
    const double vals[] = {
        10.0, 20.0, 30.0,
        40.0, 50.0, 60.0
    };

    opticalflow::Image loaded(dim(3, 2));

    opticalflow::image::mex_load_image(vals, loaded);

    for (std::size_t idx = 0; idx < loaded.get_size(); ++idx)
        EXPECT_DOUBLE_EQ(loaded.get_val(idx), vals[idx]);
}


TEST_F(ImageTest, MexSaveImage)
{
    double vals[6] = {};

    opticalflow::image::mex_save_image(vals, image);

    for (std::size_t idx = 0; idx < image.get_size(); ++idx)
        EXPECT_DOUBLE_EQ(vals[idx], image.get_val(idx));
}


TEST_F(ImageTest, MexLoadAndSaveImage)
{
    const double input[] = {
        10.0, 20.0, 30.0,
        40.0, 50.0, 60.0
    };

    double output[6] = {};

    opticalflow::Image loaded(dim(3, 2));

    opticalflow::image::mex_load_image(input, loaded);
    opticalflow::image::mex_save_image(output, loaded);

    for (std::size_t idx = 0; idx < 6; ++idx)
        EXPECT_DOUBLE_EQ(output[idx], input[idx]);
}

