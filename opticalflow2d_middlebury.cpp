#define cimg_display 0 // Remove if making use of plot functions from cimg_library. If so, add -lX11 to compilation flags.

#include "include/CImg.h"

#include "include/coord2d.hpp"
#include "include/Field.hpp"
#include "include/interp2d.hpp"
#include "include/ImageRegistration.h"

#include <chrono>

void convert_cimg_to_opticalflow(opticalflow::Image& image, cimg_library::CImg<double>& cimage) {
    const dim dimin(cimage.width(), cimage.height());
    const std::size_t size = dimin.x * dimin.y;

    if (image.get_dimensions() != dimin)
        throw std::runtime_error("In convert_cimg_to_opticalflow(Image&, cimg_library::CImg<double>&), dimensions of input and target have to equal");

    // Convert cimage to grayscale, raw image. Average over 3 color channels
    double* image_gs = new double[size];
    double* cimage_rgb = cimage.data();

    for (std::size_t idx = 0; idx < size; ++idx) {
        image_gs[idx] = (cimage_rgb[idx] + cimage_rgb[idx + size] + cimage_rgb[idx + 2*size]) / 3.0;
    }

    // Set data from opticalflow::Image target to raw data values
    opticalflow::image::mex_load_image(image_gs, image);

    // Normalize intensities between zero and one
    opticalflow::image::normalize(image);

    // Free memory
    delete[] image_gs;
}

int main() {
    // Load images
    std::cout << "Loading images...";
    cimg_library::CImg<double> Iref_rgb("img/other-color-twoframes/other-data/RubberWhale/frame10.png");
    cimg_library::CImg<double> Imov_rgb("img/other-color-twoframes/other-data/RubberWhale/frame11.png");
    std::cout << "Images loaded: " << Iref_rgb.width() << "x" << Iref_rgb.height() << std::endl;

    // Convert to opticalflow type
    std::cout << "Convert to opticalflow type...";
    const dim dimin(Iref_rgb.width(), Iref_rgb.height());

    opticalflow::Image Iref(dimin);
    opticalflow::Image Imov(dimin);

    convert_cimg_to_opticalflow(Iref, Iref_rgb);
    convert_cimg_to_opticalflow(Imov, Imov_rgb);
    std::cout << "Complete!" << std::endl;

    // Initialize registration class
    std::cout << "Initialize ImageRegistration class...";
    std::size_t nscales = 3;
    std::size_t nrefine = 2;
    std::size_t niter[4] = {200, 200, 200, 200};
    double alpha = 0.4;
    double eps = 1e-4;

    ImageRegistration myImageRegistration(dimin, nscales, niter, alpha, eps, nrefine);
    std::cout << "Complete!" << std::endl;

    // Set images
    std::cout << "Set reference and moving image to ImageRegistration class...";
    myImageRegistration.set_reference_image(Iref);
    myImageRegistration.set_moving_image(Imov);
    std::cout << "Complete!" << std::endl;

    // Register
    std::cout << "Estimate optical flow...";
    
    auto start = std::chrono::steady_clock::now();
    myImageRegistration.estimate_optical_flow();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Completed in " << elapsed.count() << "s" << std::endl;

    // Get the motion field
    const opticalflow::Motion& motion = myImageRegistration.get_estimated_motion();

    // Get the registered image
    opticalflow::Image Ireg(dimin);
    interp2d::warp2d(Ireg, Imov, motion);

    // Report on MSE
    double mse_initial = opticalflow::image::mse(Iref, Imov);
    double mse_final = opticalflow::image::mse(Iref, Ireg);

    std::cout << "MSE (initial): " << mse_initial << std::endl;
    std::cout << "MSE (final): " << mse_final << std::endl;
}