#define cimg_display 0 // Remove if making use of plot functions from cimg_library. If so, add -lX11 to compilation flags.
#include "CImg.h"

#include "coord2d.hpp"
#include "Field.hpp"
#include "interp2d.hpp"
#include "ImageRegistration.h"
#include "json.hpp"

#include <fstream>
#include <string>
#include <stdexcept>
#include <chrono>
#include <vector>

///////////////////////////////////////////////////////////////////////////////////////////////////
// Read json configuration file and store in structure
///////////////////////////////////////////////////////////////////////////////////////////////////
struct json_config {
    // Paths to input images
    std::string path_reference_image;
    std::string path_moving_image;

    // Registration parameters with default values
    ModelOption option = ModelOption::HORN_SCHUNCK;

    std::size_t nrefine = 0;
    std::vector<std::size_t> niter = {100, 100, 100, 100};
    std::size_t nscales = 3;

    double alpha = 0.1;
    double beta = 5.0;
    double eps = 1e-4;
    double resampling_factor = 0.5;
};

json_config load_config(const std::string& filename) {
    // Open .json configuration file
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Could not open json configuration file: " + filename);
    nlohmann::json json;
    file >> json;

    // Create json_config structure
    json_config config;

    // Set the image paths
    config.path_reference_image = json.at("reference_image").get<std::string>();
    config.path_moving_image = json.at("moving_image").get<std::string>();

    // Set json_config variables from .json fields
    std::string str_option = json.at("optical_flow_option").get<std::string>();
    auto it = mapper_model_option.find(str_option);
    if (it != mapper_model_option.end())
        config.option = it->second;

    if (json.contains("registration")) {
        const auto& registration = json.at("registration");

        if (registration.contains("nrefine"))
            config.nrefine = registration.at("nrefine").get<std::size_t>();

        if (registration.contains("niter")) {
            config.niter = registration.at("niter").get<std::vector<std::size_t>>();
            if (config.niter.empty())
                throw std::runtime_error("niter must contain at least one value.");
            config.nscales = config.niter.size() - 1;
        }

        if (registration.contains("alpha"))
            config.alpha = registration.at("alpha").get<double>();
        
        if (registration.contains("beta"))
            config.beta = registration.at("beta").get<double>();
        
        if (registration.contains("eps"))
            config.eps = registration.at("eps").get<double>();

        if (registration.contains("resampling_factor"))
            config.resampling_factor = registration.at("resampling_factor").get<double>();
    }

    return config;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Read images using the cimg_library and convert to opticalflow::Image type from Field.hpp,
// which is the input for this ImageRegistration class implementation
///////////////////////////////////////////////////////////////////////////////////////////////////
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
    opticalflow::image::load_image(image_gs, image);

    // Normalize intensities between zero and one
    opticalflow::image::normalize(image);

    // Free memory
    delete[] image_gs;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " config.json\n";
        return 1;
    }

    json_config config = load_config(argv[1]);

    // Load images
    std::cout << "Loading images...";
    cimg_library::CImg<double> Iref_rgb(config.path_reference_image.c_str());
    cimg_library::CImg<double> Imov_rgb(config.path_moving_image.c_str());
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
    ImageRegistration myImageRegistration(dimin, config.option, config.nscales, config.niter.data(), config.alpha, config.beta, config.eps, config.nrefine, config.resampling_factor);
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