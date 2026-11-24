#ifndef _CONV_2D_HPP_
#define _CONV_2D_HPP_

#include "Field.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

///////////////////////////////////////////////////////////////////////////////////////////////////
// Separable 2d convolution
///////////////////////////////////////////////////////////////////////////////////////////////////
namespace opticalflow {
    class separable_conv2d {
        public:
            // Constructors and deconstructors
            separable_conv2d(const dim dimconv) : 
                _dimconv(dimconv),
                _stepconv(1, dimconv.x),
                _sizeconv(dimconv.x*dimconv.y),
                _weights_i(dimconv.x),
                _weights_j(dimconv.y) {

                // Check convolution dimensions
                if (dimconv.x == 0 || dimconv.y == 0)
                    throw std::runtime_error("In separable_conv2d(const dim), convolution dimensions must be non-zero.");

                if (dimconv.x % 2 == 0 || dimconv.y % 2 == 0)
                    throw std::runtime_error("In separable_conv2d(const dim), convolution dimensions must be odd.");
            }
            virtual ~separable_conv2d() = default;

            // Apply convolution
            template<typename T>
            void convolute(opticalflow::Field<T>& fin) const {
                const dim dimin = fin.get_dimensions();

                // Convolution centers
                const std::size_t center_i = _dimconv.x / 2;
                const std::size_t center_j = _dimconv.y / 2;

                // Apply convolution in y-direction
                opticalflow::Field<T> intermediate(dimin);

                for (std::size_t i = 0; i < dimin.x; ++i) {
                    for (std::size_t j = 0; j < dimin.y; ++j) {
                        T val{};

                        for (std::size_t kj = 0; kj < _dimconv.y; ++kj) {
                            long jj = static_cast<long>(j) + static_cast<long>(kj) - static_cast<long>(center_j);

                            // Clip to boundary
                            if (jj < 0) {jj = 0;}
                            if (jj > dimin.y-1) {jj = dimin.y-1;}

                            val += fin.at(i, static_cast<std::size_t>(jj)) * _weights_j[kj];
                        }

                        intermediate.set(val, i, j);
                    }
                }

                // Apply convolution in x-direction
                opticalflow::Field<T> result(dimin);
                
                for (std::size_t i = 0; i < dimin.x; ++i) {
                    for (std::size_t j = 0; j < dimin.y; ++j) {
                        T val{};

                        for (std::size_t ki = 0; ki < _dimconv.x; ++ki) {
                            long ii = static_cast<long>(i) + static_cast<long>(ki) - static_cast<long>(center_i);

                            // Clip to boundary
                            if (ii < 0) {ii = 0;}
                            if (ii > dimin.x-1) {ii = dimin.x-1;}

                            val += intermediate.at(static_cast<std::size_t>(ii), j) * _weights_i[ki];
                        }

                        result.set(val, i, j);
                    }
                }
                fin = std::move(result);
            }
            

        protected:
            void normalize_weights() {
                double sum_i = std::accumulate(_weights_i.begin(), _weights_i.end(), 0.0);
                for (auto& elem : _weights_i)
                    elem /= sum_i;
                
                double sum_j = std::accumulate(_weights_j.begin(), _weights_j.end(), 0.0);
                for (auto& elem : _weights_j)
                    elem /= sum_j;
            }

            dim _dimconv;
            dim _stepconv;
            std::size_t _sizeconv;

            std::vector<double> _weights_i;
            std::vector<double> _weights_j;
    };

    class average_conv2d : public separable_conv2d {
        public:
            average_conv2d(const dim dimconv) : separable_conv2d(dimconv) {
                std::fill(_weights_i.begin(), _weights_i.end(), 1.0);
                std::fill(_weights_j.begin(), _weights_j.end(), 1.0);

                normalize_weights();
            }
            ~average_conv2d() = default;
    };

    class gaussian_conv2d : public separable_conv2d {
        public:
            // Constructors and deconstructors
            gaussian_conv2d(const dim dimconv, const vector2d sigma) : separable_conv2d(dimconv), _sigma(sigma) {
                if (sigma.x <= 0.0 || sigma.y <= 0.0)
                    throw std::runtime_error("In gaussian_conv2d(const dim, const vector2d), sigma must be positive.");

                _weights_i = gaussian_weights(dimconv.x, sigma.x);
                _weights_j = gaussian_weights(dimconv.y, sigma.y);

                normalize_weights();
            }
            gaussian_conv2d(const vector2d sigma) : gaussian_conv2d(dim(2*static_cast<std::size_t>(std::ceil(3.0 * sigma.x)) + 1), sigma) {}
            virtual ~gaussian_conv2d() = default;

            //  Getters and setters
            vector2d get_sigma() const {return _sigma;}

        private:
            std::vector<double> gaussian_weights(std::size_t n, double sigma) {
                std::vector<double> weights(n);

                const double center = static_cast<double>(n-1) / 2.0;

                for (std::size_t i = 0; i < n; ++i) {
                    double x = static_cast<double>(i) - center;
                    weights[i] = std::exp(-x*x / (2.0 * sigma * sigma));
                }

                return weights;
            }

            vector2d _sigma;
    };
}

#endif
