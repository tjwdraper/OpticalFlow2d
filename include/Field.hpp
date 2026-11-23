#ifndef _FIELD_NEW_HPP_
#define _FIELD_NEW_HPP_

#include "coord2d.hpp"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <numeric>

namespace opticalflow {
    template <class T>
    class Field {
        public:
            // Constructors and deconstructors
            explicit Field(dim dimin) : _dimin(dimin), 
                                        _step(1, dimin.x), 
                                        _size(dimin.x*dimin.y),
                                        _field(_size) {}
                                        
            ~Field() = default;

            T* data() noexcept { return _field.data(); }
            const T* data() const noexcept { return _field.data(); }

            dim get_dimensions() const { return _dimin; }
            dim get_step() const { return _step; }
            std::size_t get_size() const { return _size; }



            const T& at(std::size_t idx) const {
                return _field.at(idx); // .at check bounds automatically
            }

            const T& at(std::size_t i, std::size_t j) const {
                return _field.at(index(i,j));
            }

            void set_val(const T& val, std::size_t idx) {
                _field.at(idx) = val;
            }

            void set_val(const T& val, std::size_t i, std::size_t j) {
                _field.at(index(i,j)) = val;
            }

            void fill(T val) {
                _field.assign(_size, val);
            }

            // Operator overloading
            Field<T>& operator=(const Field<T>& fin) {
                if (_dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator=(const Field<T>&) dimensions of input and target do not match.");

                std::copy(fin.data(), fin.data() + _size, _field.data());
                return *this;
            }

            Field<T> operator+(const Field<T>& fin) const {
                Field<T> fout(*this);
                fout += fin;
                return fout;
            }

            Field<T> operator-(const Field<T>& fin) const {
                Field<T> fout(*this);
                fout -= fin;
                return fout;
            }

            Field<T>& operator+=(const Field<T>& fin) {
                if (_dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator+=(const Field<T>&) dimensions of input and target do not match.");

                for (std::size_t idx = 0; idx < _size; ++idx) 
                    _field[idx] += fin.at(idx);
                return *this;
            }

            Field<T>& operator-=(const Field<T>& fin) {
                if (_dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator-=(const Field<T>&) dimensions of input and target do not match.");

                for (std::size_t idx = 0; idx < _size; ++idx) 
                    _field[idx] -= fin.at(idx);
                return *this;
            }

            Field<T> operator*(double val) const {
                Field<T> fout(*this);
                fout *= val;
                return fout;
            }

            Field<T>& operator*=(double val) {
                for (std::size_t idx = 0; idx < _size; ++idx)
                    _field[idx] *= val;
                return *this;
            }

            Field<T> operator/(double val) const {
                if (val == 0.0)
                    throw std::runtime_error("In Field<T> operator/(double ) const, division by zero.");

                Field<T> fout(*this);
                fout /= val;
                return fout;
            }

            Field<T>& operator/=(double val) {
                if (val == 0.0)
                    throw std::runtime_error("In Field<T>& operator/=(double ), division by zero.");

                // _field /= val;
                for (std::size_t idx = 0; idx < _size; ++idx)
                    _field[idx] /= val;
                return *this;
            }

            friend Field<T> operator*(double val, const Field<T>& fin) {
                return fin * val;
            }

        private:
            std::size_t index(std::size_t i, std::size_t j) const {
                if (i >= _dimin.x || j >= _dimin.y)
                    throw std::out_of_range("Field index out of bounds.");

                return i * _step.x + j * _step.y;
            }

            dim _dimin;
            dim _step;
            std::size_t _size;
            std::vector<T> _field;
    };

    using Image = Field<double>;
    using Motion = Field<vector2d>;

    namespace image {
        // TODO: change to mxArray* in future or separate from namespace
        inline void load_image(const double* vals, Image& image) {
            std::copy(vals, vals + image.get_size(), image.data());
        }

        // TODO: change to mxArray* in future or separate from namespace
        inline void save_image(double* vals, const Image& image) {
            std::copy(image.data(), image.data() + image.get_size(), vals);
            // std::copy(image.get_field(), image.get_field() + image.get_size(), vals);
        }

        inline double norm(const Image& image) {
            return std::sqrt(std::inner_product(image.data(), image.data() + image.get_size(), image.data(), 0.0));
            // double norm(0.0);
            // for (std::size_t idx = 0; idx < image.get_size(); ++idx) 
            //     norm += std::pow(image.at(idx), 2);
            // return std::sqrt(norm);
        }

        inline double sum(const Image& image) {
            return std::accumulate(image.data(), image.data() + image.get_size(), 0.0);
            // double sum(0.0);
            // for (std::size_t idx = 0; idx < image.get_size(); ++idx)
            //     sum += image.at(idx);
            // return sum;
        }

        inline double max(const Image& image) {
            return *std::max_element(image.data(), image.data() + image.get_size());
            // double max(image.at(0));
            // if (image.get_size() == 1)
            //     return max;

            // for (std::size_t idx = 1; idx < image.get_size(); ++idx)
            //     if (image.at(idx) > max)
            //         max = image.at(idx);
            // return max;
        }

        inline double min(const Image& image) {
            return *std::min_element(image.data(), image.data() + image.get_size());
            // double min(image.at(0));
            // if (image.get_size() == 1)
            //     return min;

            // for (std::size_t idx = 1; idx < image.get_size(); ++idx)
            //     if (image.at(idx) < min)
            //         min = image.at(idx);
            // return min;
        }

        inline void normalize(Image& image) {
            double low = opticalflow::image::min(image);
            double high = opticalflow::image::max(image);

            if (low == high) 
                throw std::runtime_error("In opticalflow::image::normalize(Image&) input.min() = input.max().");

            for (std::size_t idx = 0; idx < image.get_size(); ++idx) 
                image.set_val((image.at(idx)-low)/(high-low), idx);
        }

        inline double mse(const Image& image_a, const Image& image_b) {
            if (image_a.get_dimensions() != image_b.get_dimensions())
                throw std::runtime_error("In opticalflow::image::mse(const Image&, const Image&), input images don't share the same dimensions.");

            double val(0.0);
            for (std::size_t idx = 0; idx < image_a.get_size(); ++idx) 
                val += std::pow(image_a.at(idx) - image_b.at(idx), 2.0);
            return val / image_a.get_size();
        }
    }

    namespace motion {
        inline void save_motion(double* vals, const Motion& motion) {
            std::size_t N = motion.get_size();
            for (std::size_t idx = 0; idx < N; ++idx) {
                const vector2d v = motion.at(idx);
                vals[idx + 0*N] = v.x;
                vals[idx + 1*N] = v.y;
            }
        }

        inline double norm(const Motion& motion) {
            double norm(0.0);
            for (std::size_t idx = 0; idx < motion.get_size(); ++idx) 
                norm += normsq(motion.at(idx));
            return std::sqrt(norm);
        }

        inline vector2d max(const Motion& motion) {
            return *std::max_element(motion.data(), motion.data() + motion.get_size());

            // vector2d max(motion.at(0));
            // if (motion.get_size() == 1)
            //     return max;

            // for (std::size_t idx = 1; idx < motion.get_size(); ++idx)
            //     if (motion.at(idx) > max)
            //         max = motion.at(idx);
            // return max;
        }

        inline vector2d min(const Motion& motion) {
            return *std::min_element(motion.data(), motion.data() + motion.get_size());
            // vector2d min(motion.at(0));
            // if (motion.get_size() == 1)
            //     return min;

            // for (std::size_t idx = 1; idx < motion.get_size(); ++idx)
            //     if (motion.at(idx) < min)
            //         min = motion.at(idx);
            // return min;
        }
    }
}

#endif