#ifndef FIELD_HPP
#define FIELD_HPP

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

            // Getters and setters
            T* data() noexcept { return _field.data(); }
            const T* data() const noexcept { return _field.data(); }

            typename std::vector<T>::iterator begin() noexcept { return _field.begin(); } // typename tells the compiler that std::vector<T>::iterator is a type, not a static member
            typename std::vector<T>::iterator end() noexcept { return _field.end(); }
            typename std::vector<T>::const_iterator begin() const noexcept { return _field.begin(); }
            typename std::vector<T>::const_iterator end() const noexcept { return _field.end(); }

            dim get_dimensions() const { return _dimin; }
            dim get_step() const { return _step; }
            std::size_t get_size() const { return _size; }

            const T& at(std::size_t idx) const {
                return _field.at(idx); // .at check bounds automatically
            }

            const T& at(std::size_t i, std::size_t j) const {
                return _field.at(index(i,j));
            }

            void set(const T& val, std::size_t idx) {
                _field.at(idx) = val;
            }

            void set(const T& val, std::size_t i, std::size_t j) {
                _field.at(index(i,j)) = val;
            }

            void fill(T val) {
                _field.assign(_size, val);
            }

            // Operator overloading
            Field<T>& operator=(const Field<T>& fin) {
                if (_dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator=(const Field<T>&) dimensions of input and target do not match.");

                std::copy(fin.begin(), fin.end(), _field.data());
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
                for (auto& elem : _field)
                    elem *= val;
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

                for (auto& elem : _field)
                    elem /= val;
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
            std::copy(image.begin(), image.end(), vals);
        }

        inline double norm(const Image& image) {
            return std::sqrt(std::inner_product(image.begin(), image.end(), image.begin(), 0.0));
        }

        inline double sum(const Image& image) {
            return std::accumulate(image.begin(), image.end(), 0.0);
        }

        inline double max(const Image& image) {
            return *std::max_element(image.begin(), image.end());
        }

        inline double min(const Image& image) {
            return *std::min_element(image.begin(), image.end());
        }

        inline void normalize(Image& image) {
            double low = opticalflow::image::min(image);
            double high = opticalflow::image::max(image);

            if (low == high) 
                throw std::runtime_error("In opticalflow::image::normalize(Image&) input.min() = input.max().");

            for (std::size_t idx = 0; idx < image.get_size(); ++idx) 
                image.set((image.at(idx)-low)/(high-low), idx);
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
            return *std::max_element(motion.begin(), motion.end());
        }

        inline vector2d min(const Motion& motion) {
            return *std::min_element(motion.begin(), motion.end());
        }
    }
}

#endif