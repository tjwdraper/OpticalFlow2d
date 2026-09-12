#ifndef _FIELD_NEW_HPP_
#define _FIELD_NEW_HPP_

#include "include/coord2d.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace opticalflow {
    template <class T>
    class Field {
        public:
            // Constructors and deconstructors
            Field(dim dimin) : dimin(dimin), 
                            step(1, dimin.x), 
                            size(dimin.x*dimin.y),
                            field(new T[size]) {}

            Field(const Field<T>& fin) : dimin(fin.get_dimensions()), 
                                        step(fin.get_step()), 
                                        size(fin.get_size()),
                                        field(new T[fin.get_size()]) {
                std::copy(fin.get_field(), fin.get_field() + size, field);
            }
            Field(Field<T>&& other) noexcept : dimin(other.get_dimensions()),
                                            step(other.get_step()),
                                            size(other.get_size()),
                                            field(other.get_field()) {
                other.field = nullptr;
            }
            ~Field() { delete[] field; }

            // Getters and setters
            T* get_field() { return field; }
            const T* get_field() const { return field; }
            dim get_dimensions() const { return dimin; }
            dim get_step() const { return step; }
            std::size_t get_size() const { return size; }

            T get_val(std::size_t idx) const {
                Field::check_idx(idx);
                return field[idx];
            }

            T get_val(std::size_t i, std::size_t j) const {
                Field::check_idx(i, j);
                return field[i * step.x + j * step.y];
            }

            void set_val(T val, std::size_t idx) {
                Field::check_idx(idx);
                field[idx] = val;
            }

            void set_val(T val, std::size_t i, std::size_t j) {
                Field::check_idx(i, j);
                field[i * step.x + j * step.y] = val;
            }

            // Operator overloading
            Field<T>& operator=(const Field<T>& fin) {
                if (dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator=(const Field<T>&) dimensions of input and target do not match.");

                std::copy(fin.get_field(), fin.get_field() + size, field);
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
                if (dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator+=(const Field<T>&) dimensions of input and target do not match.");

                const T* finv = fin.get_field();
                for (std::size_t idx = 0; idx < size; ++idx) 
                    field[idx] += finv[idx];
                return *this;
            }

            Field<T>& operator-=(const Field<T>& fin) {
                if (dimin != fin.get_dimensions())
                    throw std::runtime_error("In Field<T>& operator-=(const Field<T>&) dimensions of input and target do not match.");

                const T* finv = fin.get_field();
                for (std::size_t idx = 0; idx < size; ++idx) 
                    field[idx] -= finv[idx];
                return *this;
            }

            Field<T> operator*(double val) const {
                Field<T> fout(*this);
                fout *= val;
                return fout;
            }

            Field<T>& operator*=(double val) {
                for (std::size_t idx = 0; idx < size; ++idx)
                    field[idx] *= val;
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

                for (std::size_t idx = 0; idx < size; ++idx)
                    field[idx] /= val;
                return *this;
            }

            friend Field<T> operator*(double val, const Field<T>& fin) {
                return fin * val;
            }

        private:
            void check_idx(std::size_t i, std::size_t j) const {
                if (i >= dimin.x || j >= dimin.y)
                    throw std::runtime_error("In T Field::get_val(std::size_t, std::size_t) input indices out of bound.");
            }

            void check_idx(std::size_t idx) const {
                if (idx >= size)
                    throw std::runtime_error("In T Field::get_val(std::size_t) input indices out of bound.");
            }

            const dim dimin;
            const dim step;
            const std::size_t size;
            T* field = nullptr;
    };

    using Image = Field<double>;
    using Motion = Field<vector2d>;

    namespace image {
        // TODO: change to mxArray* in future or separate from namespace
        void mex_load_image(const double* vals, Image& image) {
            std::copy(vals, vals + image.get_size(), image.get_field());
        }

        // TODO: change to mxArray* in future or separate from namespace
        void mex_save_image(double* vals, const Image& image) {
            std::copy(image.get_field(), image.get_field() + image.get_size(), vals);
        }

        double norm(const Image& image) {
            double norm(0.0);
            for (std::size_t idx = 0; idx < image.get_size(); ++idx) 
                norm += std::pow(image.get_val(idx), 2);
            return std::sqrt(norm);
        }

        double sum(const Image& image) {
            double sum(0.0);
            for (std::size_t idx = 0; idx < image.get_size(); ++idx)
                sum += image.get_val(idx);
            return sum;
        }

        double max(const Image& image) {
            double max(image.get_val(0));
            if (image.get_size() == 1)
                return max;

            for (std::size_t idx = 1; idx < image.get_size(); ++idx)
                if (image.get_val(idx) > max)
                    max = image.get_val(idx);
            return max;
        }

        double min(const Image& image) {
            double min(image.get_val(0));
            if (image.get_size() == 1)
                return min;

            for (std::size_t idx = 1; idx < image.get_size(); ++idx)
                if (image.get_val(idx) < min)
                    min = image.get_val(idx);
            return min;
        }

        void normalize(Image& image) {
            double low = opticalflow::image::min(image);
            double high = opticalflow::image::max(image);

            if (low == high) 
                throw std::runtime_error("In opticalflow::image::normalize(Image&) input.min() = input.max().");

            for (std::size_t idx = 0; idx < image.get_size(); ++idx) 
                image.set_val((image.get_val(idx)-low)/(high-low), idx);
        }
    }

    namespace motion {
        void mex_save_motion(double* vals, const Motion& motion) {
            std::size_t N = motion.get_size();
            for (std::size_t idx = 0; idx < N; ++idx) {
                const vector2d v = motion.get_val(idx);
                vals[idx + 0*N] = v.x;
                vals[idx + 1*N] = v.y;
            }
        }

        double norm(const Motion& motion) {
            double norm(0.0);
            for (std::size_t idx = 0; idx < motion.get_size(); ++idx) 
                norm += normsq(motion.get_val(idx));
            return std::sqrt(norm);
        }

        vector2d max(const Motion& motion) {
            vector2d max(motion.get_val(0));
            if (motion.get_size() == 1)
                return max;

            for (std::size_t idx = 1; idx < motion.get_size(); ++idx)
                if (motion.get_val(idx) > max)
                    max = motion.get_val(idx);
            return max;
        }

        vector2d min(const Motion& motion) {
            vector2d min(motion.get_val(0));
            if (motion.get_size() == 1)
                return min;

            for (std::size_t idx = 1; idx < motion.get_size(); ++idx)
                if (motion.get_val(idx) < min)
                    min = motion.get_val(idx);
            return min;
        }
    }
}

#endif