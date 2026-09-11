#ifndef _FIELD_NEW_HPP_
#define _FIELD_NEW_HPP_

#include "include/coord2d.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

template <class T>
class Field {
    public:
        // Constructors and deconstructors
        Field(dim dimin) : dimin(dimin), 
                           step(1, dimin.x), 
                           size(dimin.x*dimin.y),
                           field(new T[dimin.x*dimin.y]) {}

        Field(const Field<T>& fin) : dimin(fin.get_dimensions()), 
                                     step(fin.get_step()), 
                                     size(fin.get_size()),
                                     field(new T[size]) {
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

        T* field = nullptr;
        const dim dimin;
        const dim step;
        const std::size_t size;
};

typedef Field<double> Image;
typedef Field<vector2d> Motion;

// #include <src/Field.tpp>

#endif