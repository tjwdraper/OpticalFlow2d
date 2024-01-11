#ifndef _FIELD_H_
#define _FIELD_H_

#include <src/coord2d.h>
#include <src/Kernel.h>

template <class T>
class Field {
    public:
        // Constructors and deconstructors
        Field(const dim dimin);
        Field (const Field& fieldin);
        ~Field();

        // Getters and setters
        dim get_dimensions() const;
        dim get_step() const;
        unsigned int get_size() const;

        // Upsample and downsample the field
        void upSample(const Field<T>& fieldin);
        void downSample(const Field<T>& fieldin);

        // Convolute with kernel
        void convolute(const Kernel& kernel);

        // Overload operators
        virtual Field<T> operator+(const Field<T>& fieldin) const;
        virtual Field<T>& operator+=(const Field<T>& fieldin);
        virtual Field<T> operator-(const Field<T>& fieldin) const;
        virtual Field<T>& operator-=(const Field<T>& fieldin);

        virtual Field<T>& operator*=(const float& val);

    protected:
        // Getters and setters
        T* get_field() const;

        T *field;
        dim dimin;
        unsigned int sizein;
        dim step;
};

#include <src/Field.tpp>

#endif