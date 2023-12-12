#ifndef _FIELD_H_
#define _FIELD_H_

#include <coord2d.h>

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

        // Overload operator
        Field<T>& operator=(const Field& fieldin);

    protected:
        // Getters and setters
        T* get_field() const;

        T *field;
        dim dimin;
        unsigned int sizein;
        dim step;
};

#include <Field.tpp>

#endif