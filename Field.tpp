#include <iostream>
#include <math.h>
#include <cstring>

// Constructors and deconstructors
template <class T>
Field<T>::Field(const dim dimin) {
    // Set the dimensions of the field
    this->dimin = dimin;
    this->sizein = dimin.x * dimin.y;
    this->step = dim(1, dimin.x);

    // Allocate memory for the field
    this->field = new T[this->sizein];

    // Set values to zero
    memset(this->field, 0, this->sizein*sizeof(T));
}

template <class T>
Field<T>::Field(const Field& fieldin) {
    // Set the dimensions of the field
    this->dimin = dimin;
    this->sizein = dimin.x * dimin.y;
    this->step = dim(1, dimin.x);

    // Allocate memory for the field
    this->field = new T[this->sizein];

    // Copy contents from the input field to the object
    memcpy(this->field, fieldin.get_field(), this->sizein*sizeof(T));
}

template <class T>
Field<T>::~Field() {
    // Free up the memory
    delete[] this->field;
}

// Getters and setters
template <class T>
T* Field<T>::get_field() const {
    return this->field;
}

template <class T>
dim Field<T>::get_dimensions() const {
    return this->dimin;
}

template <class T>
dim Field<T>::get_step() const {
    return this->step;
}

template <class T>
unsigned int Field<T>::get_size() const {
    return this->sizein;
}

// Upsample and downsample
template <class T>
void Field<T>::downSample(const Field<T>& fieldin) {
    // Get the size of the input field
    const unsigned int& sizein = fieldin.get_size();

    // Get the dimensions and step size of the input and output iamge
    const dim& dimin = fieldin.get_dimensions();
    const dim& dimout = this->dimin;
    const dim& stepin = fieldin.get_step();
    const dim& stepout = this->step;

    // Get a copy to the data of the fields
    T* datain = fieldin.get_field();
    T* dataout = this->field;

    // Get the factor between input and output iomage
    dim factor(dimin.x/dimout.x, dimin.y/dimout.y);

    // Index in the input and output image
    unsigned int idxin;
    unsigned int idxout;

    // Iterate over voxels
    for (int i = 0; i < dimout.x; i++) {
        for (int j = 0; j < dimout.y; j++) {

            // Compute the index in input and output image
            idxout = i * stepout.x + j * stepout.y;
            idxin = i * factor.x * stepin.x + j * factor.y * stepin.y;

            // Iterate over voxels in patch of the input image
            T val(0.0f);
            int p = 0;

            for (int ii = 0; ii < factor.x; ii++) {
                for (int jj = 0; jj < factor.y; jj++) {
                    if (idxin + ii * stepin.x + jj * stepin.y >= sizein) {
                        continue;
                    }
                    val += datain[idxin + ii * stepin.x + jj * stepin.y];
                    p++;
                }
            }

            // Set the new value
            if (p != 0) {
                dataout[idxout] = val / (float) p;
            }
            
        }
    }

    // Done
    return;
}

template <class T>
void Field<T>::upSample(const Field<T>& fieldin) {
    // Get the size of the input field
    const unsigned int& sizein = fieldin.get_size();

    // Get the dimensions and step size of the input and output iamge
    const dim& dimin = fieldin.get_dimensions();
    const dim& dimout = this->dimin;
    const dim& stepin = fieldin.get_step();
    const dim& stepout = this->step;

    // Get a copy to the data of the fields
    T* datain = fieldin.get_field();
    T* dataout = this->field;

    //
    unsigned int idx;
    for (unsigned int i = 0; i < dimout.x; i++) {
        for (unsigned int j = 0; j < dimout.y; j++) {
            idx = i * stepout.x + j * stepout.y;

            // Get the corresponding coordinate in the original grid
            float px = (float) i * dimin.x / (float) dimout.x; int dx = std::floor(px); float fx = px - dx;
            float py = (float) j * dimin.y / (float) dimout.y; int dy = std::floor(py); float fy = py - dy;

            // Check that the indices are within the image bounds
            unsigned int idxO = dx * stepin.x + dy * stepin.y;
            if (idxO >= sizein) {
                continue;
            } 

            // Linear interpolation
            T val = datain[idxO]*(1-fx)*(1-fy);
            float weight = (1-fx)*(1-fy);
            if (dx < dimin.x-1) {
                val += datain[idxO + stepin.x]*fx*(1-fy);
                weight += fx*(1-fy);
            }
            if (dy < dimin.y-1) {
                val += datain[idxO + stepin.y]*(1-fx)*fy;
                weight += (1-fx)*fy;
            }
            if ((dx < dimin.x-1) && (dy < dimin.y-1)) {
                val += datain[idxO + stepin.x + stepin.y]*fx*fy;
                weight += fx*fy;
            }

            // Check if weight is non-zero
            if (weight != 0) {
                dataout[idx] = val / weight;
            }
        }
    }

    // Done
    return;
}

// Overload operators
template <class T>
Field<T>& Field<T>::operator=(const Field<T>& fieldin) {
    if (this->dimin != fieldin.get_dimensions()) {
        std::cout << "Error: in Field<T>::operator=(const Field<T>& )," 
                      "assignment cannot be done because dimensions of "
                      "input and output object are not the same." << std::endl;
    }

    // Copy the contents
    memcpy(this->field, fieldin.get_field(), this->sizein*sizeof(T));
    
    // Done
    return *this;
}

