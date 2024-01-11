#include <iostream>
#include <cstring>
#include <stdexcept>
#include <math.h>
#include <mex.h>

// Constructors and deconstructors
template <class T>
Field<T>::Field(const dim dimin) {
    // Set the dimensions of the field
    this->dimin = dimin;
    this->sizein = dimin.x * dimin.y;
    this->step = dim(1, dimin.x);

    // Allocate memory for the field and set values to zero
    try {
        this->field = new T[this->sizein];
        memset(this->field, 0, this->sizein*sizeof(T));
    }
    catch (const std::bad_alloc& e) {
        const std::string mes = "Error in Field<T>::Field(const dim dimin): " + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
    }
}

template <class T>
Field<T>::Field(const Field& fieldin) {
    // Set the dimensions of the field
    this->dimin  = fieldin.get_dimensions();
    this->sizein = fieldin.get_size();
    this->step   = fieldin.get_step();

    // Allocate memory for the field and copy values from input
    try {
        this->field = new T[this->sizein];
        memcpy(this->field, fieldin.get_field(), this->sizein*sizeof(T));
    }
    catch (const std::bad_alloc& e) {
        const std::string mes = "Error in Field<T>::Field(const Field& fieldin): " + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
    }
}

template <class T>
Field<T>::~Field() {
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

    // Check that input field has larger input dimensions
    if ((dimout.x > dimin.x) || (dimout.y > dimin.y)) {
        throw std::invalid_argument("Error in Field<T>::downSample(const FIeld<T>& fieldin): input has to have same dimensions as target");
    }

    // Get a copy to the data of the fields
    T* datain = fieldin.get_field();
    T* dataout = this->field;

    // Get the factor between input and output iomage
    dim factor;
    try {
        factor = dimin / dimout;
    }
    catch (const std::runtime_error& e) {
        const std::string mes = std::string("Error in Field<T>::downSample(const Field<T>& fieldin): ") + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
    }

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

    // Check that input field has larger input dimensions
    if ((dimout.x < dimin.x) || (dimout.y < dimin.y)) {
        throw std::invalid_argument("Error in Field<T>::downSample(const FIeld<T>& fieldin): input has to have same dimensions as target");
    }

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

// Convolute with kernel
template<class T>
void Field<T>::convolute(const Kernel& kernel) {
    // Get the dimensions of the field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get the dimensions of the kernel
    const dim& dimkernel = kernel.get_dimensions();
    const dim& stepkernel = kernel.get_step();

    const int cx = (dimkernel.x - 1) / 2;
    const int cy = (dimkernel.y - 1) / 2;

    // Get a copy to the pointer of the kernel data
    const double* k = kernel.get_kernel();

    // Make a copy of the input field
    Field<T> tmp(*this);

    // Get a copy to the pointer to the motion data
    T *field    = this->get_field();
    T *fieldtmp = tmp.get_field();

    // Iterate over voxels
    int idx, idxkernel;
    for (int i = 0; i < dimin.x; i++) {
        for (int j = 0; j < dimin.y; j++) {
            // Get absolute index
            idx = i * step.x + j * step.y;

            // Iterate over kernel
            T val;
            double weight = 0.0f;
            for (int ii = -cx; ii <= cx; ii++) {
                for (int jj = -cy; jj <= cy; jj++) {
                    // Check if we are within bounds
                    if ((i + ii) * step.x + (j + jj) * step.y < 0 ||
                        (i + ii) * step.x + (j + jj) * step.y >= sizein) {
                        continue;
                    } 
                    else {
                        // Get absolute index in the kernel
                        idxkernel = (ii + cx) * stepkernel.x + (jj + cy) * stepkernel.y;

                        // Add contribution of the kernel element
                        val += fieldtmp[idx + ii * step.x + jj * step.y] * k[idxkernel];
                        weight += k[idxkernel];
                    }
                }
            }

            // Set value in motion field
            if (weight != 0) {
                field[idx] = val / weight;
            }
        }
    }

    // Done
    return;
}

// Overload operators
template <class T>
Field<T>& Field<T>::operator+=(const Field<T>& fieldin) {
    if (this->dimin != fieldin.get_dimensions()) {
        throw std::invalid_argument("input argument has to have same dimensions as target");
    }
    // Get a copy to the pointer to the content of input
    T *datain = fieldin.get_field();

    // Iterate over voxels, add together
    for (unsigned int i = 0 ; i < this->sizein; i++) {
        this->field[i] += datain[i];
    }

    // Done
    return *this;
}

template <class T>
Field<T> Field<T>::operator+(const Field<T>& fieldin) const {
    if (this->dimin != fieldin.get_dimensions()) {
        throw std::invalid_argument("input argument has to have same dimensions as target");
    }
    // Create output field as copy of input
    Field<T> fieldout(*this);

    // Add input to it
    fieldout += fieldin;

    // Done
    return fieldout;
}

template <class T>
Field<T>& Field<T>::operator-=(const Field<T>& fieldin) {
    if (this->dimin != fieldin.get_dimensions()) {
        throw std::invalid_argument("input argument has to have same dimensions as target");
    }
    // Get a copy to the pointer to the content of input
    T *datain = fieldin.get_field();

    // Iterate over voxels, add together
    for (unsigned int i = 0; i < this->sizein; i++) {
        this->field[i] -= datain[i];
    }

    // Done
    return *this;
}

template <class T>
Field<T> Field<T>::operator-(const Field<T>& fieldin) const {
    if (this->dimin != fieldin.get_dimensions()) {
        throw std::invalid_argument("input argument has to have same dimensions as target");
    }
    // Create output field as copy of input
    Field<T> fieldout(*this);

    // Add input to it
    fieldout -= fieldin;

    // Done
    return fieldout;
}

template <class T>
Field<T>& Field<T>::operator*=(const float& val) {
    // Iterate over voxels, add together
    for (unsigned int i = 0; i < this->sizein; i++) {
        this->field[i] *= val;
    }

    // Done
    return *this;
}