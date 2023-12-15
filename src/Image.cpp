#include <src/Image.h>

#include <mex.h>
#include <math.h>

// Constructors and deconstructors
Image::Image(const dim dimin) : Field<float>(dimin) {}

Image::Image(const Image& im) : Field<float>(im) {}

Image::~Image() {}

// Getters and setters
void Image::set_image(const double* im) {
    // Get the dimensions of the image
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Store the input in the object
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            this->field[i*step.x + j*step.y] = (float) im[i*step.x + j*step.y];
        }
    }

    // Done
    return;
}

float* Image::get_image() const {
    return this->Field<float>::get_field();
}

// Copy content to input
void Image::copy_image_to_input(double* im) const {
    // Get the dimensions of the image
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Store the input in the object
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            im[i*step.x + j*step.y] = (double) this->field[i*step.x + j*step.y];
        }
    }

    // Done
    return;
}

// Upsample and downsample
void Image::upSample(const Image& im) {
    this->Field<float>::upSample(im);
}

void Image::downSample(const Image& im) {
    this->Field<float>::downSample(im);
}

// Get some information from the image
float Image::sum() const {
    float sum = 0.0f;
    for (int i = 0; i < this->sizein; i++) {
        sum += this->field[i];
    }
    return sum;
}

float Image::max() const {
    float max = 0.0f;
    for (int i = 0; i < this->sizein; i++) {
        if (this->field[i] > max) {
            max = this->field[i];
        }
    }
    return max;
}

float Image::min() const {
    float min = 0.0f;
    for (int i = 0; i < this->sizein; i++) {
        if (this->field[i] < min) {
            min = this->field[i];
        }
    }
    return min;
}

// Normalize the image
void Image::normalize() {
    float max = this->Image::max();
    float min = this->Image::min();

    for (int i = 0; i < this->sizein; i++) {
        this->field[i] = (this->field[i] - min) / (max - min);
    }

    return;
}

// Warp image with motion field
void Image::warp2d(const Motion& mo) {
    // Copy the image to a temporary array
    float *Itmp = new float[this->sizein];
    memcpy(Itmp, this->field, this->sizein*sizeof(float));

    // Get a copy of the pointer to the contents of the input motion
    vector2d *motion = mo.get_motion();

    // Iterate over voxels
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    int idx;
    for (int i = 0; i < dimin.x; i++) {
        for (int j = 0; j < dimin.y; j++) {

            // Get index in image
            idx = i * step.x + j * step.y;
            
            // Get the warped index
            float px = i + motion[idx].x; int dx = std::floor(px); float fx = px - dx;
            float py = j + motion[idx].y; int dy = std::floor(py); float fy = py - dy;

            // Check if warped index is out of bounds
            if ((dx < 0) || (dx >= dimin.x) ||
                (dy < 0) || (dy >= dimin.y)) {
                continue;
            }
            
            // Otherwise, get the value through linear interpolation
            int idxO = dx * step.x + dy * step.y;
            float val = Itmp[idxO]*(1-fx)*(1-fy);
            float weight = (1-fx)*(1-fy);
            if (dx < dimin.x-1) {
                val += Itmp[idxO + step.x]*fx*(1-fy);
                weight += fx*(1-fy);
            }
            if (dy < dimin.y-1) {
                val += Itmp[idxO + step.y]*(1-fx)*fy;
                weight += (1-fx)*fy;
            }
            if ((dx < dimin.x-1) && (dy < dimin.y-1)) {
                val += Itmp[idxO + step.x + step.y]*fx*fy;
                weight += fx*fy;
            }

            // Get new value
            if (weight != 0) {
                this->field[idx] = val / weight;
            }
        }
    }

    // Free temporary image
    delete[] Itmp;

    // Done
    return;
} 

// Overload operator=
Image& Image::operator=(const Image& im) {
    if (this->dimin != im.get_dimensions()) {
        mexErrMsgTxt("Error: in Image::operator=(const Image& )," 
                      "assignment cannot be done because dimensions of "
                      "input and output object are not the same.\n");
    }
    else {
        // Copy the contents from the input to object
        memcpy(this->field, im.get_field(), this->sizein*sizeof(float));
    }

    // Done
    return *this;
}

// Overload operators from base class
Image Image::operator+(const Image& im) const {
    Image imout(*this);
    imout.Field<float>::operator+=(im);
    //return imout.Field<float>::operator+=(im);
    return imout;
}

Image& Image::operator+=(const Image& im) {
    this->Field<float>::operator+=(im);
    return *this;
}

Image Image::operator-(const Image& im) const {
    Image imout(*this);
    imout.Field<float>::operator-=(im);
    return imout;
}

Image& Image::operator-=(const Image& im) {
    this->Field<float>::operator-=(im);
    return *this;
}