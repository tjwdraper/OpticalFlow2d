#include <src/Motion.h>

#include <mex.h>

// Constructors and deconstructors
Motion::Motion(const dim dimin) : Field<vector2d>(dimin) {}

Motion::Motion(const Motion& mo) : Field<vector2d>(mo) {}

Motion::~Motion() {}

// Getters and setters
vector2d* Motion::get_motion() const {
    return this->Field<vector2d>::get_field();
}


// Copy to input
void Motion::copy_motion_to_input(double* mo) const {
    // Get the dimensions of the Motion
    const dim& dimin = this->dimin;
    const dim& step = this->step;
    const unsigned int& sizein = this->sizein;

    // Store the input in the object
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            mo[i*step.x + j*step.y + 0*sizein] = (double) this->field[i*step.x + j*step.y].x;
            mo[i+step.x + j*step.y + 1*sizein] = (double) this->field[i*step.x + j*step.y].y;
        }
    }

    // Done
    return;
}

// Get some motion field properties
float Motion::norm() const {
    float norm = 0.0f;
    for (unsigned int i = 0; i < this->sizein; i++) {
        norm += std::pow(this->field[i].x, 2) + std::pow(this->field[i].y, 2);
    }
    return std::sqrt(norm)/this->sizein;
}

// Upsample and downsample
void Motion::upSample(const Motion& mo) {
    this->Field<vector2d>::upSample(mo);

    const dim& dimin     = this->get_dimensions();
    const dim& dimmotion = mo.get_dimensions();

    vector2d ratio((float) dimin.x / (float) dimmotion.x, 
                   (float) dimin.y / (float) dimmotion.y);

    for (int i = 0; i < this->sizein; i++) {
        this->field[i].x *= ratio.x;
        this->field[i].y *= ratio.y;
    }

    // Done
    return;
}

void Motion::downSample(const Motion& mo) {
    this->Field<vector2d>::downSample(mo);

    const dim& dimin     = this->get_dimensions();
    const dim& dimmotion = mo.get_dimensions();

    vector2d ratio((float) dimin.x / (float) dimmotion.x, 
                   (float) dimin.y / (float) dimmotion.y);

    for (int i = 0; i < this->sizein; i++) {
        this->field[i].x *= ratio.x;
        this->field[i].y *= ratio.y;
    }

    // Done
    return;
}

// Overload operator=
Motion& Motion::operator=(const Motion& im) {
    if (this->dimin != im.get_dimensions()) {
        mexErrMsgTxt("Error: in Motion::operator=(const Motion& )," 
                      "assignment cannot be done because dimensions of "
                      "input and output object are not the same.\n");
    }
    else {
        // Copy the contents from the input to object
        memcpy(this->field, im.get_field(), this->sizein*sizeof(vector2d));
    }

    // Done
    return *this;
}

// Overload operators from base class
Motion Motion::operator+(const Motion& mo) const {
    Motion moout(*this);
    moout.Field<vector2d>::operator+=(mo);
    return moout;
}

Motion& Motion::operator+=(const Motion& mo) {
    this->Field<vector2d>::operator+=(mo);
    return *this;
}

Motion Motion::operator-(const Motion& mo) const {
    Motion moout(*this);
    moout.Field<vector2d>::operator-=(mo);
    return moout;
}

Motion& Motion::operator-=(const Motion& mo) {
    this->Field<vector2d>::operator-=(mo);
    return *this;
}