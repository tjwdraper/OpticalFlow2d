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

void Motion::reset() {
    memset(this->field, 0, this->sizein*sizeof(vector2d));
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
        norm += std::sqrt(std::pow(this->field[i].x, 2) + std::pow(this->field[i].y, 2));
    }
    return norm / this->sizein;
    //return std::sqrt(norm)/this->sizein;
}

// Upsample and downsample
void Motion::upSample(const Motion& mo) {
    try {
        this->Field<vector2d>::upSample(mo);
    }
    catch (const std::invalid_argument& e) {
        const std::string mes = "Error in Motion::upSample(const Motion& mo): " + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
    }

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
    try {
        this->Field<vector2d>::downSample(mo);
    }
    catch (const std::invalid_argument& e) {
        const std::string mes = "Error in Motion::downSample(const Motion& im): " + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
    }

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

void Motion::accumulate(const Motion& mo) {
    if (this->dimin != mo.get_dimensions()) {
        throw std::invalid_argument("Error in Motion::accumulate(const Motion& mo): input dimensions should match target dimensions");
    }

    // Get the dimensions of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;
    
    // Make a copy of this object
    vector2d *motot = new vector2d[this->sizein];
    memcpy(motot, this->field, this->sizein*sizeof(vector2d));

    // Get a copy to the pointer of the input motion field
    vector2d *moin = mo.get_field();

    // Iterate over output voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            // Get index in image
            idx = i * step.x + j * step.y;
            
            // Get the warped index
            float px = i + moin[idx].x; int dx = std::floor(px); float fx = px - dx;
            float py = j + moin[idx].y; int dy = std::floor(py); float fy = py - dy;

            // Check if warped index is out of bounds
            if ((dx < 0) || (dx >= dimin.x) ||
                (dy < 0) || (dy >= dimin.y)) {
                continue;
            }

            // Get initial update
            this->field[idx] = moin[idx];
            
            // Otherwise, get the value through linear interpolation
            int idxO = dx * step.x + dy * step.y;
            vector2d val = motot[idxO]*(1-fx)*(1-fy);
            float weight = (1-fx)*(1-fy);
            if (dx < dimin.x-1) {
                val += motot[idxO + step.x]*fx*(1-fy);
                weight += fx*(1-fy);
            }
            if (dy < dimin.y-1) {
                val += motot[idxO + step.y]*(1-fx)*fy;
                weight += (1-fx)*fy;
            }
            if ((dx < dimin.x-1) && (dy < dimin.y-1)) {
                val += motot[idxO + step.x + step.y]*fx*fy;
                weight += fx*fy;
            }

            // Get new value
            if (weight != 0) {
                this->field[idx] += val / weight;
            }
        }
    }

    // Free up the memory
    delete[] motot;

    // Done
    return;
}

// Overload operator=
Motion& Motion::operator=(const Motion& mo) {
    if (this->dimin != mo.get_dimensions()) {
        throw std::invalid_argument("Motion::operator=(const Motion& mo) input argument has to have same dimensions as target");
    }

    // Copy the contents from the input to object
    try {
        memcpy(this->field, mo.get_field(), this->sizein*sizeof(vector2d));
    }
    catch(const std::bad_alloc& e) {
        const std::string mes = "Error in Motion::operator=(const Motion& mo) " + 
            std::string(e.what()) + 
            std::string("\n");
        mexErrMsgTxt(mes.c_str());
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