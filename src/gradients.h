#ifndef _GRADIENTS_H_
#define _GRADIENTS_H_

#include <src/coord2d.h>

namespace gradients {
    template <typename T>
    __inline__ T partial_x(T* field, const unsigned int idx, const unsigned int i, const dim& dimin) {
        if (i == 0) {
            return field[idx+1] - field[idx]; 
        }
        else if (i == dimin.x - 1) {
            return field[idx] - field[idx-1];
        }
        else {
            return (field[idx+1] - field[idx-1])/2.0f;
        }
    }

    template <typename T>
    __inline__ T partial_y(T *field, const unsigned int idx, const unsigned int j, const dim& dimin) {
        if (j == 0) {
            return field[idx + dimin.x] - field[idx];
        }
        else if (j == dimin.y - 1) {
            return field[idx] - field[idx - dimin.x];
        }
        else {
            return (field[idx + dimin.x] - field[idx-dimin.x])/2.0f;
        }
    }

    template <typename T>
    __inline__ T partial_xx(T *field, const unsigned int idx, const unsigned int i, const dim& dimin) {
        if (i == 0) {
            return field[idx] - field[idx+1]*2 + field[idx+2];
        }
        else if (i == dimin.x-1) {
            return field[idx-2] - field[idx-1]*2 + field[idx];
        }
        else {
            return field[idx+1] - field[idx]*2 + field[idx-1];
        }
    }

    template <typename T>
    __inline__ T partial_yy(T *field, const unsigned int idx, const unsigned int j, const dim& dimin) {
        if (j == 0) {
            return field[idx] - field[idx+dimin.x]*2 + field[idx+2*dimin.x];
        }
        else if (j == dimin.y-1) {
            return field[idx-2*dimin.x] - field[idx-dimin.x]*2 + field[idx];
        }
        else {
            return field[idx+dimin.x] - field[idx]*2 + field[idx-dimin.x];
        }
    }

    template <typename T>
    __inline__ T qlaplacian(T *field, const unsigned int idx, const unsigned int i, const unsigned int j, const dim& dimin) {
        if ((i == 0) || (i == dimin.x-1) ||
            (j == 0) || (j == dimin.y-1)) {
            return T(0.0f);
        }
        else {
            return (field[idx-1] + field[idx+1] + field[idx-dimin.x] + field[idx+dimin.x])/6.0f
                + (field[idx-1-dimin.x] + field[idx-1+dimin.x] +  field[idx+1-dimin.x] + field[idx+1+dimin.x])/12.0f;
            //return (field[idx - 1] + field[idx + 1] + field[idx - dimin.x] + field[idx + dimin.x])/4.0f;
        }
    }
}

#endif