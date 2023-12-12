#ifndef _MOTION_H_
#define _MOTION_H_

#include <src/Field.h>

class Motion : public Field<vector2d> {
    public:
        // Constructors and deconstructors
        Motion(const dim dimin);
        Motion(const Motion& mo);
        ~Motion();

        // Getters and setters
        vector2d* get_motion() const;

        // Copy to input
        void copy_motion_to_input(double* mo) const;

        // Get some motion field properties
        float norm() const;

        // Upsample and downsample
        void upSample(const Motion& mo);
        void downSample(const Motion& mo);

        // Overload operators
        Motion& operator=(const Motion& mo);
        Motion operator+(const Motion& mo);
        Motion& operator+=(const Motion& mo);
        Motion operator-(const Motion& mo);
        Motion& operator-=(const Motion& mo);
};

#endif