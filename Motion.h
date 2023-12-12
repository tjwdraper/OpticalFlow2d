#ifndef _MOTION_H_
#define _MOTION_H_

#include <Field.h>

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

        // Upsample and downsample
        void upSample(const Motion& mo);
        void downSample(const Motion& mo);
};

#endif