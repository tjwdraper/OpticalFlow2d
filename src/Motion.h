#ifndef _MOTION_H_
#define _MOTION_H_

#include <src/Field.h>
#include <src/Kernel.h>

class Motion : public Field<vector2d> {
    public:
        // Constructors and deconstructors
        Motion(const dim dimin);
        Motion(const Motion& mo);
        ~Motion();

        // Getters and setters
        vector2d* get_motion() const;
        void reset();

        // Copy to input
        void copy_motion_to_input(double* mo) const;

        // Get some motion field properties
        float norm() const;
        float maxabs() const;

        // Upsample and downsample
        void upSample(const Motion& mo);
        void downSample(const Motion& mo);

        // Accumulate motion
        void accumulate(const Motion& mo);

        // Enforce boundary conditions
        void Neumann_boundaryconditions();
        void Dirichlet_boundaryconditions();

        // Motion field exponential
        void exp();

        // Convolute with kernel
        void convolute(const Kernel& kernel);

        // Overload operator=
        Motion& operator=(const Motion& mo);

        // Overload operators from base class
        Motion operator+(const Motion& mo) const;
        Motion& operator+=(const Motion& mo);
        Motion operator-(const Motion& mo) const;
        Motion& operator-=(const Motion& mo);

        Motion& operator*=(const float& val);
};

#endif