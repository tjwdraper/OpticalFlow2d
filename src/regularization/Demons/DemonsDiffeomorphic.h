#ifndef _DEMONS_DIFFEOMORPHIC_H_
#define _DEMONS_DIFFEOMORPHIC_H_

#include <src/regularization/Demons/Demons.h>

class DemonsDiffeomorphic : public Demons {
    public:
        // Constructors and deconstructors
        DemonsDiffeomorphic(const dim dimin, 
            const float sigma_i = 1.0, const float sigma_x = 0.25, 
            const float sigma_diffusion = 2.0, const float sigma_fluid = 2.0,
            const unsigned int kernelwidth = 5);
        ~DemonsDiffeomorphic();

        // Overload method from base class
        void get_update(Motion *motion, const Image *Iref, const Image* Imov);
};

#endif