#include <src/regularization/OpticalFlow/OpticalFlowFluid.h>
#include <src/Logger.h>
#include <src/gradients.h>

#define PI 3.14159265

void OpticalFlowFluid::SOR_iteration(Motion* motion) const {
    // Get the dimensions and step size of the motion field
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get the overrelaxation and regularisation parameters
    const float& omega  = this->omega;
    const float& mu     = this->mu;
    const float& lambda = this->lambda;

    // Get a copy to the pointer of the vector fields
    vector2d *x = motion->get_motion();
    vector2d *b = this->force->get_motion();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 1; i < dimin.x-1; i++) {
        for (unsigned int j = 1; j < dimin.y-1; j++) {
            idx = i * step.x + j * step.y;

            x[idx].x = (1.0f-omega)*x[idx].x + omega / (-6*mu-2*lambda) * ( b[idx].x - 
                mu * (x[idx + step.x].x + x[idx - step.x].x + x[idx + step.y].x + x[idx - step.y].x) -
                (mu+lambda) * (x[idx + step.x].x + x[idx - step.x].x + 0.25f*(x[idx + step.x + step.y].y - x[idx - step.x + step.y].y - x[idx + step.x - step.y].y + x[idx - step.x - step.y].y))    
            );

            x[idx].y = (1.0f-omega)*x[idx].y + omega / (-6*mu-2*lambda) * ( b[idx].y - 
                mu * (x[idx + step.x].y + x[idx - step.x].y + x[idx + step.y].y + x[idx - step.y].y) -
                (mu+lambda) * (x[idx + step.x].y + x[idx - step.x].y + 0.25f*(x[idx + step.x + step.y].x - x[idx - step.x + step.y].x - x[idx + step.x - step.y].x + x[idx - step.x - step.y].x))    
            );
        }
    }

    // Done
    return;
}

OpticalFlowFluid::OpticalFlowFluid(const dim dimin, const float mu, const float lambda, const float omega) : OpticalFlow(dimin) {
    // Set regularisation and model parameters
    this->mu = mu;
    this->lambda = lambda;
    this->omega = omega;

    // Allocate memory for the velocity field
    this->velocity = new Motion(this->dimin);
    this->increment = new Motion(this->dimin);
}

OpticalFlowFluid::~OpticalFlowFluid() {
    delete this->velocity;
    delete this->increment;
}


void OpticalFlowFluid::get_increment(const Motion* motion) {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Get copy of pointer to the motion and force field data
    vector2d *mo  = motion->get_motion();
    vector2d* vel = this->velocity->get_motion();
    vector2d* R = this->increment->get_motion();

    // Iterate over voxels
    unsigned int idx;
    vector2d dudx, dudy;
    vector2d u, v;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u = mo[idx];
            v = vel[idx];

            dudx = gradients::partial_x(mo, idx, i, dimin);
            dudy = gradients::partial_y(mo, idx, j, dimin);

            R[idx] = v - dudx * v.x - dudy * v.y;
        }
    }

    // Done
    return;
}

void OpticalFlowFluid::estimate_timestep() {
    this->timestep = this->dumax / this->increment->maxabs();
    mexPrintf("Dumax: %.3f\tMaxabs increment: %.3f\t Timestep: %.3f\n", this->dumax, this->increment->maxabs(), this->timestep);
}

void OpticalFlowFluid::integrate(Motion *motion) const {
    // Get dimensions of images/motion fields
    const dim& dimin = this->dimin;
    const dim& step = this->step;

    // Time step
    const float& timestep = this->timestep;

    // Get copy of pointer to the motion and force field data
    vector2d *u = motion->get_motion();
    vector2d *R = this->increment->get_motion();

    // Iterate over voxels
    unsigned int idx;
    for (unsigned int i = 0; i < dimin.x; i++) {
        for (unsigned int j = 0; j < dimin.y; j++) {
            idx = i * step.x + j * step.y;

            u[idx] += R[idx] * timestep;
        }
    }

    // Done
    return;
}

void OpticalFlowFluid::get_update(Motion* motion, const Image* Iref, const Image* Imov) {
    // Get the force
    this->OpticalFlow::get_force(this->force, motion);

    // Get the velocity field
    this->OpticalFlowFluid::SOR_iteration(this->velocity);

    // Integrate the material derivative equation to get the next iteration of the motion field
    this->OpticalFlowFluid::get_increment(motion);

    this->OpticalFlowFluid::estimate_timestep();

    if (this->timestep >= 65.0f) {
        return;
    }

    this->OpticalFlowFluid::integrate(motion);
}