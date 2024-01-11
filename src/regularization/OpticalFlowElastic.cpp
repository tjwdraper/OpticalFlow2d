#include <src/regularization/OpticalFlowElastic.h>
#include <src/gradients.h>

// Constructors and deconstructors
OpticalFlowElastic::OpticalFlowElastic(const dim dimin, const float mu, const float lambda, const float omega) : OpticalFlow(dimin) {
    this->mu = mu;
    this->lambda = lambda;
    this->omega = omega;
}

OpticalFlowElastic::~OpticalFlowElastic() {}

void OpticalFlowElastic::get_update(Motion* motion, const Image* Iref, const Image* Imov) {
    // Get the force
    this->OpticalFlow::get_force(this->force, motion);

    // Do a SOR iteration
    this->OpticalFlowElastic::SOR_iteration(motion);
}

void OpticalFlowElastic::SOR_iteration(Motion* motion) const {
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