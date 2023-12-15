#include <src/Logger.h>

#include <mex.h>

// Constructors and deconstructors
Logger::Logger(const dim dimin, const unsigned int niter) {
    // Get the size and dimensions of the motion fields
    this->dimin = dimin;
    this->sizein = dimin.x * dimin.y;
    this->step = dim(1, dimin.x);

    // Allocate memory for the auxiliary motion fields
    this->prev = new Motion(this->dimin);
    this->diff = new Motion(this->dimin);

    // Allocate memory for the error array
    this->niter = niter;
    this->error = new float[niter+1];
}

Logger::~Logger() {
    // Free memory from the stack
    delete this->prev;
    delete this->diff;
    delete this->error;
}

// Update error with new motion field
void Logger::update_error(const Motion* motion) {
    // Take the difference between current and previous iteration
    //*this->diff = *motion;
    //*this->diff -= *this->prev;
    *this->diff = *motion - *this->prev;

    // Calculate the Euclidean norm of the difference
    float prevnorm = this->prev->norm();

    this->error[this->iter] = (prevnorm == 0 ? 0.0f : this->diff->norm() / prevnorm);

    // Copy current estimate to prev
    *this->prev = *motion;

    // Show current error
    this->Logger::show_error_at_current_iteration();

    // Increment counter
    this->iter++;
}

// Display error
void Logger::show_error_at_iteration(const unsigned int iter) const {
    if (iter <= this->niter) {
        mexPrintf("Iteration: %d\tError:%.4f\n", iter, this->error[iter]);
    }
    else {
        mexErrMsgTxt("Error: Logger::iter > Logger::niter, so current iteration cannot be shown.\n");
    }
}

void Logger::show_error_at_current_iteration() const {
    this->show_error_at_iteration(this->iter);
}

void Logger::show_all_error() const {
    for (unsigned int iter = 0; iter <= this->iter; iter++) {
        this->show_error_at_iteration(iter);
    }
}
