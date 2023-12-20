#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <src/coord2d.h>
#include <src/Motion.h>

class Logger {
    public:
        // Constructors and deconstrucotrs
        Logger(const dim dimin, const unsigned int niter);
        ~Logger();

        // Update error array with new motion field
        void update_error(const Motion* motion);

        // Get currennt error
        float get_error_at_current_iteration() const;

    private:
        // Display error
        void show_error_at_iteration(const unsigned int iter) const;
        void show_error_at_current_iteration() const;
        void show_all_error() const;

        // Size and dimensions of the motion field
        dim dimin;
        unsigned int sizein;
        dim step;
        
        // Auxiliary motion field
        Motion *prev;
        Motion *diff;
        
        // Error array
        unsigned int niter;
        float *error;

        // Counter to keep track at which iteration we are
        unsigned int iter = 0;
};

#endif