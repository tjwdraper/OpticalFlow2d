#ifndef _ITERATIVE_SOLVER_H_
#define _ITERATIVE_SOLVER_H_

#include "coord2d.hpp"
#include "conv2d.hpp"
#include "gradients.hpp"
#include "Field.hpp"
#include "RegistrationConfiguration.hpp"

#include <memory>

// The interface for the iterative solver
class IterativeSolverStrategy {
    protected:
        SolverParameters _parameters;

    public:
        IterativeSolverStrategy(const SolverParameters& parameters) : _parameters(parameters) {}
        virtual ~IterativeSolverStrategy() = default;
        virtual void estimate_optical_flow(
            opticalflow::Motion& motion,
            const opticalflow::Image& Iref,
            const opticalflow::Image& Imov
        ) = 0;

        const SolverParameters& parameters() const {
            return _parameters;
        }
};

// Concrete implementations of the iterative solver interface
class HornSchunckStrategy : public IterativeSolverStrategy {
    private:
        dim dimin;

        std::unique_ptr<opticalflow::Motion> _motionAverage;
        std::unique_ptr<opticalflow::Motion> _imageGradient;
        std::unique_ptr<opticalflow::Image> _imageDifference;

        // Remove the default constructor
        HornSchunckStrategy() = delete;

    public:
        HornSchunckStrategy(const SolverParameters& parameters) : dimin(parameters.dimin), IterativeSolverStrategy(parameters) {
            _motionAverage = std::make_unique<opticalflow::Motion>(dimin);
            _imageGradient = std::make_unique<opticalflow::Motion>(dimin);
            _imageDifference = std::make_unique<opticalflow::Image>(dimin);
        }
        ~HornSchunckStrategy() = default;

        void estimate_optical_flow(opticalflow::Motion& motion, const opticalflow::Image& Iref, const opticalflow::Image& Imov) override {
            // Registration parameters
            const SolverParameters& parameters = _parameters;

            // Auxiliary variable
            opticalflow::Motion motionNext(dimin);

            // Create convolution kernels
            const average_conv2d filter(dim(3,3));

            // Dereference some variables
            opticalflow::Motion& motionAverage = *_motionAverage;
            opticalflow::Motion& imageGradient = *_imageGradient;
            opticalflow::Image& imageDifference = *_imageDifference;

            // Calculate spatial and temporal derivative
            gradients::gradient(imageGradient, 0.5*(Iref + Imov)); // Symmetric gradient

            imageDifference = Imov - Iref;
            filter.convolute(imageDifference);

            // Regularization parameters
            double falpha = 4.0 * parameters.alpha;

            for (std::size_t iter = 0; iter < _parameters.niter; ++iter) {
                gradients::horn_schunck_average(motionAverage, motion);

                for (std::size_t idx = 0; idx < _parameters.size; ++idx) {
                    // Get values
                    const vector2d& mAvg = motionAverage.at(idx);
                    const vector2d& dI = imageGradient.at(idx);
                    const double& It = imageDifference.at(idx);

                    // Calculate prefactor
                    double s = (dot(mAvg, dI) + It) / (falpha + normsq(dI));

                    // Horn-Schunck iteration
                    motionNext.set(mAvg - s * dI, idx);
                }

                // Convergence criteria
                double du = opticalflow::motion::norm(motion - motionNext);
                double m = opticalflow::motion::norm(motionNext);
                if (du < _parameters.eps * std::max(1.0, m))
                    break;

                // Swap
                std::swap(motion, motionNext);
            }

        }
};

class CorneliusKanadeStrategy : public IterativeSolverStrategy {
    private:
        dim dimin;

        std::unique_ptr<opticalflow::Motion> _motionAverage;
        std::unique_ptr<opticalflow::Image> _cAverage;
        std::unique_ptr<opticalflow::Motion> _imageGradient;
        std::unique_ptr<opticalflow::Image> _imageDifference;

        // Remove the default constructor
        CorneliusKanadeStrategy() = default;


    public:
        CorneliusKanadeStrategy(const SolverParameters& parameters) : dimin(parameters.dimin), IterativeSolverStrategy(parameters) {
            _motionAverage = std::make_unique<opticalflow::Motion>(dimin);
            _cAverage = std::make_unique<opticalflow::Image>(dimin);
            _imageGradient = std::make_unique<opticalflow::Motion>(dimin);
            _imageDifference = std::make_unique<opticalflow::Image>(dimin);
        }
        ~CorneliusKanadeStrategy() = default;

        void estimate_optical_flow(opticalflow::Motion& motion, const opticalflow::Image& Iref, const opticalflow::Image& Imov) override {
            // Registration parameters
            const SolverParameters& parameters = _parameters;

            // The residue field
            opticalflow::Image c(dimin);

            // Auxiliary variable
            opticalflow::Motion motionNext(dimin);
            opticalflow::Image cNext(dimin);

            // Create convolution kernels
            average_conv2d filter(dim(3,3));

            // Dereference some variables
            opticalflow::Motion& motionAverage = *_motionAverage;
            opticalflow::Image& cAverage = *_cAverage;
            opticalflow::Motion& imageGradient = *_imageGradient;
            opticalflow::Image& imageDifference = *_imageDifference;

            // Calculate spatial and temporal derivative
            gradients::gradient(imageGradient, 0.5*(Iref + Imov)); // Symmetric gradient

            imageDifference = Imov - Iref;
            filter.convolute(imageDifference);

            // Regularization parameters
            double falpha = 4.0 * parameters.alpha;
            double fbeta = 4.0 * parameters.beta;

            for (std::size_t iter = 0; iter < parameters.niter; ++iter) {
                gradients::horn_schunck_average(motionAverage, motion);
                gradients::horn_schunck_average(cAverage, c);

                for (std::size_t idx = 0; idx < parameters.size; ++idx) {
                    // Get values
                    const vector2d& mAvg = motionAverage.at(idx);
                    const double& cAvg = cAverage.at(idx);
                    const vector2d& dI = imageGradient.at(idx);
                    const double& It = imageDifference.at(idx);

                    // Calculate prefactor
                    double s = (dot(mAvg, dI) + It - cAvg) / (1.0 + normsq(dI)/falpha + 1.0/fbeta);

                    // Cornelius-Kanade iteration
                    motionNext.set(mAvg - s * dI / falpha, idx);
                    cNext.set(cAvg + s / fbeta, idx);
                }

                // Convergence criteria
                double du = opticalflow::motion::norm(motion - motionNext);
                double m = opticalflow::motion::norm(motionNext);
                if (du < _parameters.eps * std::max(1.0, m))
                    break;

                // Swap
                std::swap(motion, motionNext);
                std::swap(c, cNext);
            }
        }
};

// The context
class IterativeSolver {
    private:
        std::unique_ptr<IterativeSolverStrategy> _strategy;

    public:
        // Constructors and deconstructors
        explicit IterativeSolver(std::unique_ptr<IterativeSolverStrategy> strategy) : _strategy(std::move(strategy)) {};
        ~IterativeSolver() = default;

        // Getters and setters
        const SolverParameters& parameters() const {
            return _strategy->parameters();
        }

        // Estimate the optical flow
        void estimate_optical_flow(
            opticalflow::Motion& motion, 
            const opticalflow::Image& Iref, 
            const opticalflow::Image& Imov) {
            _strategy->estimate_optical_flow(motion, Iref, Imov);
        }        
};

#endif