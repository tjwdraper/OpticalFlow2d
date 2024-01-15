/*
    We provide the definitions of the functions for image registration using optical flow + fluid 
    regularization. Each CUDA kernel is wrapped around a host function.

    @company: UMC Utrecht
    @author: Tom Draper
    @date: 15-6-2022
*/
#include <cuda.h>
#include <mex.h>
#include <dim.cuh>
#include <coord3D.cuh>
#include <fluid_solver.cuh>
#include <gradients.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include <mex.h>

struct norm_unary : public unary_function<coord3D, coord3D> {
    __host__ __device__ float operator()(const coord3D &c) const {
        return sqrt(c.x*c.x+c.y*c.y+c.z*c.z);
    }
};

/*
    Jacobian
*/

__global__ void jacobian_determinantKernel(float *jacobian, coord3D *motion, dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) || 
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    //
    coord3D dudx = partial_x(motion, idx, i, dimin);
    coord3D dudy = partial_y(motion, idx, j, dimin);
    coord3D dudz = partial_z(motion, idx, k, dimin);
    // Get the minimum value of the Jacobian of the transformation
    jacobian[idx] = (1.0+dudx.x)*((1.0+dudy.y)*(1.0+dudz.z)-dudy.z*dudz.y) - 
            dudx.y*(dudy.x*(1.0+dudz.z)-dudz.x*dudy.z) + 
            dudx.z*(dudy.x*dudz.y-dudz.x*(1.0+dudy.y));
    // Done 
    return;
}

__host__ void jacobian_determinant(float *jacobian, coord3D *motion, dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    jacobian_determinantKernel <<<nBlocks, threadsPerBlock>>> (jacobian,
            motion,
            dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch jacobian kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

/*
    Dirichlet boundary conditions
*/
__global__ void enforceDirichletBoundaryConditionsKernel(coord3D *field, dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Check if thread calculates on boundary
    if ((k == 0) || (j == 0) || (i == 0) || 
        (k == dimin.z-1) || (j == dimin.y-1) || (i == dimin.x-1)) {
        // Compute the absolute index for future use.
        idx = k * dimin.x * dimin.y + j * dimin.x + i;
        // Set field to zero
        field[idx] = 0.0;
    }
}

__host__ void enforceDirichletBoundaryConditions(coord3D *field, dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    enforceDirichletBoundaryConditionsKernel <<< nBlocks, threadsPerBlock >>> (field, dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch enforceDirichletBoundaryConditions kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

/* 
    Neumann boundary conditions
*/

__global__ void enforceNeumannBoundaryConditionsKernel(coord3D *field, dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    // Get the step
    dim step(1, dimin.x, dimin.x * dimin.y);
    // Sides
    if (i == 0)              { field[idx] = field[idx + step.x]; }
    if (i == dimin.x - 1)    { field[idx] = field[idx - step.x]; }
    if (j == 0)              { field[idx] = field[idx + step.y]; }
    if (j == dimin.y - 1)    { field[idx] = field[idx - step.y]; }
    if (k == 0)              { field[idx] = field[idx + step.z]; } 
    if (k == dimin.z - 1)    { field[idx] = field[idx - step.z]; }
    // Vertices
    if ((i == 0) && (j == 0))                    { field[idx] = field[idx + step.x + step.y]; }
    if ((i == 0) && (j == dimin.y-1))            { field[idx] = field[idx + step.x - step.y]; }
    if ((i == 0) && (k == 0))                    { field[idx] = field[idx + step.x + step.z]; }
    if ((i == 0) && (k == dimin.z-1))            { field[idx] = field[idx + step.x - step.z]; }
    if ((i == dimin.x-1) && (j == 0))            { field[idx] = field[idx - step.x + step.y]; }
    if ((i == dimin.x-1) && (j == dimin.y-1))    { field[idx] = field[idx - step.x - step.y]; }
    if ((i == dimin.x-1) && (k == 0))            { field[idx] = field[idx - step.x + step.z]; }
    if ((i == dimin.x-1) && (k == dimin.z-1))    { field[idx] = field[idx - step.x - step.z]; }
    if ((j == 0) && (k == 0))                    { field[idx] = field[idx + step.y + step.z]; }
    if ((j == 0) && (k == dimin.z-1))            { field[idx] = field[idx + step.y - step.z]; }
    if ((j == dimin.y-1) && (k == 0))            { field[idx] = field[idx - step.y + step.z]; }
    if ((j == dimin.y-1) && (k == dimin.z-1))    { field[idx] = field[idx - step.y - step.z]; }
    // Corners
    if ((i == 0) && (j == 0) && (k == 0))                            {field[idx] = field[idx + step.x + step.y + step.z];}
    if ((i == 0) && (j == 0) && (k == dimin.z-1))                    {field[idx] = field[idx + step.x + step.y - step.z];}
    if ((i == 0) && (j == dimin.y-1) && (k == 0))                    {field[idx] = field[idx + step.x - step.y + step.z];}
    if ((i == 0) && (j == dimin.y-1) && (k == dimin.z-1))            {field[idx] = field[idx + step.x - step.y - step.z];}
    if ((i == dimin.x-1) && (j == 0) && (k == 0))                    {field[idx] = field[idx - step.x + step.y + step.z];}
    if ((i == dimin.x-1) && (j == 0) && (k == dimin.z-1))            {field[idx] = field[idx - step.x + step.y - step.z];}
    if ((i == dimin.x-1) && (j == dimin.y-1) && (k == 0))            {field[idx] = field[idx - step.x - step.y + step.z];}
    if ((i == dimin.x-1) && (j == dimin.y-1) && (k == dimin.z-1))    {field[idx] = field[idx - step.x - step.y - step.z];}
    // Done
    return;
}

__host__ void enforceNeumannBoundaryConditions(coord3D *field, dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    enforceNeumannBoundaryConditionsKernel <<< nBlocks, threadsPerBlock >>> (field, dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch enforceNeumannBoundaryConditions kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

/*
    Get force vector from SSD

*/
__global__ void generate_forceKernel(coord3D *force, 
    float *Iref, float *Ireg, 
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    force[idx] = coord3D(partial_x(Ireg, idx, i, dimin),
                         partial_y(Ireg, idx, j, dimin),
                         partial_z(Ireg, idx, k, dimin)) * (Ireg[idx] - Iref[idx]);
    // Done 
    return;
}

__host__ void generate_force(coord3D *force, 
    float *Iref, float *Ireg, 
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    generate_forceKernel <<<nBlocks, threadsPerBlock>>> (force,
            Iref, Ireg,
            dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch generate_force kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

__global__ void generate_add_reverse_forceKernel(coord3D *force, 
    float *Imov, float *Ireg, 
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;   
    
    coord3D reverse_force = coord3D(partial_x(Ireg, idx, i, dimin),
                                    partial_y(Ireg, idx, j, dimin),
                                    partial_z(Ireg, idx, k, dimin)) * (Imov[idx] - Ireg[idx]);
    force[idx] = (force[idx] + reverse_force)/2;
    // Done 
    return;
}

__host__ void generate_add_reverse_force(coord3D *force, 
    float *Imov, 
    float *Ireg, 
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    generate_add_reverse_forceKernel <<<nBlocks, threadsPerBlock>>> (force,
            Imov, Ireg,
            dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch generate_force kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

__global__ void generate_gradient_forceKernel(coord3D *force, 
    float *Iref, float *Ireg, 
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    //
    float eta = 0.03;
    coord3D grad_f = coord3D(partial_x(Ireg, idx, i, dimin),
                             partial_y(Ireg, idx, j, dimin),
                             partial_z(Ireg, idx, k, dimin));
    coord3D grad_g = coord3D(partial_x(Iref, idx, i, dimin),
                             partial_y(Iref, idx, j, dimin),
                             partial_z(Iref, idx, k, dimin));
    float norm_eta_f = sqrt(grad_f.x*grad_f.x + grad_f.y*grad_f.y + grad_f.z*grad_f.z + eta*eta);
    float norm_eta_g = sqrt(grad_g.x*grad_g.x + grad_g.y*grad_g.y + grad_g.z*grad_g.z + eta*eta);
    //
    grad_f /= norm_eta_f;
    grad_g /= norm_eta_g;
    //
    float innerprod = grad_f.x*grad_g.x + grad_f.y*grad_g.y + grad_f.z*grad_g.z;
    //
    float dfdxx = partial_xx(Ireg, idx, i, dimin);
    float dfdyy = partial_yy(Ireg, idx, j, dimin);
    float dfdzz = partial_zz(Ireg, idx, k, dimin);
    float dfdxy = partial_xy(Ireg, idx, i, j, dimin);
    float dfdxz = partial_xz(Ireg, idx, i, k, dimin);
    float dfdyz = partial_yz(Ireg, idx, j, k, dimin);
    //
    force[idx].x = (float)- 2.0*innerprod/norm_eta_f*( dfdxx*(grad_g.x - innerprod*grad_f.x) +
                                             dfdxy*(grad_g.y - innerprod*grad_f.y) +
                                             dfdxz*(grad_g.z - innerprod*grad_f.z) );
    force[idx].x = (float) -2.0*innerprod/norm_eta_f*( dfdxy*(grad_g.x - innerprod*grad_f.x) +
                                             dfdyy*(grad_g.y - innerprod*grad_f.y) +
                                             dfdyz*(grad_g.z - innerprod*grad_f.z) );
    force[idx].x = (float) -2.0*innerprod/norm_eta_f*( dfdxz*(grad_g.x - innerprod*grad_f.x) +
                                             dfdyz*(grad_g.y - innerprod*grad_f.y) +
                                             dfdzz*(grad_g.z - innerprod*grad_f.z) );

    // Done 
    return;
}

__host__ void generate_gradient_force(coord3D *force, 
    float *Iref, float *Ireg, 
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    generate_gradient_forceKernel <<<nBlocks, threadsPerBlock>>> (force,
            Iref, Ireg,
            dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch generate_force kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

__global__ void optFlowKernel(coord3D *forceField,
    float *refImage, float *movImage,
    coord3D *motion,
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    forceField[idx] = coord3D(partial_x(movImage, idx, i, dimin),
                              partial_y(movImage, idx, j, dimin),
                              partial_z(movImage, idx, k, dimin));
    forceField[idx] *= movImage[idx] - refImage[idx] + motion[idx].x*forceField[idx].x + motion[idx].y*forceField[idx].y + motion[idx].z*forceField[idx].z; 
    // Done 
    return;
}

__host__ void optFlow(coord3D *forceField, 
    float *refImage, float *movImage,
    coord3D *motion, 
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    optFlowKernel <<<nBlocks, threadsPerBlock>>> (forceField,
            refImage, movImage,
            motion,
            dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch optFlow kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    // Check if regridding procedure should be initiated
    return;
}

/*
        Adaptive time-step and increment in motion field
*/
__global__ void incrementKernel(coord3D *increment,
    coord3D *motion,
    coord3D *velocity,
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    // 
    coord3D dudx = partial_x(motion, idx, i, dimin);
    coord3D dudy = partial_y(motion, idx, j, dimin);
    coord3D dudz = partial_z(motion, idx, k, dimin);
    // Get values for R and get dt
    increment[idx] = velocity[idx] - dudx*velocity[idx].x - dudy*velocity[idx].y - dudz*velocity[idx].z; 
    // Done
    return;
}

__host__ void increment(coord3D *increment, 
    coord3D *motion,
    coord3D *velocity, 
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    incrementKernel <<<nBlocks, threadsPerBlock>>> (increment,
        motion,
        velocity,
        dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch increment kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return; 
}


/*
    Integrate motion field over time using explicit Euler method
*/
__global__ void integrateKernel(coord3D *motion,
    coord3D *increment, float dt,
    dim dimin) {
    // Determine the relative index of the current GPU thread.
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds.
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use.
    idx = k * dimin.x * dimin.y + j * dimin.x + i;
    // Calculate new entry
    motion[idx] += increment[idx]*dt;
    // Done
    return;
}

__host__ void integrate(coord3D *motion, 
    coord3D *increment, float dt,
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Launch kernel.
    integrateKernel <<<nBlocks, threadsPerBlock>>> (motion, increment, dt, dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch eulerIntegration kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return;
}