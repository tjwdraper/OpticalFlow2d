#include <spectral_solver.cuh>
#include <mex.h>

#define PI 3.14159265

__global__ void getG (coord3D *g, coord3D *velocity, coord3D *force, float gamma, dim dimin) {
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
    g[idx] = velocity[idx] - force[idx] * gamma;
    // Done
    return;
}

__global__ void getGcoefs (cufftComplex *coefs_g_x, cufftComplex *coefs_g_y, cufftComplex *coefs_g_z,
                           cufftComplex *coefs_v_x, cufftComplex *coefs_v_y, cufftComplex *coefs_v_z,
                           cufftComplex *coefs_f_x, cufftComplex *coefs_f_y, cufftComplex *coefs_f_z,
                           float gamma,
                           dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    idx = k + j*dimin.z + i*dimin.z*dimin.y;
    // Get update
    coefs_g_x[idx].x = coefs_v_x[idx].x - coefs_f_x[idx].x * gamma;
    coefs_g_x[idx].y = coefs_v_x[idx].y - coefs_f_x[idx].y * gamma;
    coefs_g_y[idx].x = coefs_v_y[idx].x - coefs_f_y[idx].x * gamma;
    coefs_g_y[idx].y = coefs_v_y[idx].y - coefs_f_y[idx].y * gamma;
    coefs_g_z[idx].x = coefs_v_z[idx].x - coefs_f_z[idx].x * gamma;
    coefs_g_z[idx].y = coefs_v_z[idx].y - coefs_f_z[idx].y * gamma;
    //
    return;
}

__global__ void getComponentsKernel(cufftReal *g_x, cufftReal *g_y, cufftReal *g_z, coord3D *g, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    int idxColMajor = i + j*dimin.x + k*dimin.x*dimin.y;
    int idxRowMajor = k + j*dimin.z + i*dimin.z*dimin.y;
    // Get update
    g_x[idxRowMajor] = g[idxColMajor].x;
    g_y[idxRowMajor] = g[idxColMajor].y;
    g_z[idxRowMajor] = g[idxColMajor].z;
}

__host__ void getComponents(cufftReal *g_x, cufftReal *g_y, cufftReal *g_z, coord3D *g, dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);

    // Decompose the components of g
    getComponentsKernel <<<nBlocks, threadsPerBlock>>> (g_x, g_y, g_z, g, dimin);

    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch getComponents kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return;
}

__global__ void divide(cufftComplex *coefs_g_x, cufftComplex *coefs_g_y, cufftComplex *coefs_g_z, 
    float *eigenvals, float mu, float lambda, float gamma, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    idx = k + j*dimin.z + i*dimin.z*dimin.y;

    // Real part
    const int step = dimin.x*dimin.y*dimin.z;
    cufftComplex a,b,c;
    a.x = eigenvals[idx + 0*step] * coefs_g_x[idx].x + eigenvals[idx + 3*step] * coefs_g_y[idx].x + eigenvals[idx + 4*step] * coefs_g_z[idx].x;
    b.x = eigenvals[idx + 3*step] * coefs_g_x[idx].x + eigenvals[idx + 1*step] * coefs_g_y[idx].x + eigenvals[idx + 5*step] * coefs_g_z[idx].x;
    c.x = eigenvals[idx + 4*step] * coefs_g_x[idx].x + eigenvals[idx + 5*step] * coefs_g_y[idx].x + eigenvals[idx + 2*step] * coefs_g_z[idx].x;

    // Imaginary part
    a.y = eigenvals[idx + 0*step] * coefs_g_x[idx].y + eigenvals[idx + 3*step] * coefs_g_y[idx].y + eigenvals[idx + 4*step] * coefs_g_z[idx].y;
    b.y = eigenvals[idx + 3*step] * coefs_g_x[idx].y + eigenvals[idx + 1*step] * coefs_g_y[idx].y + eigenvals[idx + 5*step] * coefs_g_z[idx].y;
    c.y = eigenvals[idx + 4*step] * coefs_g_x[idx].y + eigenvals[idx + 5*step] * coefs_g_y[idx].y + eigenvals[idx + 2*step] * coefs_g_z[idx].y;

    // Set new values
    coefs_g_x[idx] = a;
    coefs_g_y[idx] = b;
    coefs_g_z[idx] = c;
}

__global__ void divide(cufftComplex *coefs_v_x, cufftComplex *coefs_v_y, cufftComplex *coefs_v_z,
    cufftComplex *coefs_g_x, cufftComplex *coefs_g_y, cufftComplex *coefs_g_z, 
    float *eigenvals, float mu, float lambda, float gamma, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    idx = k + j*dimin.z + i*dimin.z*dimin.y;

    // Real part
    const int step = dimin.x*dimin.y*dimin.z;
    coefs_v_x[idx].x = eigenvals[idx + 0*step] * coefs_g_x[idx].x + eigenvals[idx + 3*step] * coefs_g_y[idx].x + eigenvals[idx + 4*step] * coefs_g_z[idx].x;
    coefs_v_y[idx].x = eigenvals[idx + 3*step] * coefs_g_x[idx].x + eigenvals[idx + 1*step] * coefs_g_y[idx].x + eigenvals[idx + 5*step] * coefs_g_z[idx].x;
    coefs_v_z[idx].x = eigenvals[idx + 4*step] * coefs_g_x[idx].x + eigenvals[idx + 5*step] * coefs_g_y[idx].x + eigenvals[idx + 2*step] * coefs_g_z[idx].x;

    // Imaginary part
    coefs_v_x[idx].y = eigenvals[idx + 0*step] * coefs_g_x[idx].y + eigenvals[idx + 3*step] * coefs_g_y[idx].y + eigenvals[idx + 4*step] * coefs_g_z[idx].y;
    coefs_v_y[idx].y = eigenvals[idx + 3*step] * coefs_g_x[idx].y + eigenvals[idx + 1*step] * coefs_g_y[idx].y + eigenvals[idx + 5*step] * coefs_g_z[idx].y;
    coefs_v_z[idx].y = eigenvals[idx + 4*step] * coefs_g_x[idx].y + eigenvals[idx + 5*step] * coefs_g_y[idx].y + eigenvals[idx + 2*step] * coefs_g_z[idx].y;
}

__global__ void construct(coord3D *g, cufftReal *g_x, cufftReal *g_y, cufftReal *g_z, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    // Get the length of the array
    int length = dimin.x * dimin.y * dimin.z;
    // Compute the absolute index for future use
    int idxColMajor = i + j*dimin.x + k*dimin.x*dimin.y;
    int idxRowMajor = k + j*dimin.z + i*dimin.z*dimin.y;
    // Get update
    g[idxColMajor].x = g_x[idxRowMajor]/length;
    g[idxColMajor].y = g_y[idxRowMajor]/length;
    g[idxColMajor].z = g_z[idxRowMajor]/length;
}

__global__ void divide_v2(
    cufftComplex* coefs_v_x, cufftComplex* coefs_v_y, cufftComplex* coefs_v_z,
    cufftComplex* coefs_f_x, cufftComplex* coefs_f_y, cufftComplex* coefs_f_z,
    float* eigenvals,
    float mu, float lambda, float gamma,
    dim dimin,
    int siter) {
    // Determine the relative index of the current GPU thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    int j = idx / dimin.x;
    int i = idx % dimin.x;
    
    // Check that the indices are within image bounds
    if ((k < 0) || (j < 0) || (i < 0) ||
        (k > dimin.z - 1) || (j > dimin.y - 1) || (i > dimin.x - 1)) {
        return;
    }
    
    // Compute the absolute index for future use
    idx = k + j * dimin.z + i * dimin.z * dimin.y;
    
    // Get step
    const int step = dimin.x * dimin.y * dimin.z;
    
    // Get components of inverse matrix
    float Axx = eigenvals[idx + 0 * step];
    float Ayy = eigenvals[idx + 1 * step];
    float Azz = eigenvals[idx + 2 * step];
    float Axy = eigenvals[idx + 3 * step];
    float Axz = eigenvals[idx + 4 * step];
    float Ayz = eigenvals[idx + 5 * step];
    
    cufftComplex coefs_v_x_val = coefs_v_x[idx];
    cufftComplex coefs_v_y_val = coefs_v_y[idx];
    cufftComplex coefs_v_z_val = coefs_v_z[idx];
    
    cufftComplex coefs_f_x_val = coefs_f_x[idx];
    cufftComplex coefs_f_y_val = coefs_f_y[idx];
    cufftComplex coefs_f_z_val = coefs_f_z[idx];
    
    float coefs_v_x_real = coefs_v_x_val.x;
    float coefs_v_x_imag = coefs_v_x_val.y;
    
    float coefs_v_y_real = coefs_v_y_val.x;
    float coefs_v_y_imag = coefs_v_y_val.y;
    
    float coefs_v_z_real = coefs_v_z_val.x;
    float coefs_v_z_imag = coefs_v_z_val.y;
    
    float coefs_f_x_real = coefs_f_x_val.x;
    float coefs_f_x_imag = coefs_f_x_val.y;
    
    float coefs_f_y_real = coefs_f_y_val.x;
    float coefs_f_y_imag = coefs_f_y_val.y;
    
    float coefs_f_z_real = coefs_f_z_val.x;
    float coefs_f_z_imag = coefs_f_z_val.y;
    
    for (int r = 0; r < siter; ++r) {
        // Real part
        coefs_v_x_real = Axx * (coefs_v_x_real - coefs_f_x_real * gamma) +
                         Axy * (coefs_v_y_real - coefs_f_y_real * gamma) +
                         Axz * (coefs_v_z_real - coefs_f_z_real * gamma);
        
        coefs_v_y_real = Axy * (coefs_v_x_real - coefs_f_x_real * gamma) +
                         Ayy * (coefs_v_y_real - coefs_f_y_real * gamma) +
                         Ayz * (coefs_v_z_real - coefs_f_z_real * gamma);
        
        coefs_v_z_real = Axz * (coefs_v_x_real - coefs_f_x_real * gamma) +
                         Ayz * (coefs_v_y_real - coefs_f_y_real * gamma) +
                         Azz * (coefs_v_z_real - coefs_f_z_real * gamma);
        
        // Imaginary part
        coefs_v_x_imag = Axx * (coefs_v_x_imag - coefs_f_x_imag * gamma) +
                         Axy * (coefs_v_y_imag - coefs_f_y_imag * gamma) +
                         Axz * (coefs_v_z_imag - coefs_f_z_imag * gamma);
        
        coefs_v_y_imag = Axy * (coefs_v_x_imag - coefs_f_x_imag * gamma) +
                         Ayy * (coefs_v_y_imag - coefs_f_y_imag * gamma) +
                         Ayz * (coefs_v_z_imag - coefs_f_z_imag * gamma);
        
        coefs_v_z_imag = Axz * (coefs_v_x_imag - coefs_f_x_imag * gamma) +
                         Ayz * (coefs_v_y_imag - coefs_f_y_imag * gamma) +
                         Azz * (coefs_v_z_imag - coefs_f_z_imag * gamma);
    }
    
    // Update the output values
    coefs_v_x[idx].x = coefs_v_x_real;
    coefs_v_x[idx].y = coefs_v_x_imag;
    
    coefs_v_y[idx].x = coefs_v_y_real;
    coefs_v_y[idx].y = coefs_v_y_imag;
    
    coefs_v_z[idx].x = coefs_v_z_real;
    coefs_v_z[idx].y = coefs_v_z_imag;
}


__host__ void spectral_update_v2(coord3D *velocity,
    cufftReal *g_x, cufftReal *g_y, cufftReal *g_z,
    cufftComplex *coefs_f_x, cufftComplex *coefs_f_y, cufftComplex *coefs_f_z,
    cufftComplex *coefs_v_x, cufftComplex *coefs_v_y, cufftComplex *coefs_v_z,
    float *eigenvals, 
    float mu, float lambda, float gamma, 
    int siter,
    dim dimin,
    cufftHandle planbackward) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);

    // Get the dimension of the FFT
    dim dimcoefs(dimin.x, dimin.y, static_cast<int>(floor(dimin.z/2) + 1)  );

    divide_v2 <<< nBlocks, threadsPerBlock >>> (coefs_v_x, coefs_v_y, coefs_v_z,
                                                coefs_f_x, coefs_f_y, coefs_f_z,
                                                eigenvals, 
                                                mu, lambda, gamma,
                                                dimcoefs,
                                                siter);
    cudaDeviceSynchronize();

    // Inverse FFT
    cufftExecC2R(planbackward, coefs_v_x, g_x);
    cufftExecC2R(planbackward, coefs_v_y, g_y);
    cufftExecC2R(planbackward, coefs_v_z, g_z);

    // Compose coord3D object
    construct <<<nBlocks, threadsPerBlock>>> (velocity, g_x, g_y, g_z, dimin);

    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch update kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return;
}

__host__ void spectral_update(coord3D *velocity, coord3D *force, coord3D *g,
    cufftReal *g_x, cufftReal *g_y, cufftReal *g_z,
    cufftComplex *coefs_g_x, cufftComplex *coefs_g_y, cufftComplex *coefs_g_z,
    cufftHandle planforward, cufftHandle planbackward,
    float *eigenvals, 
    float mu, float lambda, float gamma, 
    int siter,
    dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimin.x*dimin.y/numThreads + (((dimin.x*dimin.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimin.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);

    // Get the dimension of the FFT
    dim dimcoefs(dimin.x, dimin.y, static_cast<int>(floor(dimin.z/2) + 1)  );

    for (int r = 0; r < siter; r++) {

        // Launch kernel to get g
        getG <<< nBlocks, threadsPerBlock >>> (g, velocity, force, gamma, dimin);

        // Decompose the components of g
        getComponentsKernel <<<nBlocks, threadsPerBlock>>> (g_x, g_y, g_z, g, dimin);

        // Calculate the FFT of the individual components
        cufftExecR2C(planforward, g_x, coefs_g_x);
        cufftExecR2C(planforward, g_y, coefs_g_y);
        cufftExecR2C(planforward, g_z, coefs_g_z);

        // Divide with eigenvalues of differential operator
        divide <<<nBlocks, threadsPerBlock>>> (coefs_g_x, coefs_g_y, coefs_g_z, eigenvals, mu, lambda, gamma, dimcoefs);

        // Inverse FFT
        cufftExecC2R(planbackward, coefs_g_x, g_x);
        cufftExecC2R(planbackward, coefs_g_y, g_y);
        cufftExecC2R(planbackward, coefs_g_z, g_z);

        // Compose coord3D object
        construct <<<nBlocks, threadsPerBlock>>> (velocity, g_x, g_y, g_z, dimin);

        // Synchronize
        cudaDeviceSynchronize();
    }

    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch update kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return;
}

__global__ void getEigenValsKernel(float *eigenvals, float gamma, float mu, float lambda, dim dimcoefs, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int q = idx / dimcoefs.x;
    int p = idx % dimcoefs.x;
    // Check that the indices are within image bounds
    if ((r < 0) || (q < 0) || (p < 0) ||
        (r > dimcoefs.z - 1) || (q > dimcoefs.y - 1) || (p > dimcoefs.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    idx = r + q*dimcoefs.z + p*dimcoefs.z*dimcoefs.y;
    // Set beta
    float beta = gamma*(mu+lambda);
    // Set the components of the diagonal matrix
    float d11 = 1.0 / (1.0 + gamma*mu*(6-2*cosf(2*PI*p/dimin.x)-2*cosf(2*PI*q/dimin.y)-2*cosf(2*PI*r/dimin.z)) + gamma*(mu+lambda)*(2-2*cosf(2*PI*p/dimin.x)+sinf(2*PI*p/dimin.x)*sinf(2*PI*p/dimin.x)));
    float d22 = 1.0 / (1.0 + gamma*mu*(6-2*cosf(2*PI*p/dimin.x)-2*cosf(2*PI*q/dimin.y)-2*cosf(2*PI*r/dimin.z)) + gamma*(mu+lambda)*(2-2*cosf(2*PI*q/dimin.y)+sinf(2*PI*q/dimin.y)*sinf(2*PI*q/dimin.y)));
    float d33 = 1.0 / (1.0 + gamma*mu*(6-2*cosf(2*PI*p/dimin.x)-2*cosf(2*PI*q/dimin.y)-2*cosf(2*PI*r/dimin.z)) + gamma*(mu+lambda)*(2-2*cosf(2*PI*r/dimin.z)+sinf(2*PI*r/dimin.z)*sinf(2*PI*r/dimin.z)));
    // Set the components of T
    float T1 = sinf(2*PI*p/dimin.x);
    float T2 = sinf(2*PI*q/dimin.y);
    float T3 = sinf(2*PI*r/dimin.z);
    // Set the denominator
    float denom = 1.0 - beta * (d11 * T1 * T1 + d22 * T2 * T2 + d33 * T3 * T3);
    // Set the components
    eigenvals[idx + 0*dimcoefs.x*dimcoefs.y*dimcoefs.z] = d11 + beta * d11 * T1 * d11 * T1 / denom; // A(1,1)
    eigenvals[idx + 1*dimcoefs.x*dimcoefs.y*dimcoefs.z] = d22 + beta * d22 * T2 * d22 * T2 / denom; // A(2,2)
    eigenvals[idx + 2*dimcoefs.x*dimcoefs.y*dimcoefs.z] = d33 + beta * d33 * T3 * d33 * T3 / denom; // A(3,3)
    eigenvals[idx + 3*dimcoefs.x*dimcoefs.y*dimcoefs.z] = beta * d11 * T1 * d22 * T2 / denom; // A(1,2)
    eigenvals[idx + 4*dimcoefs.x*dimcoefs.y*dimcoefs.z] = beta * d11 * T1 * d33 * T3 / denom; // A(1,3)
    eigenvals[idx + 5*dimcoefs.x*dimcoefs.y*dimcoefs.z] = beta * d22 * T2 * d33 * T3 / denom; // A(2,3)
}

__global__ void getEigenValsKernelv2(float *eigenvals, float gamma, float mu, float lambda, dim dimcoefs, dim dimin) {
    /// Determine the relative index of the current GPU thread
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    int q = idx / dimcoefs.x;
    int p = idx % dimcoefs.x;
    // Check that the indices are within image bounds
    if ((r < 0) || (q < 0) || (p < 0) ||
        (r > dimcoefs.z - 1) || (q > dimcoefs.y - 1) || (p > dimcoefs.x - 1)) {
        return;
    }
    // Compute the absolute index for future use
    idx = r + q*dimcoefs.z + p*dimcoefs.z*dimcoefs.y;
    // Get the center of the grid
    int centerx = floorf(dimin.x/2)+1;
    int centery = floorf(dimin.y/2)+1;
    int centerz = floorf(dimin.z/2)+1;

    if (p > centerx) {p = centerx - p;}
    if (q > centery) {q = centery - q;}

    // Get the norm of the frequency
    float normpsq = powf((float) p/ (float) dimin.x, 2.0) + powf((float) q/(float) dimin.y, 2.0) + powf((float) r/ (float) dimin.z, 2.0);
    // Set beta
    float denom1 = (float) 1.0 / (float) (1.0 + 4.0*PI*PI*gamma*mu*normpsq);
    float denom2 = (float) 1.0 / (float)  (1.0 + 4.0*PI*PI*gamma*(2*mu+lambda)*normpsq);
    // Set the eigenvals
    eigenvals[idx + 0*dimcoefs.x*dimcoefs.y*dimcoefs.z] = denom1 - denom2 * p * p / (dimin.x * dimin.x); // A(1,1)
    eigenvals[idx + 1*dimcoefs.x*dimcoefs.y*dimcoefs.z] = denom1 - denom2 * q * q / (dimin.y * dimin.y); // A(2,2)
    eigenvals[idx + 2*dimcoefs.x*dimcoefs.y*dimcoefs.z] = denom1 - denom2 * r * r / (dimin.z * dimin.z); // A(3,3)
    eigenvals[idx + 3*dimcoefs.x*dimcoefs.y*dimcoefs.z] = -denom2 * p * q / (dimin.x * dimin.y); // A(1,2)
    eigenvals[idx + 4*dimcoefs.x*dimcoefs.y*dimcoefs.z] = -denom2 * p * r / (dimin.x * dimin.z); // A(1,3)
    eigenvals[idx + 5*dimcoefs.x*dimcoefs.y*dimcoefs.z] = -denom2 * q * r / (dimin.y * dimin.z); // A(2,3)
}

__host__ void getEigenVals(float *eigenvals, float gamma, float mu, float lambda, dim dimcoefs, dim dimin) {
    // Establish the number of threads per block and the arrangement of the
    // blocks in the grid.
    int numThreads = 256;
    dim3 threadsPerBlock(numThreads, 1, 1);
    int numberOfBlocksX = dimcoefs.x*dimcoefs.y/numThreads + (((dimcoefs.x*dimcoefs.y)%numThreads==0)?0:1);
    int numberOfBlocksY = dimcoefs.z;
    dim3 nBlocks(numberOfBlocksX, numberOfBlocksY, 1);
    // Execute kernel
    getEigenValsKernel <<<nBlocks, threadsPerBlock>>> (eigenvals, gamma, mu, lambda, dimcoefs, dimin);
    // Sync
    cudaDeviceSynchronize();
    // Check for errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        mexPrintf("Failed to launch update kernel (error code: %s)!\n",
            cudaGetErrorString(err));
    }
    //Done
    return;
}