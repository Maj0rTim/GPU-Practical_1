#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <ctime>
#include <helper_cuda.h>

#define ITER 100'000'000


/*

Monte Carlo calculation of PI - An attempt - Timothy Fischer

The following code should estimate the value of pi. It will consistently output a value around 1.57, which is about 
half the value of pi, I'm not sure why this happens. I end up just doubling the estimated value to get a vlaue close to pi.

Some of the ways I have tried to fix/optimize this code is by understanding how the threads and streaming multiprocessors 
work. The GTX 750 TI can have a max of 2048 threads running on each of the five streaming multiprocessors. 
This gives a total of 10 480 threads. If we want to generate close to 100,000,000 numbers, we need each thread to 
generate 9765 random numbers. This gives us 10480*9765=99,993,600 random numbers for the estimation. 
The block and grid size have been set to 160 x 64 respectivly to ensure we utilize the total number of threads available.

Achieved occupancy: 84.6


*/


__global__ void cudaRand(double *device_counts)
{
    int i, local_count = 0;
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    curandState state;
    curand_init(35791246, idx, 0, &state);

    for (i=0; i<9765; i++)
    {
        double x = curand_uniform_double(&state);
        double y = curand_uniform_double(&state);
        if ((x * x) + (y * y) <= 1.0) 
        {
            local_count++;
        }
    }
    
    device_counts[idx] = local_count;

}

int main(int argc, char** argv)
{
    // initialise CUDA timing
	float milli;
	cudaEvent_t start_kernel, stop_kernel, start_program, stop_program;
	cudaEventCreate(&start_kernel);
	cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_program);
	cudaEventCreate(&stop_program);

    cudaEventRecord(start_program); 

    int niter = ITER;
    size_t size = 64*160;
    double *host_counts = new double[size];

    double *device_counts;
    checkCudaErrors(cudaMalloc((void**)&device_counts, size * sizeof(double)));


    // Set Kernel Parameters
    dim3 block(64, 1);
    dim3 grid(160, 1, 1);

    // Launch Kernel
    cudaEventRecord(start_kernel);  
    checkCudaErrors(cudaDeviceSynchronize());
    cudaRand <<< block, grid >>> (device_counts);
    cudaEventRecord(stop_kernel);
    checkCudaErrors(cudaEventSynchronize(stop_kernel));
    cudaEventElapsedTime(&milli, start_kernel, stop_kernel);  

    printf("Double Generator <<<(%d,%d), (%d,%d)>>>\n", grid.x, grid.y,
        block.x, block.y);
    printf("Kernel Execution Time: %f ms\n", milli);
     // Copy final counter values from device to host
    checkCudaErrors(cudaMemcpy(host_counts, device_counts, size * sizeof(double), cudaMemcpyDeviceToHost));

    double final_count = 0.0;
    for (size_t i = 0; i < size; i++)
        final_count+=host_counts[i];
    
    double pi = (double)final_count / (niter) * 4.0;

    cudaEventRecord(stop_program);
    checkCudaErrors(cudaEventSynchronize(stop_program));
    cudaEventElapsedTime(&milli, start_program, stop_program);  
    printf("Total Program Time: %f ms\n", milli);
    printf("Estimate for PI: %g\n", pi);

    checkCudaErrors(cudaFree(device_counts));
    delete[] host_counts;

    return 0;
}