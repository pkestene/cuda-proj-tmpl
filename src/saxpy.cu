/**
 * Compute saxpy 
 * - on CPU : serial and OpenMP version
 * - on GPU : first using CUDA, then library CuBLAS
 *
 * compare timings.
 *
 */

// =========================
// standard imports
// =========================
#include <stdio.h>
#include <stdlib.h>

// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>
#include <cublas.h>

// =========================
// our imports
// =========================
#include "my_cuda_utils.h"
#include "SimpleTimer.h"
#include "CudaTimer.h"

// =========================
// global variables and configuration section
// =========================

// problem size (vector length) N
//static int N = 1234567;
static int N = 1<<22;

// number of repetitions of the timing loop
// (CPU timers only have a ~ms resolution)
static int numTimingReps = 100;


// =========================
// kernel function (CPU)
// =========================
void saxpy_serial(int n, float alpha, float *x, float *y)
{
  int i;
  for (i=0; i<n; i++)
    y[i] = alpha*x[i] + y[i];
}


// =========================
// kernel function (CUDA device)
// =========================
__global__ void saxpy_cuda(int n, float alpha, float *x, float *y)
{
  // compute the global index in the vector from
  // the number of the current block, blockIdx,
  // the number of threads per block, blockDim,
  // and the number of the current thread within the block, threadIdx
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // except for special cases, the total number of threads in all blocks
  // adds up to more than the vector length n, so this conditional is
  // EXTREMELY important to avoid writing past the allocated memory for
  // the vector y.
  if (i<n)
    y[i] = alpha*x[i] + y[i];
}


// =========================
// main routine
// =========================
int main (int argc, char **argv)
{
  SimpleTimer cpuTimer;
  CudaTimer   gpuTimer;

  // =========================
  // (1) initialisations:
  //     implemented in tools.c
  //     '0' is the device to use, see lesson0
  // =========================
  initCuda(0);

  
  // =========================
  // (2) allocate memory on host (main CPU memory) and device,
  //     h_ denotes data residing on the host, d_ on device
  // =========================
  float *h_x = (float*)malloc(N*sizeof(float));
  float *h_y = (float*)malloc(N*sizeof(float));
  float *d_x;
  cudaMalloc((void**)&d_x, N*sizeof(float));
  float *d_y;
  cudaMalloc((void**)&d_y, N*sizeof(float));
  checkErrors("memory allocation");


  // =========================
  // (3) initialise data on the CPU
  // =========================
  int i;
  for (i=0; i<N; i++)
  {
    h_x[i] = 1.0f + i;
    h_y[i] = (float)(N-i+1);
  }


  // =========================
  // (4) copy data to device
  // =========================
  cudaMemcpy(d_x, h_x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y, N*sizeof(float), cudaMemcpyHostToDevice);
  checkErrors("copy data to device");

  
  // =========================
  // (5) perform computation on host
  //     use our straight forward code
  //     and our utility functions to time everything,
  //     note that gettimeofday has ~ms resolution, so
  //     perform everything in a loop to minimise
  //     timing noise
  // =========================
  float alpha = 2.0;
  int iter;
  cpuTimer.start();
  for (iter=0; iter<numTimingReps; iter++)
    saxpy_serial(N, alpha, h_x, h_y);
  cpuTimer.stop();
  double elapsed = cpuTimer.elapsed();
  printf("OUR CPU CODE: %8d elements, %10.6f ms per iteration, %6.3f GFLOP/s, %7.3f GB/s\n",
         N,
         (elapsed*1000.0)/(double)numTimingReps,
         2.0*N*numTimingReps / (elapsed*1e9),
         3.0*N*sizeof(float)*numTimingReps / (elapsed*1e9) );

  
  // =========================
  // (7) perform computation on device, our implementation
  //     use CUDA events to time the execution:
  //     (a) insert "tag" into instruction stream
  //     (b) execute kernel
  //     (c) insert another tag into instruction stream
  //     (d) synchronize (ie, wait for) this tag (event)
  //     CUDA events have a resolution of ~0.5us
  // =========================
  float time;

  // Mapping onto the device:
  // - each thread computes one element of the output array in situ
  // - all threads and blocks are independent
  // - use 256 threads per block
  // - use as many blocks as necessary (the last block is not entirely
  //   full if n is not a multiple of 256)
  int numThreadsPerBlock = 128;
  int numBlocks = (N+numThreadsPerBlock-1) / numThreadsPerBlock;

  gpuTimer.start();
  saxpy_cuda<<<numBlocks, numThreadsPerBlock>>>(N, alpha, d_x, d_y);  
  gpuTimer.stop();
  time = gpuTimer.elapsed();
  printf("OUR GPU CODE: %8d elements, %10.6f ms per iteration, %6.3f GFLOP/s, %7.3f GB/s\n",
         N,
         time*1000,
         2.0*N / (time*1e9),
         3.0*N*sizeof(float) / (time*1e9) );

  
  // =========================
  // (8) read back result from device into temp vector
  // =========================
  float *h_z = (float*)malloc(N*sizeof(float));
  cudaMemcpy(h_z, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  checkErrors("copy data from device");

  
  // =========================
  // (9) perform computation on device, CUBLAS
  // =========================
  gpuTimer.reset();
  gpuTimer.start();
  cublasSaxpy(N, alpha, d_x, 1, d_y, 1);
  gpuTimer.stop();
  time = gpuTimer.elapsed();
  printf("CUBLAS      : %8d elements, %10.6f ms per iteration, %6.3f GFLOP/s, %7.3f GB/s\n",
         N,
         time*1000,
         2.0*N / (time*1e9),
         3.0*N*sizeof(float) / (time*1e9) );
  

  // =========================
  // (10) perform result comparison
  //      we need to re-run the CPU code because
  //      it has been executed 1000 times before
  // =========================
  int errorCount = 0;
  for (i=0; i<N; i++)
  {
    h_x[i] = 1.0f + i;
    h_y[i] = (float)(N-i+1);
  }
  saxpy_serial(N, alpha, h_x, h_y);
  for (i=0; i<N; i++) 
  {
    if (abs(h_y[i]-h_z[i]) > 1e-6)
      errorCount = errorCount + 1;
  }
  if (errorCount > 0)
    printf("Result comparison failed.\n");
  else
    printf("Result comparison passed.\n");

  

  // =========================
  // (11) clean up, free memory
  // =========================
  free(h_x);
  free(h_y);
  free(h_z);
  cudaFree(d_x);
  cudaFree(d_y);

  return 0;
}
