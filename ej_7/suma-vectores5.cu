#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 65535*512+1000
#define numThreads 512
#define numBlocks 65535

__device__ int DA[N];
__device__  int DB[N];
__device__ int DC[N];
__device__ int stopKernel = 0;

__global__ void VecAdd(int elPerThread)
{
    for (int i = blockIdx.x*blockDim.x + threadIdx.x*elPerThread; i<elPerThread && i<N; i+=1)
      DC[i] = DA[i] + DB[i];
}

__global__ void initVec(int elPerThread)
{
      for (int i = blockIdx.x*blockDim.x + threadIdx.x*elPerThread; i<elPerThread && i<N; i+=1)
      {  
        DA[i] = -i;
        DB[i] = 3*i;
      }
}

__global__ void checkAdd(int elPerThread)
{
    if(stopKernel == 0)
    {
      for (int i = blockIdx.x*blockDim.x + threadIdx.x*elPerThread; i<elPerThread && i<N; i+=1)
      {  
        if (DC[i]!= (DA[i]+DB[i]))
        {
          printf("error en componente %d\n", i);
          stopKernel = 1;
        }
      }
    }
}

int main()
{ 
  int elPerThread  = round(N/(numBlocks*numThreads));

  cudaError_t err = cudaMalloc((void **)&elPerThread, sizeof(int));
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

  // iniciamos los valores de los vectores DA y DB
  initVec <<<numBlocks, numThreads>>>(elPerThread);

  // llamamos al kernel
  VecAdd <<<numBlocks, numThreads>>>(elPerThread);	// N hilos ejecutan el kernel en paralelo
  
  // comprobamos si la suma en DC ha sido correcta
  checkAdd <<<numBlocks, numThreads>>>(elPerThread);
    
  return 0;
} 
