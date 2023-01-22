#include <stdio.h>

#define N 1000000

__device__ int DA[N];
__device__  int DB[N];
__device__ int DC[N];
__device__ int stopKernel = 0;

__global__ void VecAdd()
{
	// int i = threadIdx.x;
  // for(int i=0; i<N; i++)
  // int i = blockIdx.x;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<N)  DC[i] = DA[i] + DB[i];
}

__global__ void initVec()
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    DA[i] = -i;
    DB[i] = 3*i;
}

__global__ void checkAdd()
{
    if(stopKernel == 0)
    {
      int i = blockIdx.x*blockDim.x + threadIdx.x;
      if (DC[i]!= (DA[i]+DB[i]))
      {
        printf("error en componente %d\n", i);
        stopKernel = 1;
      }
    }
}

int main()
{ 
  // iniciamos los valores de los vectores DA y DB
  initVec <<<(N+255)/256, 256>>>();

  // llamamos al kernel
  VecAdd <<<(N+255)/256, 256>>>();	// N hilos ejecutan el kernel en paralelo
  
  // comprobamos si la suma en DC ha sido correcta
  checkAdd <<<(N+255)/256, 256>>>();
    
  return 0;
} 
