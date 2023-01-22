#include <stdio.h>

#define N 600

__global__ void VecAdd(int* DA, int* DB, int* DC)
{
	// int i = threadIdx.x;
  int i = threadIdx.x;
    DC[i] = DA[i] + DB[i];
}

int main()
{ int HA[N], HB[N], HC[N];
  int *DA, *DB, *DC;
  int i; int size = N*sizeof(int);
  
  // reservamos espacio en la memoria global del device
  cudaError_t err_DA = cudaMallocHost((void**)&DA, size);
  cudaError_t err_DB = cudaMallocHost((void**)&DB, size);
  cudaError_t err_DC = cudaMallocHost((void**)&DC, size);
  
  if (err_DA != cudaSuccess) printf("%s\n", cudaGetErrorString(err_DA));
  if (err_DB != cudaSuccess) printf("%s\n", cudaGetErrorString(err_DB));
  if (err_DC != cudaSuccess) printf("%s\n", cudaGetErrorString(err_DC));

  // inicializamos HA y HB
  for (i=0; i<N; i++) {HA[i]=-i; HB[i] = 3*i;}
  
  // copiamos HA y HB del host a DA y DB en el device, respectivamente
  cudaError_t err_DAHA = cudaMemcpy(DA, HA, size, cudaMemcpyHostToDevice);
  cudaError_t err_DBHB = cudaMemcpy(DB, HB, size, cudaMemcpyHostToDevice);
  
  if (err_DAHA != cudaSuccess) printf("%s\n", cudaGetErrorString(err_DAHA));
  if (err_DBHB != cudaSuccess) printf("%s\n", cudaGetErrorString(err_DBHB));

  // llamamos al kernel (1 bloque de N hilos)
  VecAdd <<<1, N>>>(DA, DB, DC);	// N hilos ejecutan el kernel en paralelo
  
  cudaError_t errSync = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();

  if(errSync != cudaSuccess) printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if(errAsync != cudaSuccess) printf("Async kernel error: %s\n", cudaGetErrorString(cudaGetLastError()));

  // copiamos el resultado, que está en la memoria global del device, (DC) al host (a HC)
  cudaError_t err_HCDC = cudaMemcpy(HC, DC, size, cudaMemcpyDeviceToHost);
  
  if (err_HCDC != cudaSuccess) printf("%s\n", cudaGetErrorString(err_HCDC));

  // liberamos la memoria reservada en el device
  cudaFreeHost(DA); cudaFreeHost(DB); cudaFreeHost(DC);
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  // esta comprobación debe quitarse una vez que el programa es correcto (p. ej., para medir el tiempo de ejecución)
  for (i = 0; i < N; i++) // printf("%d + %d = %d\n",HA[i],HB[i],HC[i]);
    if (HC[i]!= (HA[i]+HB[i])) 
		{printf("error en componente %d\n", i); break;}
    
  return 0;
} 
