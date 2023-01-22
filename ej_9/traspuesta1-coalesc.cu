/* Copiar traspuesta de matriz h_a[F][C] en matriz h_b[C][F] aunque el n.º de hebras de 
   los bloques no divida al n.º de componentes de las matrices */
// #include <stdlib.h>
#include <stdio.h>

#define F 25
#define C 43
// matriz original de F filas y C columnas
#define H 8
#define K 8
// bloques de H x K hebras (HxK<=512, cap. cpto. 1.3)

 __global__ void trspta1(int *dev_a, int *dev_b, int filas, int cols)
{ int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ((ix<cols)&&(iy<filas))
  { dev_b[ix*filas+iy] = dev_a[iy*cols+ix];}	// B[ix][iy] = A[iy][ix] (A matriz representada por dev_a, etc.)
}

__global__ void trspta_coal(int *dev_a, int *dev_b, int filas, int cols)
{
  __shared__ int tile[H][K];
 
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  
  if(ix < cols && iy < filas)
  {
    tile[threadIdx.y][threadIdx.x] = dev_a[iy*cols + ix];
  }

  ix = blockIdx.y * blockDim.x + threadIdx.x;
  iy = blockIdx.x * blockDim.y + threadIdx.y;

  if(ix < filas && iy < cols)
  {
    dev_b[iy*filas + ix] = tile[threadIdx.x][threadIdx.y];
  }
}

int main(int argc, char** argv)
{
  int h_a[F][C], h_b[C][F];
  int *d_a, *d_b;
  int i, j, aux, size = F * C * sizeof(int);
  dim3 hebrasBloque(K, H); // bloques de H x K hebras
  int numBlf = (F+H-1)/H;  // techo de F/H
  int numBlc = (C+K-1)/K;  // techo de C/K
  dim3 numBloques(numBlc,numBlf);

  // reservar espacio en el device para d_a y d_b
  cudaMalloc((void**) &d_a, size); 
  cudaMalloc((void**) &d_b, size);

  // dar valores a la matriz h_a en la CPU e imprimirlos
  printf("\nMatriz origen\n");
  for (i=0; i<F; i++) {
    for (j=0; j<C; j++) {
      aux = i*C+j;
      h_a[i][j] = aux;
      printf("%d ", aux);
    }
    printf("\n");
  }

  // copiar matriz h_a en d_a
  cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
  
  // llamar al kernel que obtiene en d_b la traspuesta de d_a
  trspta_coal<<<numBloques, hebrasBloque>>>(d_a, d_b, F, C);

  // copiar matriz d_b en h_b
  cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);
  
  // una vez que tenemos los resultados en el host, comprobamos que son correctos
  for (i=0; i<F; i++)
    for (j=0; j<C; j++) 
      if (h_a[i][j]!= h_b[j][i]) 
		{printf("error en componente %d %d de matriz de entrada \n", i,j); break;}
 
// imprimir matriz resultado
  printf("\nMatriz resultado\n");
  for (i=0; i<C; i++) {
    for (j=0; j<F; j++) {
      printf("%d ", h_b[i][j]);
    }
    printf("\n");
  }
  printf("\n");

  cudaFree(d_a); cudaFree(d_b);
  
  return 0;
} 
