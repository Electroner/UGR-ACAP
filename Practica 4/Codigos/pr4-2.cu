#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

#define MaxBlocks 256			// Maximo numero de bloques
#define MaxThreadsPerBlock 1024 // Maximo numero de threads por bloque

__global__ void getMaxGPU(int *vector, int *mayor, int _size) 
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x; 
    int offset = blockDim.x*gridDim.x; 
 
    int maximo = vector[0];
    while(tid < _size){ 
        if(vector[tid] > maximo){ 
            maximo = vector[tid]; 
        }
        tid += offset; 
    } 

    __syncthreads(); 

    atomicMax(mayor, maximo); 
}

int getMaxCPU(int *array, int size)
{
	int min = array[0];
	for (int i = 1; i < size; i++)
	{
		if (array[i] > min)
		{
			min = array[i];
		}
	}
	return min;
}

float TrueRand(float min, float max)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_real_distribution<double> distribution(min, max);

	return distribution(generator);
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Uso: <size>\n");
		printf("Size es el tamaño del vector\n");
		printf("Ejemplo: ./VectorMin 11\n");
		return 1;
	}

	int size = atoi(argv[1]);

	// Elegir la cantidad de bloques y threads por bloque dependiendo del tamaño del vector
	int Blocks;
	int Threads;
	if (size <= MaxBlocks)
	{
		Blocks = size;
		Threads = MaxThreadsPerBlock;
	}
	else
	{
		Blocks = MaxBlocks;
		Threads = MaxThreadsPerBlock;
	}

	// Mostrar los Threads y Blocks que se usaran
	printf("----- Eleccion de Threads y Blocks -----\n");
	printf("Threads: %d\n", Threads);
	printf("Blocks: %d\n", Blocks);
	printf("----------------------------------------\n\n");

	// Generamos las matrices
	int *Arr = (int *)malloc(size * sizeof(int));
	int *Arr_d;
	int sol_cpu;

	int *result = (int *)malloc(sizeof(int));
	int *result_d;

	for (int i = 0; i < size; i++)
	{
		Arr[i] = (int)TrueRand(0, 1000);
	}

	// Si el tamaño es menor que 10, imprimimos el vector
	if (size < 11)
	{
		printf("Vector:\n");
		for (int i = 0; i < size; i++)
		{
			printf("%d ", Arr[i]);
		}
		printf("\n");
	}

	float timeCPU = 0;
	// Medir Tiempo CPU
	clock_t start = clock();
	//#################################################### CPU ####################################################//
	sol_cpu = getMaxCPU(Arr, size);
	clock_t end = clock();
	timeCPU = (float)(end - start) / CLOCKS_PER_SEC;

	// Calculamos el tiempo de ejecución con cuda creado un evento
	cudaEvent_t Begining, Ending;
	cudaEventCreate(&Begining);
	cudaEventCreate(&Ending);

	// Reservamos memoria en cuda
	cudaMalloc((void **)&Arr_d, size * sizeof(float));
	cudaMalloc((void **)&result_d, sizeof(float));

	// Copiamos los datos a la GPU
	cudaMemcpy(Arr_d, Arr, size * sizeof(float), cudaMemcpyHostToDevice);

	// Calculamos el tiempo de ejecución
	cudaEventRecord(Begining);
	//#################################################### GPU ####################################################//
	getMaxGPU<<<Blocks, Threads>>>(Arr_d, result_d, size);
	cudaEventRecord(Ending);
	cudaDeviceSynchronize();

	// Copiamos los datos de la GPU a la CPU
	cudaMemcpy(result, result_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Calculamos el tiempo de ejecución
	cudaEventSynchronize(Ending);
	float timeGPU;
	cudaEventElapsedTime(&timeGPU, Begining, Ending);
	cudaDeviceSynchronize();

	// Mostramos el resultado de CPU
	printf("\nValor Resultante CPU:\n");
	printf("%d\n", sol_cpu);

	// Mostramos el resultado de GPU
	printf("\nValor Resultante GPU:\n");
	printf("%d\n", *result);

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION CPU: %.3f\n", timeCPU * 1000);

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION GPU: %.3f\n", timeGPU);

	// Liberamos memoria
	free(Arr);
	free(result);

	// Liberamos memoria Cuda
	cudaFree(Arr_d);
	cudaFree(result_d);
}