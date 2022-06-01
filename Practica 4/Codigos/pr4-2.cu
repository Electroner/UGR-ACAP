#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

#define MaxBlocks 64			// Maximo numero de bloques
#define MaxThreadsPerBlock 1024 // Maximo numero de threads por bloque

//Get the minimun value of the array and assign it to result
__global__ void getMinGPU(float *_arr,float *_result,unsigned int _size){
	*_result = _arr[0];
}

float getMinCPU (float *array, int size) {
	float min = array[0];
	for (int i = 1; i < size; i++) {
		if (array[i] < min) {
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
	int Blocks = 1;
	int Threads = 1;

	// Mostrar los Threads y Blocks que se usaran
	printf("----- Eleccion de Threads y Blocks -----\n");
	printf("Threads: %d\n", Threads);
	printf("Blocks: %d\n", Blocks);
	printf("----------------------------------------\n\n");

	// Generamos las matrices
	float *Arr = (float *)malloc(size * sizeof(float));
	float *Arr_d;
	float sol_cpu;

	float *result = (float *)malloc(sizeof(float));
	float *result_d;

	for (int i = 0; i < size; i++)
	{
		Arr[i] = TrueRand(0, 100);
	}

	// Si el tamaño es menor que 10, imprimimos el vector
	if (size < 11)
	{
		printf("Vector:\n");
		for (int i = 0; i < size; i++)
		{
			printf("%.3f ", Arr[i]);
		}
		printf("\n");
	}

	float timeCPU = 0;
	// Medir Tiempo CPU
	clock_t start = clock();
	//#################################################### CPU ####################################################//
	sol_cpu = getMinCPU(Arr, size);
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
	getMinGPU<<<Blocks, Threads>>>(Arr_d,result_d, size);
	cudaEventRecord(Ending);
	cudaDeviceSynchronize();

	//Copiamos los datos de la GPU a la CPU
	cudaMemcpy(result, result_d, sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Calculamos el tiempo de ejecución
	cudaEventSynchronize(Ending);
	float timeGPU;
	cudaEventElapsedTime(&timeGPU, Begining, Ending);
	cudaDeviceSynchronize();

	// Mostramos el resultado de CPU
	printf("\nValor Resultante CPU:\n");
	printf("%f\n", sol_cpu);

	// Mostramos el resultado de GPU
	printf("\nValor Resultante GPU:\n");
	printf("%f\n", *result);

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION CPU: %f\n", timeCPU * 1000);

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION GPU: %f\n", timeGPU);

	// Liberamos memoria
	free(Arr);
	free(result);

	// Liberamos memoria Cuda
	cudaFree(Arr_d);
	cudaFree(result_d);
}