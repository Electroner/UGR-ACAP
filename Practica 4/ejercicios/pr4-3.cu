#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

#define ErrorPrecision 1 		//Decimales de error
#define MaxBlocks 64			//Maximo numero de bloques
#define MaxThreadsPerBlock 1024	//Maximo numero de threads por bloque

void SumarMatricesCPU(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
	for (unsigned int i = 0; i < N1; i++)
	{
		for (unsigned int j = 0; j < M1; j++)
		{
			C[i * M1 + j] = A[i * M1 + j] + B[i * M1 + j];
		}
	}
}

void MultiplicarMatricesCPU(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
	//Empty C
	for(unsigned int i = 0; i < N1*M2; i++)
	{
		C[i] = 0;
	}

	for (unsigned int i = 0; i < N1; i++)
	{
		for (unsigned int j = 0; j < M2; j++)
		{
			for (unsigned int k = 0; k < M1; k++)
			{
				C[i*M2 + j] += A[i*M1 + k] * B[k*M2 + j];
			}
		}
	}
}

__global__ void SumarMatricesGPU(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	while (i < N1)
	{
		while (j < M1)
		{
			C[i * M1 + j] = A[i * M1 + j] + B[i * M1 + j];
			j += gridDim.y;
		}
		i += gridDim.x;
	}
}

__global__ void MultiplicarMatricesGPU(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

	while (i < N1)
	{
		while (j < M2)
		{
			for (unsigned int k = 0; k < M1; k++)
			{
				C[i*M2 + j] += A[i*M1 + k] * B[k*M2 + j];
			}
			j += gridDim.y;
		}
		i += gridDim.x;
	}
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
	if (argc != 5)
	{
		printf("Uso: <N1> <M1> <N2> <M2>\n");
		printf("N y M son las dimensiones de las matrices 1 y 2 respectivamente\n");
		printf("Ejemplo: ./matrix_multiply 4 5 5 4\n");
		return 1;
	}

	int N1 = atoi(argv[1]);
	int M1 = atoi(argv[2]);
	int N2 = atoi(argv[3]);
	int M2 = atoi(argv[4]);

	//Elegir la cantidad de bloques y threads por bloque dependiendo del tama??o de la matriz
	int Blocks;
	int Threads;
	if (N1 * M1 < MaxThreadsPerBlock)
	{
		Blocks = 1;
		Threads = N1 * M1;
	}
	else
	{
		if(Blocks = N1 * M1 / MaxThreadsPerBlock % MaxBlocks == 0)
		{
			Blocks = N1 * M1 / MaxThreadsPerBlock;
			Threads = MaxThreadsPerBlock;
		}
		else
		{
			Blocks = N1 * M1 / MaxThreadsPerBlock + 1;
			Threads = N1 * M1 % MaxThreadsPerBlock;
		}
	}


	// Mostrar los Threads y Blocks que se usaran
	printf("----- Eleccion de Threads y Blocks -----\n");
	printf("Threads: %d\n", Threads);
	printf("Blocks: %d\n", Blocks);
	printf("----------------------------------------\n\n");

	// Comprobar que las dimensiones de las matrices son validas
	if (N1 <= 0 || M1 <= 0 || N2 <= 0 || M2 <= 0)
	{
		printf("El tama??o de las matrices deben ser mayor que 0\n");
		return 1;
	}

	// Comprobamos que las matrices son compatibles
	if (M1 != N2)
	{
		printf("Las matrices no son compatibles\n");
		return 1;
	}

	// Generamos las matrices
	float *A = (float *)malloc(N1 * M1 * sizeof(float));
	float *B = (float *)malloc(N2 * M2 * sizeof(float));
	float *C = (float *)malloc(N1 * M2 * sizeof(float));

	float *A_d;
	float *B_d;
	float *C_d;

	float *Result = (float *)malloc(N1 * M2 * sizeof(float));

	for (int i = 0; i < N1 * M1; i++)
	{
		A[i] = TrueRand(0, 100);
	}

	for (int i = 0; i < N2 * M2; i++)
	{
		B[i] = TrueRand(0, 100);
	}

	// Si el tama??o es menor que 7, imprimimos la matriz
	if (N1 < 7 && M1 < 7)
	{
		printf("Matriz 1:\n");
		for (int i = 0; i < N1; i++)
		{
			for (int j = 0; j < M1; j++)
			{
				printf("%.3f\t", A[i * N1 + j]);
			}
			printf("\n");
		}
		printf("\nMatriz 2:\n");
		for (int i = 0; i < N2; i++)
		{
			for (int j = 0; j < M2; j++)
			{
				printf("%.3f\t", B[i * M2 + j]);
			}
			printf("\n");
		}
	}

	float timeCPU = 0;
	//Medir Tiempo CPU
	clock_t start = clock();
	//#################################################### CPU ####################################################//
	// SumarMatricesCPU(A, B, C, N1, M1, N2, M2);
	MultiplicarMatricesCPU(A, B, C, N1, M1, N2, M2);
	clock_t end = clock();
	timeCPU = (float)(end - start) / CLOCKS_PER_SEC;

	// Calculamos el tiempo de ejecuci??n con cuda creado un evento
	cudaEvent_t Begining, Ending;
	cudaEventCreate(&Begining);
	cudaEventCreate(&Ending);

	// Reservamos memoria en cuda
	cudaMalloc((void **)&A_d, N1 * M1 * sizeof(float));
	cudaMalloc((void **)&B_d, N2 * M2 * sizeof(float));
	cudaMalloc((void **)&C_d, N1 * M2 * sizeof(float));

	// Copiamos los datos a la GPU
	cudaMemcpy(A_d, A, N1 * M1 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, N2 * M2 * sizeof(float), cudaMemcpyHostToDevice);

	// Calculamos el tiempo de ejecuci??n
	cudaEventRecord(Begining);
	//#################################################### GPU ####################################################//
	// SumarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
	MultiplicarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
	cudaEventRecord(Ending);
	cudaDeviceSynchronize();

	// Copiamos los datos de la GPU a la CPU
	cudaMemcpy(Result, C_d, N1 * M2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Calculamos el tiempo de ejecuci??n
	cudaEventSynchronize(Ending);
	float timeGPU;
	cudaEventElapsedTime(&timeGPU, Begining, Ending);
	cudaDeviceSynchronize();

	// Si el tama??o es menor que 7, imprimimos la matriz C
	if (N1 < 7 && M1 < 7)
	{
		printf("\nMatriz resultante CPU:\n");
		for (int i = 0; i < M2; i++)
		{
			for (int j = 0; j < N1; j++)
			{
				printf("%.3f\t", C[i * N1 + j]);
			}
			printf("\n");
		}
	}

	// Mostramos Result
	if (N1 < 7 && M1 < 7)
	{
		printf("\nMatriz resultante GPU:\n");
		for (int i = 0; i < M2; i++)
		{
			for (int j = 0; j < N1; j++)
			{
				printf("%.3f\t", Result[i * N1 + j]);
			}
			printf("\n");
		}
	}

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION CPU: %f\n", timeCPU*1000);

	// Imprimimos los resultados
	printf("\nTIEMPO DE EJECUCION GPU: %f\n", timeGPU);

	bool exito = true;
	// Comparamos los resultados
	for (int i = 0; i < N1 * M2; i++)
	{
		if (abs(C[i] - Result[i]) > ErrorPrecision)
		{
			// With i get the row and column of the C matrix
			int row = i / N1;
			int col = i % N1;
			printf("\nError en la posicion C[%d][%d]\n", row, col);
			printf("%.3f != %.3f\n", C[i], Result[i]);
			exito = false;
		}
	}
	if (exito)
	{
		printf("\nTodos los resultados coinciden\n");
	}

	// Liberamos memoria
	free(A);
	free(B);
	free(C);
	free(Result);

	// Liberamos memoria Cuda
	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}