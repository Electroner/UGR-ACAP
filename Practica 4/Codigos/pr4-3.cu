#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <cublas_v2.h>

#define ErrorPrecision 1
#define MAXThreadsPerBlock 512

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
    for (unsigned int i = 0; i < M1; i++)
	{
		for (unsigned int j = 0; j < N2; j++)
		{
			C[i*N2 + j] = 0;
			for (unsigned int k = 0; k < N1; k++)
			{
				C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
			}
		}
	}
}

__global__ void SumarMatricesGPU(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    while(i < N1)
    {
        while(j < M1)
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

    while(i < M1)
    {
        while(j < N2)
        {
            C[i*N2 + j] = 0;
            for (unsigned int k = 0; k < N1; k++)
            {
                C[i * N2 + j] += A[i * N1 + k] * B[k * N2 + j];
            }
            j += gridDim.y;
        }
        i += gridDim.x;
    }
}

float TrueRand(float min, float max) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::uniform_real_distribution<double> distribution (min,max);

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

    //Con los valores de N y M elegir que valores de Threads y Blocks se usaran
    //Un Thread por elemento final de la matriz
    //Si el numero de elementos es mayor que el numero de threads por bloque, se usaran mas bloques
    //Si el numero de elementos es menor que el numero de threads por bloque, se usaran menos bloques
    //Si el numero de elementos es multiplo del numero de threads por bloque, se usaran el numero de bloques exacto
    int Threads;
    int Blocks;
    if(N1 * M1 < MAXThreadsPerBlock)
    {
        Threads = N1 * M1;
        Blocks = 1;
    }
    else
    {
        Threads = MAXThreadsPerBlock;
        Blocks = (N1 * M1) / MAXThreadsPerBlock;
        if ((N1 * M1) % MAXThreadsPerBlock != 0)
        {
            Blocks++;
        }
    }
    
    //Mostrar los Threads y Blocks que se usaran
    printf("----- Eleccion de Threads y Blocks -----\n");
    printf("Threads: %d\n", Threads);
    printf("Blocks: %d\n", Blocks);
    printf("----------------------------------------\n\n");

    //Comprobar que las dimensiones de las matrices son validas
    if (N1 <= 0 || M1 <= 0 || N2 <= 0 || M2 <= 0)
    {
        printf("El tamaño de las matrices deben ser mayor que 0\n");
        return 1;
    }

    //Comprobamos que las matrices son compatibles
    if (M1 != N2)
    {
        printf("Las matrices no son compatibles\n");
        return 1;
    }

    //Generamos las matrices
    float *A = (float *)malloc(N1 * M1 * sizeof(float));
    float *B = (float *)malloc(N2 * M2 * sizeof(float));
    float *C = (float *)malloc(N1 * M2 * sizeof(float));

    float *A_d = (float *)malloc(N1 * M1 * sizeof(float));
    float *B_d = (float *)malloc(N2 * M2 * sizeof(float));
    float *C_d = (float *)malloc(N1 * M2 * sizeof(float));

    float *Result = (float *)malloc(N1 * M2 * sizeof(float));

    for (int i = 0; i < N1 * M1; i++)
    {
        A[i] = TrueRand(0,1000);

    }

    for (int i = 0; i < N2 * M2; i++)
    {
        B[i] = TrueRand(0, 1000);
    }

    //Si el tamaño es menor que 7, imprimimos la matriz
    if (N1 < 7 && M1 < 7)
    {
        printf("Matriz 1:\n");
		for (int i = 0; i < M1; i++)
		{
			for (int j = 0; j < N1; j++)
			{
				printf("%.2f\t", A[i * N1 + j]);
			}
			printf("\n");
		}
		printf("\nMatriz 2:\n");
		for (int i = 0; i < M2; i++)
		{
			for (int j = 0; j < N2; j++)
			{
				printf("%.2f\t", B[i * N2 + j]);
			}
			printf("\n");
		}
    }

    //#################################################### CPU ####################################################//
    //SumarMatricesCPU(A, B, C, N1, M1, N2, M2);
    MultiplicarMatricesCPU(A, B, C, N1, M1, N2, M2);

    //Calculamos el tiempo de ejecución con cuda creado un evento
    cudaEvent_t Begining, Ending;
    cudaEventCreate(&Begining);
    cudaEventCreate(&Ending);

    //Reservamos memoria en cuda
    cudaMalloc((void **)&A_d, N1 * M1 * sizeof(float));
    cudaMalloc((void **)&B_d, N2 * M2 * sizeof(float));
    cudaMalloc((void **)&C_d, N1 * M2 * sizeof(float));

    //Copiamos los datos a la GPU
    cudaMemcpy(A_d, A, N1 * M1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, N2 * M2 * sizeof(float), cudaMemcpyHostToDevice);

    //Calculamos el tiempo de ejecución
    cudaEventRecord(Begining);
    //#################################################### GPU ####################################################//
    //SumarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
    MultiplicarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
    cudaEventRecord(Ending);
    cudaDeviceSynchronize();

    //Copiamos los datos de la GPU a la CPU
    cudaMemcpy(Result, C_d, N1 * M2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //Calculamos el tiempo de ejecución
    cudaEventSynchronize(Ending);
    float time;
    cudaEventElapsedTime(&time, Begining, Ending);

    //Si el tamaño es menor que 7, imprimimos la matriz C
    if (N1 < 7 && M1 < 7)
	{
		printf("\nMatriz resultante CPU:\n");
		for (int i = 0; i < M2; i++)
		{
			for (int j = 0; j < N1; j++)
			{
				printf("%.2f\t", C[i * N1 + j]);
			}
			printf("\n");
		}
	}

    //Mostramos Result
    if (N1 < 7 && M1 < 7)
    {
        printf("\nResultado:\n");
        for (int i = 0; i < M2; i++)
        {
            for (int j = 0; j < N1; j++)
            {
                printf("%.2f\t", Result[i * N1 + j]);
            }
            printf("\n");
        }
    }

    //Imprimimos los resultados
    printf("\nTIEMPO DE EJECUCION: %f\n", time);

    bool exito = true;
    //Comparamos los resultados
    for (int i = 0; i < N1 * M2; i++)
    {
        if (abs(C[i] - Result[i]) > ErrorPrecision)
        {
            //With i get the row and column of the C matrix
            int row = i / N1;
            int col = i % N1;
            printf("\nError en la posicion C[%d][%d]\n", row, col);
            printf("%.2f != %.2f\n", C[i], Result[i]);
            exito = false;
        }
    }
    if(exito){
        printf("\nTodos los resultados coinciden\n");
    }
}