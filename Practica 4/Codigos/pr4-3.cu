#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

#define ErrorPrecision 0.00001
#define Blocks 1
#define Threads 100

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
    for (unsigned int i = 0; i < N1; i++)
    {
        for (unsigned int j = 0; j < M2; j++)
        {
            for (unsigned int k = 0; k < M1; k++)
            {
                C[i * M2 + j] += A[i * M1 + k] * B[k * M2 + j];
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
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    while(i < N1)
    {
        while(j < M2)
        {
            while(k < M1)
            {
                C[i * M2 + j] += A[i * M1 + k] * B[k * M2 + j];
                k += gridDim.z;
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
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < M1; j++)
            {
                printf("%.2f\t", A[i + j * N1]);
            }
            printf("\n");
        }
        printf("\nMatriz 2:\n");
        for (int i = 0; i < N2; i++)
        {
            for (int j = 0; j < M2; j++)
            {
                printf("%.2f\t", B[i + j * N2]);
            }
            printf("\n");
        }
    }

    //#################################################### CPU ####################################################//
    SumarMatricesCPU(A, B, C, N1, M1, N2, M2);
    //MultiplicarMatricesCPU(A, B, C, N1, M1, N2, M2);

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
    SumarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
    //MultiplicarMatricesGPU<<<Blocks, Threads>>>(A_d, B_d, C_d, N1, M1, N2, M2);
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
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < M2; j++)
            {
                printf("%.2f\t", C[i + j * N1]);
            }
            printf("\n");
        }
    }

    //Mostramos Result
    if (N1 < 7 && M1 < 7)
    {
        printf("\nResultado:\n");
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < M2; j++)
            {
                printf("%.2f\t", Result[i + j * N1]);
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
            printf("\nError en la posición %d\n", i);
            printf("%.2f\t%.2f\n", C[i], Result[i]);
            exito = false;
        }
    }
    if(exito){
        printf("\nTodos los resultados coinciden\n");
    }
}