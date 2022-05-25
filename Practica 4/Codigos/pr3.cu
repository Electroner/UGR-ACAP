#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

__global__ void SumarMatrices(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
{
    for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N1; i += blockDim.x * gridDim.x)
    {
        for(unsigned int j = blockIdx.y * blockDim.y + threadIdx.y; j < M1; j += blockDim.y * gridDim.y)
        {
            C[i * M2 + j] = 0;
            for(unsigned int k = 0; k < N2; k++)
            {
                C[i * M2 + j] += A[i * N2 + k] * B[k * M2 + j];
            }
        }
    }
}

float drand(float low, float high)
{
    return ((float)rand() * (high - low)) / (float)RAND_MAX + low;
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

    for (int i = 0; i < N1 * M1; i++)
    {
        A[i] = drand(0,1000);

    }

    for (int i = 0; i < N2 * M2; i++)
    {
        B[i] = drand(0, 1000);
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

    //Calculamos el tiempo de ejecución con cuda creado un evento
    cudaEvent_t Begining, Ending;
    cudaEventCreate(&Begining);
    cudaEventCreate(&Ending);

    //Reservamos memoria en cuda
    cudaMalloc((void **)&A, N1 * M1 * sizeof(float));
    cudaMalloc((void **)&B, N2 * M2 * sizeof(float));
    cudaMalloc((void **)&C, N1 * M2 * sizeof(float));

    //Copiamos los datos a la GPU
    cudaMemcpy(A, A, N1 * M1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, B, N2 * M2 * sizeof(float), cudaMemcpyHostToDevice);

    //Calculamos el tiempo de ejecución
    cudaEventRecord(Begining);
    SumarMatrices<<<1, 1>>>(A, B, C, N1, M1, N2, M2);
    cudaEventRecord(Ending);

    //Copiamos los datos de la GPU a la CPU
    cudaMemcpy(C, C, N1 * M2 * sizeof(float), cudaMemcpyDeviceToHost);

    //Calculamos el tiempo de ejecución
    cudaEventSynchronize(Ending);
    float time;
    cudaEventElapsedTime(&time, Begining, Ending);

    //Imprimimos los resultados
    printf("Tiempo de ejecución: %f\n", time);

    //Si el tamaño es menor que 7, imprimimos la matriz C
    if (N1 < 7 && M1 < 7)
    {
        printf("Matriz resultante:\n");
        for (int i = 0; i < N1; i++)
        {
            for (int j = 0; j < M2; j++)
            {
                printf("%.2f\t", C[i + j * N1]);
            }
            printf("\n");
        }
    }
}