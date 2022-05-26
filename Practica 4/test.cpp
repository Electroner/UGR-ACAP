#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

using namespace std;

// N1: number of rows in the matrix A
// M1: number of columns in the matrix A
// N2: number of rows in the matrix B
// M2: number of columns in the matrix B

void MultiplicarMatrices(float *A, float *B, float *C, unsigned int N1, unsigned int M1, unsigned int N2, unsigned int M2)
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

	// Comprobar que las dimensiones de las matrices son validas
	if (N1 <= 0 || M1 <= 0 || N2 <= 0 || M2 <= 0)
	{
		printf("El tamaño de las matrices deben ser mayor que 0\n");
		return 1;
	}

	// Comprobamos que las matrices son compatibles
	if (M1 != N2)
	{
		printf("Las matrices no son compatibles\n");
		return 1;
	}

	float *A = (float *)malloc(N1 * M1 * sizeof(float));
	float *B = (float *)malloc(N2 * M2 * sizeof(float));
	float *C = (float *)malloc(N1 * M2 * sizeof(float));

	for (int i = 0; i < N1 * M1; i++)
	{
		A[i] = TrueRand(0, 10);
	}

	for (int i = 0; i < N2 * M2; i++)
	{
		B[i] = TrueRand(0, 10);
	}

	if (N1 < 7 && M1 < 7)
	{
		printf("Matriz 1:\n");
		for (int i = 0; i < N1; i++)
		{
			for (int j = 0; j < M1; j++)
			{
				printf("%.2f ", A[i * N1 + j]);
			}
			printf("\n");
		}
		printf("\nMatriz 2:\n");
		for (int i = 0; i < N2; i++)
		{
			for (int j = 0; j < M2; j++)
			{
				printf("%.2f ", B[i * M2 + j]);
			}
			printf("\n");
		}
	}

	MultiplicarMatrices(A, B, C, N1, M1, N2, M2);

	if (N1 < 7 && M1 < 7)
	{
		printf("\nMatriz resultante CPU:\n");
		for (int i = 0; i < M2; i++)
		{
			for (int j = 0; j < N1; j++)
			{
				printf("%.2f ", C[i * N1 + j]);
			}
			printf("\n");
		}
	}

	return 0;
}