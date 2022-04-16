#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <ctype.h>
#include <math.h>

float drand(float low, float high)
{
	return ((float)rand() * (high - low)) / (float)RAND_MAX + low;
}

float **reservarMatrizFloat(int tam)
{
	float **matriz = 0;
	float *matriz_container = malloc(sizeof(float) * tam * tam);
	if (matriz_container)
	{
		matriz = malloc(sizeof(float *) * tam);
		if (matriz)
		{
			for (int i = 0; i < tam; i++)
			{
				matriz[i] = &(matriz_container[i * tam]);
			}
		}
		else
		{
			printf("Error. No se ha podido reservar la carcasa 2D de la matriz.\n");
			free(matriz_container);
		}
	}
	else
	{
		printf("Error. No se ha podido reservar memoria para la matriz.\n");
	}
	return matriz;
}

void llenarMatrizAleatoria(int tam, float **matriz, int min, int max)
{
	for (int i = 0; i < tam; i++)
	{
		for (int j = 0; j < tam; j++)
		{
			matriz[i][j] = drand(min, max);
		}
	}
}

void liberarMatriz(float **matriz)
{
	if (matriz)
	{
		free(matriz[0]);
	}
	free(matriz);
}

int main(int argc, char *argv[])
{

	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// ask for the size of a matrix
	int tam = 0;
	if (rank == 0)
	{
		printf("Ingrese el tamaño de la matriz: ");
		fflush(stdout);
		scanf("%d", &tam);
		// Mostrar el tamaño
		printf("El tamaño de la matriz es: %d\n", tam);
		fflush(stdout);

		// Reservar memoria para la matriz
		float **matriz = reservarMatrizFloat(tam);

		// iniciar la matriz con valores aleatorios
		llenarMatrizAleatoria(tam, matriz, 0, 100);

		// Localmente sumar todos los elementos de la matriz
		float suma = 0;
		for (int i = 0; i < tam; i++)
		{
			for (int j = 0; j < tam; j++)
			{
				suma += matriz[i][j];
			}
		}
		// Mostrar la suma local
		printf("La suma local es: %f\n", suma);
		fflush(stdout);
	}
	else
	{
	}

	MPI_Finalize();
	return 0;
}