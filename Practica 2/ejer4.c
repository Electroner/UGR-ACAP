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
	int tam = 0;
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if (rank == 0)
	{
		printf("Ingrese el tamaño de la matriz: ");
		fflush(stdout);
		scanf("%d", &tam);
		// Mostrar el tamaño
		printf("El tamaño de la matriz es: %d\n", tam);
		fflush(stdout);

		//Enviamos el tamaño a todos los procesos
		MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

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

		//Mandamos a cada proceso la cantidad de filas que le coresponden
		int filas_por_proceso = tam / size;

		// Mandamos la cantidad de filas que le corresponde a cada proceso
		MPI_Bcast(&filas_por_proceso, 1, MPI_INT, 0, MPI_COMM_WORLD);

		printf("El proceso %d tiene %d filas\n", rank, filas_por_proceso);

		// Mandamos fila por fila la matriz
		for(int i = 1; i < size; i++)
		{
			for(int j = 0; j < filas_por_proceso; j++)
			{
				MPI_Send(&matriz[i * filas_por_proceso + j][0], tam, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
			}
		}

		//Calculamos la parte correspondiente al proceso 0
		float suma_local = 0;
		for(int i = 0; i < filas_por_proceso; i++)
		{
			for(int j = 0; j < tam; j++)
			{
				suma_local += matriz[i][j];
			}
		}

		// Recibimos la suma de cada proceso
		float suma_proceso = 0;
		float suma_total = 0;
		for(int i = 1; i < size; i++)
		{
			MPI_Recv(&suma_proceso, 1, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			printf("La suma del proceso %d es: %f\n", i, suma_proceso);
			fflush(stdout);
			suma_total += suma_proceso;
		}

		// Mostrar la suma total
		printf("La suma total es: %f\n", suma_total+suma_local);

		// Liberar memoria
		liberarMatriz(matriz);
	}
	else
	{
		// Recibimos el tamaño
		MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Recibimos la cantidad de filas que le corresponde a cada proceso
		int filas_por_proceso = 0;
		MPI_Bcast(&filas_por_proceso, 1, MPI_INT, 0, MPI_COMM_WORLD);

		// Reservar un vector donde meter los numeros
		float *vector = malloc(sizeof(float) * filas_por_proceso * tam);

		// Recibir las filas y meterlas en el vector
		for(int i = 0; i < filas_por_proceso; i++)
		{
			MPI_Recv(&vector[i * tam], tam, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}

		// Localmente sumar todos los elementos de la matriz
		float suma = 0;
		for (int i = 0; i < filas_por_proceso; i++)
		{
			for (int j = 0; j < tam; j++)
			{
				suma += vector[i * tam + j];
			}
		}

		// Mandar la suma local
		MPI_Send(&suma, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		free(vector);
	}

	MPI_Finalize();
	return 0;
}