#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define SIZE 1000

int main(int argc, char *argv[])
{
	int rank, size;
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int sumaIngresos = 0;
	int sumaGastos = 0;

	if (rank == 0)
	{
		int sumaIngresosPROC0 = 0;
		int sumaGastosPROC0 = 0;
		int *ingresos = malloc(sizeof(int) * SIZE);
		int *gastos = malloc(sizeof(int) * SIZE);
		
		// inicializar el vector ingresos y gastos de forma aleatoria
		srand(time(NULL));
		for (int i = 0; i < SIZE; i++)
		{
			ingresos[i] = (rand() % 1000);
			gastos[i] = (-(rand() % 1000));
			sumaIngresosPROC0 += ingresos[i];
			sumaGastosPROC0 += gastos[i];
		}
		printf("Solucion desde PROC0: %d\n", sumaIngresosPROC0 + sumaGastosPROC0);

		// enviar la parte de los ingresos y gastos a cada proceso
		for (int i = 1; i < size; i++)
		{
			MPI_Send(&ingresos[i * SIZE / size], SIZE / size, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&gastos[i * SIZE / size], SIZE / size, MPI_INT, i, 0, MPI_COMM_WORLD);
		}

		// recibir el resultado de cada proceso
		int localSumaIngresos = 0;
		int localSumaGastos = 0;
		for (int i = 1; i < size; i++)
		{
			MPI_Recv(&localSumaIngresos, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(&localSumaGastos, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			sumaIngresos += localSumaIngresos;
			sumaGastos += localSumaGastos;
		}
		//Parte correspondiente al proceso 0
		for (int i = 0; i < SIZE/size; i++)
		{
			sumaIngresos += ingresos[i];
			sumaGastos += gastos[i];
		}

		printf("Solucion general: %d\n", sumaIngresos + sumaGastos);
	}
	else
	{
		// recibir la parte de los ingresos y gastos de cada proceso
		int *ingresos = malloc(sizeof(int) * SIZE / size);
		int *gastos = malloc(sizeof(int) * SIZE / size);
		MPI_Recv(ingresos, SIZE / size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(gastos, SIZE / size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for (int i = 0; i < (SIZE / size); i++)
		{
			sumaIngresos += ingresos[i];
			sumaGastos += gastos[i];
		}

		// enviar el resultado a PROC0
		MPI_Send(&sumaIngresos, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		MPI_Send(&sumaGastos, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}