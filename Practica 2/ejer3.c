#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

double drand(double low, double high)
{
    return ((double)rand() * (high - low)) / (double)RAND_MAX + low;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4)
    {
        printf("Error: Se necesitan 4 procesos\n");
        return 1;
    }

    MPI_Finalize();
    return 0;
}