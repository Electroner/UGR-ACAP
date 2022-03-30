#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#define RANDMIN 0
#define RANDMAX 100

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

    if (argc < 2)
    {
        printf("No se ha de expecificar el tamaño del vector\n");
        return 1;
    }

    int n = atoi(argv[1]);
    if (rank == 0)
    {
        printf("Tamaño del vector: %d\n", n);
    }

    if (size < 2)
    {
        printf("Error: Se necesitan al menos 2 procesos\n");
        return 1;
    }

    if (rank == 0)
    {
        double *vector = malloc(sizeof(double) * n);
        double minimo = RANDMAX;
        // generar un vector de numeros aleatorios
        srand(time(NULL));
        for (int i = 0; i < n; i++)
        {
            vector[i] = drand(RANDMIN, RANDMAX);
            //printf("Vector[%d]: %f\n", i, vector[i]);
            if (vector[i] < minimo)
            {
                minimo = vector[i];
            }
        }
        printf("Minimo Secuencial: %f\n", minimo);

        // Espera a que el proceso uno pida el vector para mandarselo
        int flag = 0;
        MPI_Status status;
        MPI_Recv(&flag, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &status);

        if (flag == 1)
        {
            MPI_Send(vector, n, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        }

        // Recibe la parte del vector que le corresponde
        double *vector_local = malloc(sizeof(double) * n / size);
        MPI_Recv(vector_local, n / size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);

        // Calcula el minimo local
        double minimo_local = RANDMAX;
        for (int i = 0; i < n / size; i++)
        {
            if (vector_local[i] < minimo_local)
            {
                minimo_local = vector_local[i];
            }
        }

        // Envia el minimo local
        MPI_Send(&minimo_local, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        printf("Minimo local , Proceso 0: %f\n", minimo_local);
    }
    else if (rank == 1)
    {
        // Pide el vector al proceso 0
        int flag = 1;
        MPI_Send(&flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Recibe el vector
        double *vector = malloc(sizeof(double) * n);
        MPI_Status status;
        MPI_Recv(vector, n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

        // Reparte el vector entre los procesos
        int n_elementos = n / size;
        int resto = n % size;
        int inicio = 0;
        int fin = n_elementos;
        double minimo = RANDMAX;

        double local_vector[n_elementos];

        for (int i = 0; i < size; i++)
        {
            if (i != 1)
            {
                MPI_Send(&vector[inicio], n_elementos, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                inicio = fin;
                fin += n_elementos;
                if (resto > 0)
                {
                    n_elementos++;
                    resto--;
                }
            }else{
                for (int j = 0; j < n_elementos; j++)
                {
                    local_vector[j] = vector[inicio];
                    inicio++;
                }
            }
        }

        //Calcula el minimo 
        for (int i = 0; i < n_elementos; i++)
        {
            if (local_vector[i] < minimo)
            {
                minimo = local_vector[i];
            }
        }

        printf("Minimo local, Proceso 1: %f\n", minimo);

        // Recibe los minimos de los procesos
        double *minimos = malloc(sizeof(double) * size);
        for(int i = 0; i < size; i++){
            if(i != 1){
                MPI_Recv(&minimos[i], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            }else{
                minimos[i] = minimo;
            }
        }

        // Calcula el minimo global
        minimo = RANDMAX;
        for (int i = 0; i < size; i++)
        {
            if (minimos[i] < minimo)
            {
                minimo = minimos[i];
            }
        }

        printf("Minimo FINAL: %f\n", minimo);
    }
    else
    {
        // Recibe la parte del vector que le corresponde
        double *vector_local = malloc(sizeof(double) * n / size);
        MPI_Status status;
        MPI_Recv(vector_local, n / size, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);

        // Calcula el minimo local
        double minimo_local = RANDMAX;
        for (int i = 0; i < n / size; i++)
        {
            if (vector_local[i] < minimo_local)
            {
                minimo_local = vector_local[i];
            }
        }
        
        // Envia el minimo local
        MPI_Send(&minimo_local, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        printf("Minimo local , Proceso %d: %f\n",rank ,minimo_local);
    }

    MPI_Finalize();
    return 0;
}