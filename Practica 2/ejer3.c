#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <ctype.h>
#include <math.h>

#define true 1
#define false 0

double ExitP(int nProcs)
{
    int trash = 0.0;
    for (int i = 0; i < nProcs; i++)
    {
        MPI_Send(&trash, 1, MPI_INT, i, 666, MPI_COMM_WORLD);
    }
}

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
    int opcion;

    if (size != 4)
    {
        printf("Error: Se necesitan 4 procesos\n");
        return 1;
    }

    switch (rank)
    {
    case 0:
        while (true)
        {
            // Preguntar para ingresar la opcion
            printf("Ingrese la opcion:");
            fflush(stdout);
            scanf("%d", &opcion);

            switch (opcion)
            {
            case 0:
                MPI_Abort(MPI_COMM_WORLD, 0);
                break;

            case 1:
                printf("FUNCIONALIDAD 1\n");
                // Preguntar por una linea de texto
                char linea[256];
                printf("Ingrese una linea de texto:");
                fflush(stdout);
                //get the line
                fgets(linea, 256, stdin);

                // Enviar la linea de texto al proceso 1
                MPI_Send(linea, 256, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

                // Recibir la linea de texto del proceso 1
                MPI_Recv(linea, 256, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Mostrar la linea de texto recibida
                printf("Linea de texto recibida: %s\n", linea);
                fflush(stdout);
                break;

            case 2:
                printf("FUNCIONALIDAD 2\n");
                // Generamos un vector de 10 numeros aleatorios
                double vector[10];
                double sqrtresultado;
                double localsqrt;
                double localsuma;

                for (int i = 0; i < 10; i++)
                {
                    vector[i] = (int)drand(0, 100);
                }

                // Calcular la suma de los numeros
                for (int i = 0; i < 10; i++)
                {
                    localsuma += vector[i];
                }

                // calcular la raiz cuadrada de la suma
                localsqrt = sqrt(localsuma);

                // Mostrar la raiz cuadrada de la suma
                printf("La raiz secuencial es: %f\n", localsqrt);

                // Enviar el vector al proceso 2
                MPI_Send(vector, 10, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD);

                // Recibir la raiz del vector del proceso 2
                MPI_Recv(&sqrtresultado, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Mostrar la raiz del vector
                printf("Raiz de la suma es: %f\n", sqrtresultado);
                fflush(stdout);
                break;

            case 3:
                printf("FUNCIONALIDAD 3\n");
                // Se crea una cadena "Entrando en funcionalidad 3"
                char cadena[256];
                sprintf(cadena, "Entrando en funcionalidad 3");

                // Se envia la cadena al proceso 3
                MPI_Send(cadena, 256, MPI_CHAR, 3, 0, MPI_COMM_WORLD);

                int sumachars;

                // Se recibe la sumade los caracteres del proceso 3
                MPI_Recv(&sumachars, 1, MPI_INT, 3, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Se muestra la suma de los caracteres
                printf("La suma de los caracteres es: %d\n", sumachars);
                fflush(stdout);
                break;

            case 4:
                printf("FUNCIONALIDAD 4\n");

                break;

            default:
                MPI_Finalize();
                return 0;
                break;
            }
        }
        break;

    case 1:
        while (true)
        {
            // Recibir la linea de texto del proceso 0
            char linea[256];
            MPI_Recv(linea, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // toupper
            for (int i = 0; i < 256; i++)
            {
                linea[i] = toupper(linea[i]);
            }

            // Enviar la linea de texto al proceso 0
            MPI_Send(linea, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
        break;

    case 2:
        while (true)
        {
            // Recibir el vector del proceso 0
            double vector[10];
            MPI_Recv(vector, 10, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Calcular la raiz cuadrada del vector
            double sqrtresultado = sqrt(vector[0] + vector[1] + vector[2] + vector[3] + vector[4] + vector[5] + vector[6] + vector[7] + vector[8] + vector[9]);

            // Enviar la raiz del vector al proceso 0
            MPI_Send(&sqrtresultado, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        break;

    case 3:
        while (true)
        {
            // Recibir la cadena del proceso 0
            char cadena[256];
            MPI_Recv(cadena, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Calcular la suma de los caracteres de la cadena
            int sumachars = 0;
            for (int i = 0; i < 256; i++)
            {
                sumachars += cadena[i];
            }

            // Enviar la suma de los caracteres al proceso 0
            MPI_Send(&sumachars, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }

        break;

    default:
        break;
    }

    MPI_Finalize();
    return 0;
}