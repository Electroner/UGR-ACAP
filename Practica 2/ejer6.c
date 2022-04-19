#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <ctype.h>
#include <math.h>

#define RANGO_MIN 0
#define RANGO_MAX 40

//ELIGE EL MODO DE TEXTO PARA LA CONSOLA
//#define TEXTO
#define TIMES

float drand(float low, float high)
{
    return ((float)rand() * (high - low)) / (float)RAND_MAX + low;
}

double denom_a(double *A, unsigned int Vector_Length)
{
    double denom_a = 0;
    for (unsigned int i = 0u; i < Vector_Length; ++i)
    {
        denom_a += A[i] * A[i];
    }
}

double denom_b(double *B, unsigned int Vector_Length)
{
    double denom_b = 0;
    for (unsigned int i = 0u; i < Vector_Length; ++i)
    {
        denom_b += B[i] * B[i];
    }
}

double dot(double *A, double *B, unsigned int Vector_Length)
{
    double dot = 0;
    for (unsigned int i = 0u; i < Vector_Length; ++i)
    {
        dot += A[i] * B[i];
    }
}

double cosine_similarity(double dot, double denA, double denB)
{
    return dot / denA * denB;
}

int main(int argc, char *argv[])
{
    int rank, size, tam = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Si el numero de procesaodores es diferente de 3 no se puede ejecutar
    if (size != 3)
    {
        if (rank == 0)
        {
            printf("El numero de procesadores debe ser 3\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0)
    {
        double start_time = 0;
        double end_time = 0;

        if (argc < 2)
        {
            do
            {
                printf("Ingrese el tamaño de la matriz: ");
                fflush(stdout);
                scanf("%d", &tam);
            } while (tam <= 0);
        }else{
            if(atoi(argv[1]) <= 0){
                printf("El tamaño de la matriz debe ser mayor a 0\n");
                MPI_Finalize();
                return 0;
            }
            tam = atoi(argv[1]);
        }

        //Si esta definido TEXTO se imprime el texto
        #ifdef TEXTO
        // Mostrar el tamaño
        printf("El tamaño de la matriz es: %d\n", tam);
        fflush(stdout);
        #endif


        // Crear un vector1 de doubles con el tamaño indicado
        double *vector1 = (double *)malloc(tam * sizeof(double));

        // Crear un vector2 de doubles con el tamaño indicado
        double *vector2 = (double *)malloc(tam * sizeof(double));

        // Rellenar el vector de valores aleatorios dentro del rango
        for (int i = 0; i < tam; i++)
        {
            vector1[i] = drand(RANGO_MIN, RANGO_MAX);
            vector2[i] = drand(RANGO_MIN, RANGO_MAX);
        }

        #ifdef TEXTO
        // Si el el tamaño es menor que 6 se imprime el vector
        if (tam < 6)
        {
            printf("Vector 1: ");
            for (int i = 0; i < tam; i++)
            {
                printf("%f ", vector1[i]);
            }
            printf("\n");
            printf("Vector 2: ");
            for (int i = 0; i < tam; i++)
            {
                printf("%f ", vector2[i]);
            }
            printf("\n");
        }
        #endif

        // Empezar a calcular el tiempo
        start_time = MPI_Wtime();

        // Localmente calcular la similitud de cosenos
        double denomasec = denom_a(vector1, tam);
        double denombsec = denom_b(vector2, tam);
        double dotsec = dot(vector1, vector2, tam);
        double cosine_similarity_result_local = cosine_similarity(dotsec, denomasec, denombsec);

        // Finalizar el tiempo
        end_time = MPI_Wtime();

        #ifdef TIMES
        // Mostrar el tamaño
        printf("%d\t", tam);
        // Imprimir el tiempo
        printf("%f\t", end_time - start_time);
        fflush(stdout);
        #endif

        #ifdef TEXTO
        // Imprimir el resultado
        printf("El resultado de la similitud de cosenos es: %f\n", cosine_similarity_result_local);
        printf("El tiempo de ejecucion Secuencial es: %f\n", end_time - start_time);
        #endif

        // Enviar el tamaño de la matriz a los procesos
        MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Enviar los vectores a los demas procesadores
        MPI_Bcast(vector1, tam, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector2, tam, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Calcular el tiempo de inicio
        start_time = MPI_Wtime();

        // Calcular la parte del proceso 0 (funcion dot)
        double dot_result = dot(vector1, vector2, tam);

        // Esperar a que nos envien el resultado de la funcion denom_a y denom_b
        MPI_Status status;
        double denom_a_result, denom_b_result;
        MPI_Recv(&denom_a_result, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&denom_b_result, 1, MPI_DOUBLE, 2, 0, MPI_COMM_WORLD, &status);

        // Calcular la similitud de cosenos
        double cosine_similarity_result = cosine_similarity(dot_result, denom_a_result, denom_b_result);

        // Calcular el tiempo de finalizacion
        end_time = MPI_Wtime();

        #ifdef TIMES
        // Imprimir el tiempo
        printf("%f\n", end_time - start_time);
        fflush(stdout);
        #endif

        #ifdef TEXTO
        // Imprimir el resultado
        printf("El resultado de la similitud de cosenos es: %f\n", cosine_similarity_result);
        printf("El tiempo de ejecucion es: %f\n", end_time - start_time);
        #endif
    }
    else
    {

        // Recibir el tamaño de la matriz
        MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Crear un vector1 de doubles con el tamaño indicado
        double *vector1 = (double *)malloc(tam * sizeof(double));

        // Crear un vector2 de doubles con el tamaño indicado
        double *vector2 = (double *)malloc(tam * sizeof(double));

        // Recibir los vectores de los procesos
        MPI_Bcast(vector1, tam, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vector2, tam, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 1)
        {
            // Calcular la parte del proceso 1 (funcion denom_a)
            double denom_a_result = denom_a(vector1, tam);

            // Enviar el resultado de la funcion denom_a
            MPI_Send(&denom_a_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        else
        {
            // Calcular la parte del proceso 2 (funcion denom_b)
            double denom_b_result = denom_b(vector2, tam);

            // Enviar el resultado de la funcion denom_a y denom_b
            MPI_Send(&denom_b_result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}