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

        // Enviamos el tamaño a todos los procesos
        MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Reservar memoria para la matriz 1
        float **matriz = reservarMatrizFloat(tam);
        // Reservar memoria para la matriz 2
        float **matriz2 = reservarMatrizFloat(tam);
        // Reservar memoria para la matriz resultado
        float **matriz_resultado = reservarMatrizFloat(tam);

        // iniciar la matriz 1 con valores aleatorios
        llenarMatrizAleatoria(tam, matriz, 0, 100);
        // iniciar la matriz 2 con valores aleatorios
        llenarMatrizAleatoria(tam, matriz2, 0, 100);

        // Localmente multiplicar las dos matrices
        for (int a = 0; a < tam; a++)
        {
            // Dentro recorremos las filas de la primera (A)
            for (int i = 0; i < tam; i++)
            {
                float suma = 0;
                // Y cada columna de la primera (A)
                for (int j = 0; j < tam; j++)
                {
                    // Multiplicamos y sumamos resultado
                    suma += matriz[i][j] * matriz2[j][a];
                }
                // Lo acomodamos dentro del producto
                matriz_resultado[i][a] = suma;
            }
        }
        // Si el tamaño es menor que 8 mostrar la matriz resultado
        if (tam < 8)
        {
            printf("\nMatriz resultado secuencial:\n");
            for (int i = 0; i < tam; i++)
            {
                for (int j = 0; j < tam; j++)
                {
                    printf("%f\t", matriz_resultado[i][j]);
                    fflush(stdout);
                }
                printf("\n");
                fflush(stdout);
            }
        }

        // Mandamos a cada proceso la cantidad de filas que le coresponden
        int filas_por_proceso = tam / size;

        // Mandamos la cantidad de filas que le corresponde a cada proceso
        MPI_Bcast(&filas_por_proceso, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Mandamos la matriz 1 fila por fila a cada proceso
        for (int i = 0; i < tam; i++)
        {
            MPI_Bcast(&matriz[i][0], tam, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Mandamos la matriz 2 fila por fila a cada proceso
        for (int i = 0; i < tam; i++)
        {
            MPI_Bcast(&matriz2[i][0], tam, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Calculamos la parte correspondiente al proceso 0
        for (int i = 0; i < filas_por_proceso; i++)
        {
            for (int j = 0; j < tam; j++)
            {
                float suma = 0;
                for (int k = 0; k < tam; k++)
                {
                    suma += matriz[i][k] * matriz2[k][j];
                }
                matriz_resultado[i][j] = suma;
            }
        }

        // Recibimos los vectores resultados de cada proceso y los colocamos en la matriz resultado
        for (int i = 1; i < size; i++)
        {
            for (int j = 0; j < filas_por_proceso; j++)
            {
                MPI_Recv(&matriz_resultado[i * filas_por_proceso + j][0], tam, MPI_FLOAT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        // Si el tamaño es menor que 8 mostrar la matriz resultado
        if (tam < 8)
        {
            printf("\nMatriz resultado:\n");
            for (int i = 0; i < tam; i++)
            {
                for (int j = 0; j < tam; j++)
                {
                    printf("%f\t", matriz_resultado[i][j]);
                    fflush(stdout);
                }
                printf("\n");
                fflush(stdout);
            }
        }

        // Liberar memoria
        liberarMatriz(matriz);
        liberarMatriz(matriz2);
        liberarMatriz(matriz_resultado);
    }
    else
    {
        int filas_por_proceso = 0;

        // Recibimos el tamaño
        MPI_Bcast(&tam, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Recibimos la cantidad de filas que le corresponde a cada proceso
        MPI_Bcast(&filas_por_proceso, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Reservar memoria para la matriz 1
        float **matriz = reservarMatrizFloat(tam);
        // Reservar memoria para la matriz 2
        float **matriz2 = reservarMatrizFloat(tam);
        // Reservar memoria para la matriz resultado
        float **matriz_resultado = reservarMatrizFloat(tam);

        // Recibir la matriz 1 por filas
        for (int i = 0; i < tam; i++)
        {
            MPI_Bcast(&matriz[i][0], tam, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Recibir la matriz 2 por filas
        for (int i = 0; i < tam; i++)
        {
            MPI_Bcast(&matriz2[i][0], tam, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        // Inializar la matriz resultado a 0
        for (int i = 0; i < filas_por_proceso; i++)
        {
            for (int j = 0; j < tam; j++)
            {
                matriz_resultado[i][j] = 0;
            }
        }

        // Segun el rank del proceso multiplicar la fila correspondiente de la matriz 1 con la matriz 2
        for (int i = filas_por_proceso * rank; i < (filas_por_proceso * (rank + 1)); i++)
        {
            for (int j = 0; j < tam; j++)
            {
                float suma = 0;
                // Y cada columna de la primera (A)
                for (int k = 0; k < tam; k++)
                {
                    // Multiplicamos y sumamos resultado
                    suma += matriz[i][k] * matriz2[k][j];
                }
                // Lo acomodamos dentro del producto
                matriz_resultado[i][j] = suma;
            }
        }

        // Mandar la parte de la matriz resultado
        for (int i = filas_por_proceso * rank; i < (filas_por_proceso * (rank + 1)); i++)
        {
            MPI_Send(&matriz_resultado[i][0], tam, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        }

        // Liberar memoria
        liberarMatriz(matriz);
        liberarMatriz(matriz2);
        liberarMatriz(matriz_resultado);
    }

    MPI_Finalize();
    return 0;
}