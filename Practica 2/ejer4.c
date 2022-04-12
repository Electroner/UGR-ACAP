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

int main(int argc, char *argv[]){
    
    return 0;
}