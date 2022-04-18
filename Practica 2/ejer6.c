#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <ctype.h>
#include <math.h>

#define RANGO_MIN 0
#define RANGO_MAX 40

float drand(float low, float high)
{
    return ((float)rand() * (high - low)) / (float)RAND_MAX + low;
}

inline double denom_a(double *A,unsigned int Vector_Length){
    double denom_a = 0;
    for(unsigned int i = 0u; i < Vector_Length; ++i) {
        denom_a += A[i] * A[i];
    }
}

inline double denom_b(double *B,unsigned int Vector_Length){
    double denom_b = 0;
    for(unsigned int i = 0u; i < Vector_Length; ++i) {
        denom_b += B[i] * B[i];
    }
}

double cosine_similarity(double *A, double *B, unsigned int Vector_Length)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int i = 0u; i < Vector_Length; ++i) {
        dot += A[i] * B[i] ;
    }
    return dot / denom_a(A,Vector_Length) * denom_b(B,Vector_Length) ;
}

int main(int argc, char *argv[])
{
    int rank, size, tam = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(rank == 0){
        do{
            printf("Ingrese el tamaño de la matriz: ");
            fflush(stdout);
            scanf("%d", &tam);
        }while(tam<=0);
        // Mostrar el tamaño
        printf("El tamaño de la matriz es: %d\n", tam);
        fflush(stdout);

        //Crear un vector1 de doubles con el tamaño indicado
        double *vector1 = (double *)malloc(tam * sizeof(double));

        //Crear un vector2 de doubles con el tamaño indicado
        double *vector2 = (double *)malloc(tam * sizeof(double));

        //Rellenar el vector de valores aleatorios dentro del rango
        for(int i = 0; i < tam; i++){
            vector1[i] = drand(RANGO_MIN, RANGO_MAX);
            vector2[i] = drand(RANGO_MIN, RANGO_MAX);
        }

        //Si el el tamaño es menor que 5 se imprime el vector
        if(tam < 5){
            printf("Vector 1: ");
            for(int i = 0; i < tam; i++){
                printf("%f ", vector1[i]);
            }
            printf("\n");
            printf("Vector 2: ");
            for(int i = 0; i < tam; i++){
                printf("%f ", vector2[i]);
            }
            printf("\n");
        }



    }else{


    }

    MPI_Finalize();
    return 0;
}