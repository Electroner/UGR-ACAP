#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#define TRYTIMES 10

double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL))
	{
		printf("Error de medición de tiempo\n");
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * 0.000001;
}

double piLeibnizParallel(int steps, int rank, int size)
{
	int interval = steps / size;
	int start = rank * interval;
	int end = (rank + 1) * interval;
	double pipart = 0.0;
	double num = 1.0;
	double denom = 1.0;
	for (int i = start; i < end; i++)
	{
		pipart += num / denom;
		num = -1.0 * num;
		denom += 2.0;
	}
	return 4.0 * pipart;
}

double piRectanglesParallel(int intervals, int rank, int size)
{
	int interval = intervals / size;
	int start = rank * interval;
	int end = (rank + 1) * interval;
	double width = 1.0 / intervals;
	double sum = 0.0, x;
	for (int i = start; i < end; i++)
	{
		x = (i + 0.5) * width;
		sum += 4.0 / (1.0 + x * x);
	}
	return sum * width;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double timeLeibMean = 0.0;
	double timeRectMean = 0.0;
	double timeLeib = 0.0;
	double timeRect = 0.0;

	double piresultLeib = 0.0;
	double piresultRect = 0.0;
	int steps = atoi(argv[1]);

	for (int i = 0; i < TRYTIMES; i++)
	{
		double timeLeiblocal = 0.0;
		double timeRectlocal = 0.0;

		MPI_Barrier(MPI_COMM_WORLD);
		timeLeiblocal = get_wall_time();
		double pipart = piLeibnizParallel(steps, rank, size);
		MPI_Reduce(&pipart, &piresultLeib, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		timeLeiblocal = get_wall_time() - timeLeiblocal;
		MPI_Barrier(MPI_COMM_WORLD);
		timeLeib = timeLeib + timeLeiblocal;
	
		MPI_Barrier(MPI_COMM_WORLD);
		timeRectlocal = get_wall_time();
		double rectpart = piRectanglesParallel(steps, rank, size);
		MPI_Reduce(&rectpart, &piresultRect, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		timeRectlocal = get_wall_time() - timeRectlocal;
		MPI_Barrier(MPI_COMM_WORLD);
		timeRect = timeRect + timeRectlocal;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&timeLeib, &timeLeibMean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&timeRect, &timeRectMean, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0)
	{
		printf("%d\t", steps);

		// Leibniz
		timeLeibMean /= TRYTIMES*size;
		printf("%.10lf\t", timeLeibMean);
		//printf("Pi: %.10lf\n", piresultLeib/size);

		// Rectangles
		timeRectMean /= TRYTIMES*size;
		printf("%.10lf\n", timeRectMean);
		//printf("Pi: %.10lf\n", piresultRect);
	}

	MPI_Finalize();
	return 0;
}

//set xlabel "Iterations"
//set ylabel "Seconds"
//plot "pi_parallel.dat" u 1:2 w l title "Leibniz Paralelo","pi_parallel.dat" u 1:3 w l title "Rectangles Paralelo","pi_secuencial.dat" u 1:3 w l title "Rectangles Secuencial", "pi_secuencial.dat" u 1:2 w l title "Leibniz Secuencial"
//plot "pi_parallel.dat" u 1:($2+$3) w l title "Programa Paralelo" lc rgb 'blue',"pi_secuencial.dat" u 1:($2+$3) w l title "Programa Secuencial" lc rgb 'red'

//set xlabel "Cores"
//set ylabel "SpeedUp"
//plot "SecParPi_Ganancia.dat" u 1:(($2+$3)/($4+$5)) w l title "Ganancia" lc rgb 'black'