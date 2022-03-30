#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define TRYTIMES 5

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

double piLeibniz(int steps)
{
	double partpi = 0.0;
	double num = 1.0;
	double denom = 1.0;
	for (int i = 0; i < steps; i++)
	{
		partpi += num / denom;
		num = -1.0 * num; // Alternamos el signo
		denom += 2.0;
	}
	return 4.0 * partpi;
}

double piRectangles(int intervals)
{
	double width = 1.0 / intervals;
	double sum = 0.0, x;
	for (int i = 0; i < intervals; i++)
	{
		x = (i + 0.5) * width;
		sum += 4.0 / (1.0 + x * x);
	}
	return sum * width;
}

int main(int argc, char *argv[])
{

	double timeLeibMean = 0.0;
	double timeRectMean = 0.0;
	double timeLeib = 0.0;
	double timeRect = 0.0;

	int steps = atoi(argv[1]);

	for (int i = 0; i < TRYTIMES; i++)
	{
		double timeLeiblocal = 0.0;
		double timeRectlocal = 0.0;

		timeLeiblocal = get_wall_time();
		double pipart = piLeibniz(steps);
		timeLeiblocal = get_wall_time() - timeLeiblocal;
		timeLeibMean += timeLeiblocal;

		timeRectlocal = get_wall_time();
		double rectpart = piRectangles(steps);
		timeRectlocal = get_wall_time() - timeRectlocal;
		timeRectMean += timeRectlocal;
	}
	printf("%d\t", steps);

	// Leibniz
	timeLeibMean /= TRYTIMES;
	printf("%.10lf\t", timeLeibMean);

	// Rectangles
	timeRectMean /= TRYTIMES;
	printf("%.10lf\n", timeRectMean);
	return 0;
}

//set xlabel "Iterations"
//set ylabel "Seconds"
//plot "pi_parallel.dat" u 1:2 w l title "Leibniz Paralelo","pi_parallel.dat" u 1:3 w l title "Rectangles Paralelo","pi_secuencial.dat" u 1:3 w l title "Rectangles Secuencial", "pi_secuencial.dat" u 1:2 w l title "Leibniz Secuencial"