#include <cstdio>
#include <cstdlib>	  // srand, rand
#include <ctime>	  // time
#include <sys/time.h> // get_wall_time

#define IMDEP 256
#define SIZE (100 * 1024 * 1024) // 100 MB

//#define NBLOCKS 16
//#define THREADS_PER_BLOCK 8

#define TEXT

const int numRuns = 10;

double get_wall_time()
{
	struct timeval time;
	if (gettimeofday(&time, NULL))
	{
		printf("Error en la medicion de tiempo CPU!!\n");
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void *inicializarImagen(unsigned long nBytes)
{
	unsigned char *img = (unsigned char *)malloc(nBytes);
	for (unsigned long i = 0; i < nBytes; i++)
	{
		img[i] = rand() % IMDEP;
	}
	return img;
}

void histogramaCPU(unsigned char *img, unsigned long nBytes, unsigned int *histo)
{
	for (int i = 0; i < IMDEP; i++)
		histo[i] = 0; // Inicializacion
	for (unsigned long i = 0; i < nBytes; i++)
	{
		histo[img[i]]++;
	}
	#ifndef TEXT
		printf("Tiempo de CPU (s): %.4lf\n", 0.0);
	#endif
}

long calcularCheckSum(unsigned int *histo)
{
	long checkSum = 0;
	for (int i = 0; i < IMDEP; i++)
	{
		checkSum += histo[i];
	}
	return checkSum;
}

int compararHistogramas(unsigned int *histA, unsigned int *histB)
{
	int valido = 1;
	for (int i = 0; i < IMDEP; i++)
	{
		if (histA[i] != histB[i])
		{
			printf("Error en [%d]: %u != %u\n", i, histA[i], histB[i]);
			valido = 0;
		}
	}
	return valido;
}

__global__ void kernelHistograma(unsigned char *imagen, unsigned long size, unsigned int *histo)
{
	//threadIdx is the index of the thread in the block
	//blockIdx is the index of the block in the grid
	//blockDim is the size of the block (how many threads are in the block)
	//gridDim is the size of the grid (how many blocks are in the grid)
	__shared__ unsigned int temp[IMDEP];

	unsigned long i = threadIdx.x;
	int offset = blockDim.x;

	while (i < IMDEP)
	{
		temp[i] = 0;
		i += offset;
	}
	
	__syncthreads();

	i = threadIdx.x + blockIdx.x * blockDim.x;
	offset = blockDim.x * gridDim.x;

	while (i < size)
	{
		atomicAdd(&temp[imagen[i]], 1);
		i += offset;
	}

	__syncthreads();

	i = threadIdx.x;
	offset = blockDim.x;

	while (i < IMDEP)
	{
		atomicAdd(&(histo[i]), temp[i]);
		i += offset;
	}

	__syncthreads();
}

int main(int argc, char **argv)
{
	int NBLOCKS;
	int THREADS_PER_BLOCK;

	if(argc != 3){
		exit(1);
	}

	//get the number of blocs and therds fron argv
	NBLOCKS = atoi(argv[1]);
	THREADS_PER_BLOCK = atoi(argv[2]);

	unsigned char *imagen = (unsigned char *)inicializarImagen(SIZE);
	unsigned int histoCPU[IMDEP];
	//Medir tiempo de CPU
	double tiempoCPU = get_wall_time();
	histogramaCPU(imagen, SIZE, histoCPU);
	tiempoCPU = get_wall_time() - tiempoCPU;

	long chk = calcularCheckSum(histoCPU);
	
	#ifndef TEXT
		printf("Check-sum CPU: %ld\n", chk);
	#endif

	unsigned char *dev_imagen = 0;
	unsigned int *dev_histo = 0;
	
	//Medir tiempo de transferencia de datos
	double tiempoTransfer = get_wall_time();
	cudaMalloc((void **)&dev_imagen, SIZE);
	cudaMemcpy(dev_imagen, imagen, SIZE, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_histo, IMDEP * sizeof(unsigned int));
	tiempoTransfer = get_wall_time() - tiempoTransfer;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliSeconds = 0.0;
	float aveGPUMS = 0.0;

	for (int iter = -1; iter < numRuns; iter++)
	{ // La iteraciÃ³n -1 es para que la tarjeta se ponga en marcha, normalmente siempre da peores tiempos.
		cudaMemset(dev_histo, 0, IMDEP * sizeof(unsigned int));
		if (iter < 0)
		{
			kernelHistograma<<<NBLOCKS, THREADS_PER_BLOCK>>>(dev_imagen, SIZE, dev_histo);
		}
		else
		{
			cudaDeviceSynchronize();
			cudaEventRecord(start);
			kernelHistograma<<<NBLOCKS, THREADS_PER_BLOCK>>>(dev_imagen, SIZE, dev_histo);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliSeconds, start, stop);
			aveGPUMS += milliSeconds;
		}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	unsigned int gpuHisto[IMDEP];
	cudaMemcpy(gpuHisto, dev_histo, IMDEP * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	chk = calcularCheckSum(gpuHisto);
	#ifndef TEXT
		printf("Check-sum GPU: %ld\n", chk);
	#endif
	if (compararHistogramas(histoCPU, gpuHisto)){
		#ifndef TEXT
			printf("Calculo correcto!!\n");
		#endif
	}
	#ifndef TEXT
		printf("Tiempo medio de ejecucion del kernel<<<%d, %d>>> sobre %u bytes [s]: %.4f\n", NBLOCKS, THREADS_PER_BLOCK, SIZE, aveGPUMS / (numRuns * 1000.0));
	#endif
	printf("%d %d %.4f\n", NBLOCKS, THREADS_PER_BLOCK, aveGPUMS / (numRuns * 1000.0));
	//Mostrar tiempo de CPU
	printf("%.4lf\n", tiempoCPU);
	//Tiempo con trasnferencia de datos
	printf("%.4lf\n", tiempoTransfer);

	free(imagen);
	cudaFree(dev_imagen);
	cudaFree(dev_histo);
	return 0;
}
