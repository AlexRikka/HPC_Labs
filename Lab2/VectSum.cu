#include <malloc.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h>
#include <sys/time.h>

__device__ void singleWarp(volatile int* cache, int index){
    cache[index] += cache[index + 32];
    cache[index] += cache[index + 16];
    cache[index] += cache[index + 8];
    cache[index] += cache[index + 4];
    cache[index] += cache[index + 2];
    cache[index] += cache[index + 1]; 
}

__global__ void addKernel(int* a, int* c, int N) { 
    // buffer for storing each thread's running sum
    __shared__ int cache[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x*blockDim.x; 
    int cacheIndex = threadIdx.x;   
    
    int tmp = 0;
    for (int i = cacheIndex; i < N; i += THREADS_PER_BLOCK) {
        tmp += a[i];
    }
    cache[cacheIndex] = tmp;

    // synchronize threads in block
    __syncthreads();

    // do reduction in shared memory
    for (int s = blockDim.x/2; s > 32; s >>=1){
        if(tid < s){
            cache[cacheIndex] += cache[cacheIndex + s];
        } 
        __syncthreads();
    }

    if (cacheIndex < 32){
        singleWarp(cache, cacheIndex);
    }

    // write result to global memory
    if (cacheIndex == 0) {
        *c = cache[0];
    }
}        

// function for CPU
int conseq_add(int *a, const int N){
    int sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += a[i];
    }
    return sum;
}

double time_diff(struct timeval x, struct timeval y){
    double x_ms, y_ms, diff;
    x_ms = (double)x.tv_sec * 1000000 + (double)x.tv_usec;
    y_ms = (double)y.tv_sec * 1000000 + (double)y.tv_usec;
    diff = (double)y_ms - (double)x_ms;
    return diff;
}

int main(int argc, char* argv[]) {
	int n = atoi(argv[1]); 

    int i;
    int nb = n*sizeof(int);

    // allocate memory on host, vectors are filled with zeros
    int *a=(int*) calloc(n,sizeof(int)); 
    int *c=(int*) calloc(1,sizeof(int)); 

    // initialize vector
    for (i = 0; i < n; ++i) {
        a[i] = 2;
    }
   
    // allocate memory on device
    int *adev = NULL;  
	int *cdev = NULL;  
    cudaMalloc((void**) &adev, nb); 
    cudaMalloc((void**) &cdev, sizeof(int)); 
	
	// copy vectors from host to device 
	//cudaMemcpy(dst, src, size in bytes to copy, type of transfer)
    cudaMemcpy(adev, a, nb, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop; 
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 

    // call kernel
    addKernel<<<1, THREADS_PER_BLOCK>>>(adev, cdev, n); 
	
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0); 

    // copy result to host 
	//cudaMemcpy(dst, src, size in bytes to copy, type of transfer)
    cudaMemcpy(c, cdev, sizeof(int), cudaMemcpyDeviceToHost); 
    
    cudaEventElapsedTime(&gpuTime, start, stop); //time in ms

    // Computing time for CPU function
    double total_time = 0.0;
    struct timeval before, after;
    gettimeofday(&before, NULL);
    int cpu_sum = conseq_add(a, n);
    gettimeofday(&after, NULL);
    total_time += time_diff(before, after);

    // Outputs
    printf("N = %i\n", n);
    printf("GPU_sum = %i, ", c[0]); 
    printf("CPU_sum = %i\n", cpu_sum);  
    printf("Time for GPU: %0.3f ms\n", gpuTime); 
    printf("Time for CPU: %0.3f ms\n", total_time / ((double) 1000));

    // Очищение памяти 
    cudaEventDestroy(start); 
    cudaEventDestroy(stop); 
    cudaFree(adev);
    cudaFree(cdev);

    free(a); 
    free(c); 
    
    return 0; 
}
