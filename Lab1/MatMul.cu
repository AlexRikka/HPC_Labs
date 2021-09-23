#include <cstdio>
#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <time.h>
#include <sys/time.h>
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void gpu_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    //(Pointer to generator, Type of generator to create) 
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // Fill the array with random numbers on the device
    // (Generator, Pointer to device memory to store CUDA-generated results, Number of floats to generate)
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) { 
    // A: m*k, B: k*n, C: m*n
    // C = alpha*op(A)*op(B)+ beta*C 
    //nr_rows_A, nr_cols_A, nr_cols_B
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    // Do the actual multiplication
    // (generator, transa, transb, m,n,k, alpha,
    //  array of dimensions lda x k, leading dimension of two-dimensional array used to store the matrix A, ...
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    // Destroy the handle
    cublasDestroy(handle);
}

//function for CPU
void conseq_mmul(const float *A, const float *B, float *C, const int N) {
    int i, j, k;
    for(i=0; i < N; i++){
        for(j=0; j < N; j++){
            C[IDX2C(i, j, N)] = 0;
            for(k=0; k < N; k++)
            C[IDX2C(i, j, N)] += A[IDX2C(i, k, N)]*B[IDX2C(k, j, N)];
        }
    }
}

double time_diff(struct timeval x, struct timeval y){
	double x_ms, y_ms, diff;
	x_ms = (double)x.tv_sec * 1000000 + (double)x.tv_usec;
	y_ms = (double)y.tv_sec * 1000000 + (double)y.tv_usec;
	diff = (double)y_ms - (double)x_ms;
	return diff;
}

void print_matrix(float *Matr, int nr_rows_Matr, int nr_cols_Matr){
	for (int i = 0; i < nr_rows_Matr; ++i) {
        for (int j = 0; j < nr_cols_Matr; ++j) {
            printf("%f ", Matr[IDX2C(i, j, nr_rows_Matr)]);
        }
        printf("%s", "\n");
    }
}

int main(int argc, char* argv[]) {
    int n = atoi(argv[1]); 
    // Allocate 3 arrays on CPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    // We are going to use square arrays
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = n;
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
	
	// Fill in 2 matrices on GPU
	gpu_fill_rand(d_A, nr_rows_A, nr_cols_A); 
   	gpu_fill_rand(d_B, nr_rows_B, nr_cols_B);
	
	// Copy them on CPU
    cudaMemcpy(h_A, d_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost); 
	cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost); 

	// printf("%s", "A=\n");
    // print_matrix(h_A, nr_rows_A, nr_cols_A);
	// printf("%s", "B=\n");
    // print_matrix(h_B, nr_rows_B, nr_cols_B);
	
	cudaEvent_t start, stop; 
    float gpuTime = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	// Setting start point
    cudaEventRecord(start, 0);
	
    // Multiply A and B on GPU
    gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

	cudaDeviceSynchronize();

    // Copy (and print) the result on host memory
    cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

    // Setting end point
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop); 

    // printf("%s", "C=\n"); 
    // print_matrix(h_C, nr_rows_C, nr_cols_C);
    // Computing time for CPU function
    
	double total_time = 0.0;
	struct timeval before, after;
	gettimeofday(&before, NULL);
	conseq_mmul(h_A, h_B, h_C, n);
	gettimeofday(&after, NULL);
    total_time += time_diff(before, after);
    // printf("%s", "C=\n"); 
    // print_matrix(h_C, nr_rows_C, nr_cols_C);

    printf("N = %i\n", n); 
    printf("Time for GPU: %f seconds\n", gpuTime/1000);  //gpuTime is in milliseconds
    printf("Time for CPU: %f seconds\n", total_time / ((double) 1000000));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
