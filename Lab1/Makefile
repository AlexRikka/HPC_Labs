NVCC = nvcc 
CFLAGS = -g -G -O0 -lcurand -lcublas
MatMul: MatMul.cu
		$(NVCC) $(CFLAGS) $< -o $@