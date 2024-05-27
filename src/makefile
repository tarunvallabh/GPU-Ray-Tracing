CC=gcc
NVCC=nvcc
CFLAGS=-O3 -fopenmp -flto -march=native -mtune=native -ffast-math
CUDAFLAGS=-Xcompiler -O3 -use_fast_math -arch=sm_70
MPILIB=-lmpi
LIBS=-lm
TARGETS=gpu_version gpu_parallel parallel

all: $(TARGETS)

gpu_version: gpu_version.cu
	$(NVCC) $(CUDAFLAGS) -o $@ $<

gpu_parallel: gpu_parallel.cu
	$(NVCC) $(CUDAFLAGS) $(MPILIB) -o $@ $<

parallel: parallel.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f $(TARGETS)
