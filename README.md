# Ray Tracing Performance Analaysis (GPUs)

This project implements a ray tracing simulation using three different approaches: OpenMP for CPU parallelization, CUDA for GPU acceleration, and a hybrid MPI+CUDA for distributed GPU computing. Ray tracing is a rendering technique for generating images by tracing the path of light as pixels in an image plane and simulating the effects of its encounters with virtual objects. The simulation calculates intersections between rays and objects to determine the color and brightness of pixels, creating highly realistic images.

## Files Description

- `parallel.c`: Implements the ray tracing algorithm using OpenMP. This version runs on the CPU, utilizing multiple cores to parallelize the computation and accelerate the rendering process.
- `gpu_version.cu`: Utilizes CUDA to accelerate the ray tracing algorithm on NVIDIA GPUs. By leveraging the parallel computing capabilities of GPUs, it significantly speeds up the rendering process compared to the CPU-based version.
- `gpu_parallel.cu`: Incorporates MPI to enable the ray tracing simulation across multiple GPUs in a distributed computing environment. This version is designed for high-performance computing clusters with multiple nodes, each equipped with one or more GPUs.

## Requirements

- NVIDIA GPU (for CUDA and MPI+CUDA versions)
- MPI library (for MPI+CUDA version)
- CUDA Toolkit
- OpenMP-supported compiler (for `parallel.c`)

## Compilation

A makefile has been provided in the src directory to compile the necessary files.

## Running the Code

### OpenMP-based Parallelization (`parallel.c`)
```
./parallel <num_threads> <num_rays> <grid_size>
```
- Replace `<num_threads>` with the number of threads, `<num_rays>` with the number of rays to simulate, and `<grid_size>` with the size of the grid.
- Output will be stored in `sphere.bin`

### CUDA-based Serial Implementation (`gpu_version.cu`)
```
./gpu_version <num_rays> <grid_size> <num_blocks> <threads_per_block>
```
- Replace `<num_rays>`, `<grid_size>`, `<num_blocks>`, and `<threads_per_block>` with appropriate values for your simulation.
- Output will be stored in `cuda_out.bin`

### CUDA and MPI-based Parallel Implementation (`gpu_parallel.cu`)
```
mpirun -np <num_processes> ./gpu_parallel <num_rays> <grid_size> <num_blocks> <threads_per_block>
```
- Replace `<num_processes>` with the number of MPI processes to spawn, which should align with the number of GPUs available. Adjust the other parameters similarly to the CUDA serial version.
- Output will be stored in `mult_cuda.bin`

### Plotting (`plotting.py`)
```
python plotting.py <filename> <grid_size> <data_type>
```
- Replace `<filename>` with the path to your binary file containing the ray tracing output, `<grid_size>` with the size of the grid (either row or column assuming it is a square grid), and `<data_type>` with either `single` for `float32` data or `double` for `float64` data, depending on how the binary data was generated.
