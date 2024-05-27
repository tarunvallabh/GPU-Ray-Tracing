#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda_runtime.h>

// #define MAX_BLOCKS_PER_DIM 65535
#define MAX(a,b) (((a)>(b))?(a):(b))

__constant__ float L[3];
__constant__ float C[3];
__constant__ float W_y = 2;
__constant__ float W_max = 2;
__constant__ float R = 6;
__constant__ float PI = 3.1415926535897931;

__device__ unsigned long long totalRandomDraws = 0;




__global__ void initCurandStates(curandState *states, int numStates) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numStates) {
        // Use id * 4238811 as the seed for each state for uniqueness
        curand_init(id * 4238811ULL, 0, 0, &states[id]);
    }
}


__global__ void rayTraceKernel(float *G, int num_rays, int n, curandState *state, int totalThreads) {


    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localstate = state[tid];
    unsigned long long localDraws = 0;
    float c_c_dotprod = C[0]*C[0] + C[1]*C[1] + C[2]*C[2];

    // if (tid < num_rays/10) {
        // each stride is only dealing with 1/10th of the whole rays
        for (int ray = tid; ray < num_rays; ray += totalThreads) {
            float W_x, W_z;
            float alt;
            bool valid_ray = false;
            float I[3], N[3], S[3], V[3];    
            float v_c_dotprod;

            do {
                float phi = curand_uniform(&localstate) * PI;
                float cos_theta = curand_uniform(&localstate) * 2 - 1;
                localDraws += 2;
                float sin_theta = sqrt(1 - cos_theta * cos_theta);
                V[0] = sin_theta * cos(phi);
                V[1] = sin_theta * sin(phi);
                V[2] = cos_theta;
                W_x = W_y / V[1] * V[0];
                W_z = W_y / V[1] * V[2];
                // only do the dot products if needed. 
                if(W_x > -W_max && W_x < W_max && W_z > -W_max && W_z < W_max) {
                    v_c_dotprod = V[0]*C[0] + V[1]*C[1] + V[2]*C[2];
                    // float c_c_dotprod = C[0]*C[0] + C[1]*C[1] + C[2]*C[2];
                    alt = (v_c_dotprod * v_c_dotprod) + (R * R) - c_c_dotprod;
                    if (alt > 0) {
                        valid_ray = true;
                    } 
                }
            } while(!valid_ray);

            // calculating t
            float t = v_c_dotprod - sqrt(alt);

            // calculating I (DO IT USING FOR LOOP)
            for (int i = 0; i < 3; i++) {
                I[i] = t * V[i];
            }
            

            // calculating N (DO USING FOR LOOP?)
            for (int i = 0; i < 3; i++) {
                N[i] = I[i] - C[i];
            }

            float n_mag = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
            for (int i = 0; i < 3; i++) {
                N[i] = N[i] / n_mag;
            }
            

            // calculating S (DO USING FOR LOOP)
            for (int i = 0; i < 3; i++) {
                S[i] = L[i] - I[i];
            }

            float s_mag = sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
            for (int i = 0; i < 3; i++) {
                S[i] = S[i] / s_mag;
            }

            // calculating b
            float n_s_dotprod = N[0]*S[0] + N[1]*S[1] + N[2]*S[2];
            float b = MAX(0.0, n_s_dotprod);
                
            // updating G
            int i = (W_max - W_x)/(2.0 * W_max) * n;
            int j = (W_max + W_z)/(2.0 * W_max) * n;
            atomicAdd(&G[i*n+j], b);
        } 
    // }
    // bug fix, set the state again to ensure continuity 
    state[tid] = localstate;
    atomicAdd(&totalRandomDraws, localDraws);
}


__host__ void save_to_file(float *data, int N, const char *filename) {
    FILE *f = fopen(filename, "wb"); // Open file for writing in binary mode
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the entire data array in one call
    fwrite(data, sizeof(float), N*N, f);

    fclose(f);
}

int main(int argc, char** argv){
    if (argc != 5) {
        printf("Usage: %s <nrays> <ngrid> <nblocks> <ntpb>\n", argv[0]);
        return 1;
    }

    clock_t start_total, end_total;
    start_total = clock();

    float h_L[3] = {4, 4, -1};
    float h_C[3] = {0, 12, 0};
    cudaMemcpyToSymbol(L, h_L, sizeof(h_L));
    cudaMemcpyToSymbol(C, h_C, sizeof(h_C));

    // int num_rays = 1000000000;
    // int n = 1000;
    int num_rays = atoi(argv[1]);
    int n = atoi(argv[2]);
    int blocksPerGrid = atoi(argv[3]);
    int threadsPerBlock = atoi(argv[4]);

    int totalThreads = blocksPerGrid * threadsPerBlock;

    float *d_grid, *grid;
    size_t size = n * n * sizeof(float);

    // Allocate memory on host
    grid = (float *)calloc(n * n, sizeof(float));

    // Allocate memory on device
    cudaMalloc((void **) &d_grid, size);
    cudaMemset(d_grid, 0, size); // Initialize the device memory to zero


    curandState *d_states;
    cudaMalloc((void **) &d_states, totalThreads * sizeof(curandState));

    // Launch the kernel to initialize states
    initCurandStates<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreads);

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    cudaEventRecord(start);

    // Launch the kernel
    rayTraceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, num_rays, n, d_states, totalThreads);

    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f seconds\n", milliseconds/1000);

    unsigned long long hostTotalRandomDraws;
    cudaMemcpyFromSymbol(&hostTotalRandomDraws, totalRandomDraws, sizeof(unsigned long long));
    printf("Total random numbers drawn: %llu\n", hostTotalRandomDraws);

    // Copy the result back to host
    cudaMemcpy(grid, d_grid, size, cudaMemcpyDeviceToHost);
    

    // Save the grid to a file
    save_to_file(grid, n, "cuda_out.bin");

    // Free memory
    cudaFree(d_grid);
    free(grid);
    cudaFree(d_states);

    // Stop the clock and calculate the total execution time
    end_total = clock();
    float total_time = (float)(end_total - start_total) / CLOCKS_PER_SEC;
    printf("Total program time: %f seconds\n", total_time);

    return 0;

}




