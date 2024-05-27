#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mpi.h>

// #define MAX_BLOCKS_PER_DIM 65535
#define MAX(a,b) (((a)>(b))?(a):(b))

__constant__ float L[3];
__constant__ float C[3];
__constant__ float W_y = 2;
__constant__ float W_max = 2;
__constant__ float R = 6;
__constant__ float PI = 3.1415926535897931;
__constant__ float c_c_dotprod;





__global__ void initCurandStates(curandState *states, int numStates, int mpiRank) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < numStates) {
        // Calculate a unique seed for each state by incorporating the MPI rank
        unsigned long long seed = (mpiRank * numStates + id) * 4238811ULL;
        curand_init(seed, 0, 0, &states[id]);
    }
}



__global__ void rayTraceKernel(float *G, int num_rays, int n, curandState *state, int totalThreads, int startRay, unsigned long long *randomCount) {

    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // int totalThreads = blockDim.x * gridDim.x; // Total number of threads
    curandState localstate = state[tid];
    float c_c_dotprod = C[0]*C[0] + C[1]*C[1] + C[2]*C[2];
    unsigned long long localCount = 0;
    // if (tid < num_rays/10) {
        // each stride is only dealing with 1/10th of the whole rays
        for (int ray = startRay + tid; ray < num_rays + startRay; ray += totalThreads) {
            float W_x, W_z;
            float alt;
            bool valid_ray = false;
            float I[3], N[3], S[3], V[3];    
            float v_c_dotprod;

            do {
                float phi = curand_uniform(&localstate) * PI;
                float cos_theta = curand_uniform(&localstate) * 2 - 1;
                localCount += 2;
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

            I[0] = t*V[0];
            I[1] = t*V[1];
            I[2] = t*V[2];
    
            // calculating N (DO USING FOR LOOP?)
            // for (int i = 0; i < 3; i++) {
            //     N[i] = I[i] - C[i];
            // }
    
            N[0] = I[0] - C[0];
            N[1] = I[1] - C[1];
            N[2] = I[2] - C[2];
    
            float n_mag = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]);
            
            N[0] = N[0] / n_mag;
            N[1] = N[1] / n_mag;
            N[2] = N[2] / n_mag;
    
    
            // for (int i = 0; i < 3; i++) {
            //     N[i] = N[i] / n_mag;
            // }
            
    
            // calculating S (DO USING FOR LOOP)
            // for (int i = 0; i < 3; i++) {
            //     S[i] = L[i] - I[i];
            // }
            S[0] = L[0] - I[0];
            S[1] = L[1] - I[1];
            S[2] = L[2] - I[2];
    
            
    
            float s_mag = sqrt(S[0] * S[0] + S[1] * S[1] + S[2] * S[2]);
            // for (int i = 0; i < 3; i++) {
            //     S[i] = S[i] / s_mag;
            // }
            S[0] = S[0] / s_mag;
            S[1] = S[1] / s_mag;
            S[2] = S[2] / s_mag;
    
            // calculating b
            float n_s_dotprod = N[0]*S[0] + N[1]*S[1] + N[2]*S[2];

            // printf("b = %f\n", n_s_dotprod);
            float b = MAX(0.0, n_s_dotprod);
                
            // updating G
            int i = (W_max - W_x)/(2.0 * W_max) * n;
            int j = (W_max + W_z)/(2.0 * W_max) * n;
            atomicAdd(&G[i*n+j], b);
        } 
    // }
    // bug fix, set the state again to ensure continuity 
    state[tid] = localstate;
    randomCount[tid] = localCount;
}


// __host__ void save_to_file(float *data, int N, const char *filename) {
//     FILE *f = fopen(filename, "w");
//     if (f == NULL) {
//         perror("Error opening file");
//         return;
//     }

//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             fprintf(f, "%e", data[i*N + j]);
//             if (j < N - 1) {
//                 fprintf(f, ",");
//             }
//         }
//         fprintf(f, "\n");
//     }

//     fclose(f);
// }
__host__ void save_to_file(float *data, int N, const char *filename) {
    FILE *f = fopen(filename, "wb"); // Open file for writing in binary mode
    if (f == NULL) {
        perror("Error opening file");
        return;
    }

    // Write the entire data array in one call
    // fwrite takes a pointer to the data, size of each element, number of elements, and the FILE pointer
    fwrite(data, sizeof(float), N*N, f);

    fclose(f);
}


int main(int argc, char **argv) {
    if (argc != 5) {
        printf("Usage: %s <nrays> <ngrid> <nblocks> <ntpb>\n", argv[0]);
        return 1;
    }

    MPI_Init(&argc, &argv);
    float startTime = MPI_Wtime(); // Start MPI timer
    
    float h_L[3] = {4, 4, -1};
    float h_C[3] = {0, 12, 0};
    cudaMemcpyToSymbol(L, h_L, sizeof(h_L));
    cudaMemcpyToSymbol(C, h_C, sizeof(h_C));
    // float h_c_c_dotprod = h_C[0]*h_C[0] + h_C[1]*h_C[1] + h_C[2]*h_C[2];
    // cudaMemcpyToSymbol(c_c_dotprod, &h_c_c_dotprod, sizeof(float));


    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    cudaSetDevice(worldRank % numGPUs); // Simple round-robin
    // printf("size is %d", worldSize);

    int numRays = atoi(argv[1]); // Total rays to process
    int n = atoi(argv[2]); // Dimension of the grid
    int blocksPerGrid = atoi(argv[3]);
    int threadsPerBlock = atoi(argv[4]);

    int totalThreads = blocksPerGrid*threadsPerBlock;

    // Assuming numRays divides evenly by worldSize
    int raysPerProcess = numRays / worldSize;

    // Calculate start ray for each rank
    int startRay = worldRank * raysPerProcess;

    numRays = raysPerProcess;

    unsigned long long *d_randomCount;
    cudaMalloc(&d_randomCount, totalThreads * sizeof(unsigned long long));
    cudaMemset(d_randomCount, 0, totalThreads * sizeof(unsigned long long));



    float *d_grid;
    size_t gridSize = n * n * sizeof(float);
    cudaMalloc(&d_grid, gridSize);
    cudaMemset(d_grid, 0, gridSize);

    curandState *d_states;
    cudaMalloc(&d_states, totalThreads * sizeof(curandState));

    initCurandStates<<<blocksPerGrid, threadsPerBlock>>>(d_states, totalThreads, worldRank);

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // printf("Launching kernel with %d blocks with %d threads each...\n", blocksPerGrid, threadsPerBlock);

    cudaEventRecord(start);

    // Adjust the kernel launch parameters based on command-line inputs
    rayTraceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid, numRays, n, d_states, totalThreads, startRay, d_randomCount);

    // cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f seconds\n", milliseconds/1000);

    unsigned long long *h_randomCount = (unsigned long long *)malloc(totalThreads * sizeof(unsigned long long));
cudaMemcpy(h_randomCount, d_randomCount, totalThreads * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

unsigned long long totalRandomNumbers = 0;
for(int i = 0; i < totalThreads; ++i) {
    totalRandomNumbers += h_randomCount[i];
}

printf("Total random numbers generated: %llu\n", totalRandomNumbers);



    // collect results
    // 
    float *h_grid = NULL; // Host grid for gathering results
    if (worldRank == 0) {
        h_grid = (float *)malloc(n * n * sizeof(float)); // Allocate only on the root process
    }
    float *recv_buffer = (float *)malloc(n * n * sizeof(float)); // Temp buffer for all ranks

    // copy GPU data to host
    cudaMemcpy(recv_buffer, d_grid, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Reduce(recv_buffer, h_grid, n*n, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);


    if (worldRank == 0) {
        save_to_file(h_grid, n, "mult_cuda.bin");
        free(h_grid);
    }

    free(recv_buffer); // Free temporary buffer on all ranks
    

    float endTime = MPI_Wtime(); // Stop MPI timer
    float totalTime = endTime - startTime; // Calculate total execution time

    // Print total execution time on master process
    if (worldRank == 0) {
        printf("Total program time: %f seconds\n", totalTime);
    }

    MPI_Finalize();



    // Cleanup and finalize
    cudaFree(d_grid);
    cudaFree(d_states);
    return 0;
}
