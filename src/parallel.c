#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <stdbool.h>
#define MAX(a,b) (((a)>(b))?(a):(b))




float *grid;
float const L[3] = {4, 4, -1};
float const C[3] = {0, 12, 0};
float const W_y = 2;
float const W_max = 2;
float const R = 6;
float const PI = 3.1415926535897931;

void save_to_file(float *data, int N, const char *filename) {
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




int main(int argc, char *argv[]){

    if (argc != 4) { // Updated to expect 3 additional arguments
        fprintf(stderr, "Usage: %s <num_threads> <num_rays> <n>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[1]);
    int num_rays = atol(argv[2]);
    int n = atoi(argv[3]);

    if (num_threads <= 0) {
        fprintf(stderr, "Number of threads must be a positive integer\n");
        return 1; // Exit the program if the input is invalid
    }
    

    omp_set_num_threads(num_threads); // Set the number of threads for OpenMP
    float start = omp_get_wtime();
    grid = (float *)calloc(n * n, sizeof(float));
    float c_c_dotprod = C[0]*C[0] + C[1]*C[1] + C[2]*C[2];
    unsigned long long totalRandomNumbers = 0; 

    // parallel region

    #pragma omp parallel shared(grid, c_c_dotprod) reduction(+:totalRandomNumbers)
    {
    unsigned int seed = time(NULL) ^ omp_get_thread_num(); // Unique seed per thread
    #pragma omp for 

    for (long int ray=0; ray < num_rays; ray++){
        float W_x, W_z;
        float alt;
        bool valid_ray = false;
        //move this out
        float I[3], N[3], S[3], V[3];    
        float v_c_dotprod, phi, cos_theta, sin_theta;

        do {
                phi = ((float)rand_r(&seed) / RAND_MAX) * PI;
                cos_theta = 2.0 * ((float)rand_r(&seed) / RAND_MAX) - 1.0;
                totalRandomNumbers += 2;
                sin_theta = sqrt(1 - cos_theta * cos_theta);
                V[0] = sin_theta * cos(phi);
                V[1] = sin_theta * sin(phi);
                V[2] = cos_theta;
                W_x = W_y / V[1] * V[0];
                W_z = W_y / V[1] * V[2];
                // only do the dot products if needed. 
                if(W_x > -W_max && W_x < W_max && W_z > -W_max && W_z < W_max) {
                    v_c_dotprod = V[0]*C[0] + V[1]*C[1] + V[2]*C[2];
                    alt = (v_c_dotprod * v_c_dotprod) + (R * R) - c_c_dotprod;
                    if (alt > 0) {
                        valid_ray = true; 
                    } 
                }
            } while(!valid_ray);

        float t = v_c_dotprod - sqrt(alt);

        // REMOVE FOR LOOPS AND UNECESSARY CALCULATIONS
        
        // calculating I (DO IT USING FOR LOOP MAYBE?)
        // for (int i = 0; i < 3; i++) {
        //     I[i] = t * V[i];
        // }
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
        float b = MAX(0.0, n_s_dotprod);
            
        // updating G
        int i = (W_max - W_x)/(2.0 * W_max) * n;
        int j = (W_max + W_z)/(2.0 * W_max) * n;

        // Update the brightness at the specific grid location
        #pragma omp atomic
        grid[i * n + j] += b;
        }
    }

    // End timing
    float end = omp_get_wtime();
    float cpu_time_used = end - start;
    printf("Total random numbers generated: %llu\n", totalRandomNumbers);

    printf("Time used: %f seconds\n", cpu_time_used);

    char filename[256]; // Buffer to hold the filename
    // Choose the filename based on the number of threads
    if (num_threads == 48) {
        strcpy(filename, "sphere_48.bin");
    } else {
        strcpy(filename, "sphere.bin");
    }

    
    save_to_file(grid, n, filename);
    free(grid);

    return 0;
    }
    
