#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <pthread.h>
#include <sched.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096
// #define NI 1024
// #define NJ 1024
// #define NK 1024

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}

/* Utility function to set the thread affinity to a given core. */
void set_affinity(int core_id)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask))
        printf( "Error setting thread affinity\n");
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	    C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling optimization. */
static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj, kk;
  int TILE_SIZE = 128;

  for (ii = 0; ii < NI; ii += TILE_SIZE) {
    for (jj = 0; jj < NJ; jj += TILE_SIZE) {
      
      for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
        for (j = jj; j < jj + TILE_SIZE && j < NJ; j++) {
          C[i*NJ+j] *= beta;
        }
      }
      
      for (kk = 0; kk < NK; kk += TILE_SIZE) {
        for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
          for (k = kk; k < kk + TILE_SIZE && k < NK; k++) {
            for (j = jj; j < jj + TILE_SIZE && j < NJ; j++) {
              C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
            }
          }
        }
      }
    }
  }

}

/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int TILE_SIZE = 128;
  int i, j, k, ii, jj, kk;

  for (ii = 0; ii < NI; ii += TILE_SIZE) {
    for (jj = 0; jj < NJ; jj += TILE_SIZE) {   
      for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
        for (j = jj; j < jj + TILE_SIZE && j < NJ; j++) {
          C[i*NJ+j] *= beta;
        }
      }
      // SIMD
      for (kk = 0; kk < NK; kk += TILE_SIZE) {
        for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
          for (k = kk; k < kk + TILE_SIZE && k < NK; k++) {
            __m256 a_val = _mm256_set1_ps(alpha * A[i*NK+k]);
            for (j = jj; j < jj + TILE_SIZE && j < NJ; j += 8) {
              __m256 b_vals = _mm256_loadu_ps(&B[k*NJ+j]);
              __m256 c_vals = _mm256_loadu_ps(&C[i*NJ+j]);
              c_vals = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals), c_vals);
              _mm256_storeu_ps(&C[i*NJ+j], c_vals);
            }
          }
        }
      }
    }
  }
}

/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int TILE_SIZE = 128;

  int i, j, k, ii, jj, kk;

  #pragma omp parallel for num_threads(20) collapse(2) private(i, j, k, ii, jj, kk) // parrallelization

  for (ii = 0; ii < NI; ii += TILE_SIZE) {
    for (jj = 0; jj < NJ; jj += TILE_SIZE) {      
      for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
        for (j = jj; j < jj + TILE_SIZE && j < NJ; j++) {
          C[i*NJ+j] *= beta;
        }
      }
      // SIMD
      for (kk = 0; kk < NK; kk += TILE_SIZE) {
        for (i = ii; i < ii + TILE_SIZE && i < NI; i++) {
          for (k = kk; k < kk + TILE_SIZE && k < NK; k++) {
            __m256 a_val = _mm256_set1_ps(alpha * A[i*NK+k]);
            for (j = jj; j < jj + TILE_SIZE && j < NJ; j += 8) {
              __m256 b_vals = _mm256_loadu_ps(&B[k*NJ+j]);
              __m256 c_vals = _mm256_loadu_ps(&C[i*NJ+j]);
              c_vals = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals), c_vals);
              _mm256_storeu_ps(&C[i*NJ+j], c_vals);
            }
          }
        }
      }
    }
  }
}


void gemm_perf(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  const int TILE_SIZE = 64;
  const int TILE_SIZE_I = 64;
  const int TILE_SIZE_J = 512;
  const int TILE_SIZE_K = 64;

  int i, j, k, ii, jj, kk;

  #pragma omp parallel for num_threads(20) collapse(2) private(i, j, k, ii, jj, kk)

  for (ii = 0; ii < NI; ii += TILE_SIZE_I) {
    for (jj = 0; jj < NJ; jj += TILE_SIZE_J) {
      for (i = ii; i < ii + TILE_SIZE_I && i < NI; i++) {
        for (j = jj; j < jj + TILE_SIZE_J && j < NJ; j++) {
          C[i*NJ+j] *= beta;
        }
      }
      for (kk = 0; kk < NK; kk += TILE_SIZE_K) {
        for (i = ii; i < ii + TILE_SIZE_I && i < NI; i++) {
          for (k = kk; k < kk + TILE_SIZE_K && k < NK; k++) {
            __m256 a_val = _mm256_set1_ps(alpha * A[i*NK+k]);
            for (j = jj; j < jj + TILE_SIZE_J && j < NJ; j += 32) {
              __m256 b_vals1 = _mm256_loadu_ps(&B[k*NJ+j]);
              __m256 b_vals2 = _mm256_loadu_ps(&B[k*NJ+j+8]);
              __m256 b_vals3 = _mm256_loadu_ps(&B[k*NJ+j+16]);
              __m256 b_vals4 = _mm256_loadu_ps(&B[k*NJ+j+24]);
              __m256 c_vals1 = _mm256_loadu_ps(&C[i*NJ+j]);
              __m256 c_vals2 = _mm256_loadu_ps(&C[i*NJ+j+8]);
              __m256 c_vals3 = _mm256_loadu_ps(&C[i*NJ+j+16]);
              __m256 c_vals4 = _mm256_loadu_ps(&C[i*NJ+j+24]);
              c_vals1 = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals1), c_vals1);
              c_vals2 = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals2), c_vals2);
              c_vals3 = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals3), c_vals3);
              c_vals4 = _mm256_add_ps(_mm256_mul_ps(a_val, b_vals4), c_vals4);
              _mm256_storeu_ps(&C[i*NJ+j], c_vals1);
              _mm256_storeu_ps(&C[i*NJ+j+8], c_vals2);
              _mm256_storeu_ps(&C[i*NJ+j+16], c_vals3);
              _mm256_storeu_ps(&C[i*NJ+j+24], c_vals4);
            }
          }
        }
      }
    }
  }
}




int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  case 4: // performance competition
    /* Run kernel. */
    gemm_perf (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "perf time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
