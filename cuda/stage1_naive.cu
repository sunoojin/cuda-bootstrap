// Stage 1 — Naive CUDA: 1 thread / permutation.
// 의도적으로 비효율 — Stage 2-5 비교 baseline.
//
// Build:  nvcc -O3 -arch=sm_86 stage1_naive.cu -o stage1_naive
// Run:    ./stage1_naive returns.bin N B block_len_inv seed
//
// Input file: float64 returns, packed binary, length = N

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void bootstrap_naive(
    const double* __restrict__ returns, int N,
    double* __restrict__ sharpes, int B,
    double p, unsigned long long seed)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, b, 0, &state);

    double sum = 0.0, sum2 = 0.0;
    int i = curand(&state) % N;

    for (int t = 0; t < N; ++t) {
        double r = returns[i];
        sum += r;
        sum2 += r * r;
        if (curand_uniform_double(&state) < p) {
            i = curand(&state) % N;
        } else {
            i = (i + 1) % N;
        }
    }

    double mean = sum / N;
    double var = (sum2 - N * mean * mean) / (N - 1);
    sharpes[b] = (var > 0.0) ? mean / sqrt(var) : 0.0;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "Usage: %s returns.bin N B block_len_inv seed\n", argv[0]); return 1; }
    const char* path = argv[1];
    int N = atoi(argv[2]);
    int B = atoi(argv[3]);
    double p = atof(argv[4]);
    unsigned long long seed = strtoull(argv[5], nullptr, 10);

    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); return 1; }
    double* h_ret = (double*)malloc(sizeof(double) * N);
    if (fread(h_ret, sizeof(double), N, f) != (size_t)N) { fprintf(stderr, "read error\n"); return 1; }
    fclose(f);

    double *d_ret, *d_sharpes;
    CUDA_OK(cudaMalloc(&d_ret, sizeof(double) * N));
    CUDA_OK(cudaMalloc(&d_sharpes, sizeof(double) * B));
    CUDA_OK(cudaMemcpy(d_ret, h_ret, sizeof(double) * N, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    bootstrap_naive<<<blocks, threads>>>(d_ret, N, d_sharpes, B, p, seed);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double* h_sharpes = (double*)malloc(sizeof(double) * B);
    CUDA_OK(cudaMemcpy(h_sharpes, d_sharpes, sizeof(double) * B, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for (int i = 0; i < B; ++i) sum += h_sharpes[i];
    printf("Stage1 naive: N=%d B=%d elapsed=%.2fms throughput=%.2f Mperms/s null_mean=%.6f\n",
           N, B, ms, (double)B / (ms * 1e3), sum / B);

    free(h_ret); free(h_sharpes);
    cudaFree(d_ret); cudaFree(d_sharpes);
    return 0;
}
