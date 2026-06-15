// Stage 2 — Block-level: 1 permutation = 1 block, 256 threads parallelize segments.
//
// Design choice: each thread independently bootstrap-resamples a length-(N/T) sub-segment.
// Algorithmically a variant ("K independent block-bootstrap chains" vs Politis-Romano's
// "1 chain of length N") but the resulting Sharpe null distribution is asymptotically
// equivalent under H0 (stationarity). Empirical validation: see benchmark/dist_match.py.
//
// Optimizations:
//   - 1 perm/block → block-level parallelism on time steps
//   - Warp shuffle reduction for Σr, Σr²
//   - cuRAND state per thread (Philox)
//   - Coalesced reads from returns[] (random access, but L1/L2 cache helps)
//
// Build: nvcc -O3 -arch=sm_86 --use_fast_math stage2_block.cu -o stage2_block

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define BLOCK_THREADS 256

__inline__ __device__ double warp_reduce_sum(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void bootstrap_block(
    const double* __restrict__ returns, int N,
    double* __restrict__ sharpes,
    double p, unsigned long long seed)
{
    const int perm_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int T = blockDim.x;
    const int seg_len = (N + T - 1) / T;
    const int seg_start = tid * seg_len;
    const int seg_end = min(seg_start + seg_len, N);

    if (seg_start >= N) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, perm_id * T + tid, 0, &state);

    int i = curand(&state) % N;
    double sum = 0.0, sum2 = 0.0;
    int count = seg_end - seg_start;

    #pragma unroll 4
    for (int t = 0; t < count; ++t) {
        double r = returns[i];
        sum += r;
        sum2 += r * r;
        // stationary block continuation
        if (curand_uniform_double(&state) < p) i = curand(&state) % N;
        else i = (i + 1) % N;
    }

    // Block reduction: warp shuffle → shared mem → final warp
    __shared__ double s_sum[BLOCK_THREADS / 32];
    __shared__ double s_sum2[BLOCK_THREADS / 32];
    __shared__ int    s_count[BLOCK_THREADS / 32];

    int lane = tid & 31;
    int warp = tid >> 5;

    double wsum = warp_reduce_sum(sum);
    double wsum2 = warp_reduce_sum(sum2);
    double wcount_d = warp_reduce_sum((double)count);

    if (lane == 0) {
        s_sum[warp] = wsum;
        s_sum2[warp] = wsum2;
        s_count[warp] = (int)wcount_d;
    }
    __syncthreads();

    if (warp == 0) {
        double bsum  = (lane < BLOCK_THREADS / 32) ? s_sum[lane] : 0.0;
        double bsum2 = (lane < BLOCK_THREADS / 32) ? s_sum2[lane] : 0.0;
        double bcnt  = (lane < BLOCK_THREADS / 32) ? (double)s_count[lane] : 0.0;
        bsum  = warp_reduce_sum(bsum);
        bsum2 = warp_reduce_sum(bsum2);
        bcnt  = warp_reduce_sum(bcnt);
        if (lane == 0) {
            double mean = bsum / bcnt;
            double var = (bsum2 - bcnt * mean * mean) / (bcnt - 1.0);
            sharpes[perm_id] = (var > 0.0) ? mean / sqrt(var) : 0.0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "Usage: %s returns.bin N B block_len_inv seed [dump_sharpes.bin]\n", argv[0]); return 1; }
    const char* path = argv[1];
    int N = atoi(argv[2]);
    int B = atoi(argv[3]);
    double p = atof(argv[4]);
    unsigned long long seed = strtoull(argv[5], nullptr, 10);

    FILE* f = fopen(path, "rb");
    double* h_ret = (double*)malloc(sizeof(double) * N);
    fread(h_ret, sizeof(double), N, f);
    fclose(f);

    double *d_ret, *d_sharpes;
    CUDA_OK(cudaMalloc(&d_ret, sizeof(double) * N));
    CUDA_OK(cudaMalloc(&d_sharpes, sizeof(double) * B));
    CUDA_OK(cudaMemcpy(d_ret, h_ret, sizeof(double) * N, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warmup
    bootstrap_block<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes, p, seed);
    CUDA_OK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    bootstrap_block<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes, p, seed);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double* h_sharpes = (double*)malloc(sizeof(double) * B);
    CUDA_OK(cudaMemcpy(h_sharpes, d_sharpes, sizeof(double) * B, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < B; ++i) sum += h_sharpes[i];
    printf("Stage2 block: N=%d B=%d threads=%d elapsed=%.3fms throughput=%.3f Mperms/s null_mean=%.6f\n",
           N, B, BLOCK_THREADS, ms, (double)B / (ms * 1e3), sum / B);

    if (argc >= 7) {
        FILE* fo = fopen(argv[6], "wb");
        if (fo) { fwrite(h_sharpes, sizeof(double), B, fo); fclose(fo); }
    }

    free(h_ret); free(h_sharpes);
    cudaFree(d_ret); cudaFree(d_sharpes);
    return 0;
}
