// Stage 5 — Multi-stream batching for variant sweep.
//
// Use case: quant researcher가 한 hypothesis의 K=12 variants (다른 seed, 다른 returns 등) 를
// 동시에 검정. 단일 strategy의 추가 가속이 아닌 **multi-strategy throughput** 차원.
//
// 메커니즘:
//   - cudaStream_t 4개 (concurrent stream)
//   - pinned host memory (cudaMallocHost) → async copy 진짜 overlap
//   - strategy i → stream (i % NSTREAMS) round-robin
//   - 각 strategy: cudaMemcpyAsync(H2D) → bootstrap kernel → cudaMemcpyAsync(D2H)
//   - stream 간 copy/kernel overlap → SM 활용 + PCIe BW 동시
//
// 측정: K개 strategy 전체 wall-clock vs Stage 3 sequential (단일 stream으로 K번 launch)
//
// Build: nvcc -O3 -arch=sm_86 --use_fast_math -lineinfo stage5_multistream.cu -o stage5_multistream

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define BLOCK_THREADS 256

// Stage 3 kernel을 그대로 재사용 (signature 동일)
__inline__ __device__ double warp_reduce_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void bootstrap_block_fp32(
    const float* __restrict__ returns, int N,
    double* __restrict__ sharpes,
    uint32_t p_threshold,
    unsigned long long seed)
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

    int i = (int)(curand(&state) % (uint32_t)N);
    float sum  = 0.0f;
    float sum2 = 0.0f;
    int count = seg_end - seg_start;

    int t = 0;
    while (t + 1 < count) {
        uint4 r4 = curand4(&state);
        float v = returns[i];
        sum  += v; sum2 += v * v;
        if (r4.x < p_threshold) i = (int)(r4.y % (uint32_t)N);
        else                    i = (i + 1) % N;

        v = returns[i];
        sum  += v; sum2 += v * v;
        if (r4.z < p_threshold) i = (int)(r4.w % (uint32_t)N);
        else                    i = (i + 1) % N;
        t += 2;
    }
    if (t < count) {
        uint4 r4 = curand4(&state);
        float v = returns[i];
        sum  += v; sum2 += v * v;
        if (r4.x < p_threshold) i = (int)(r4.y % (uint32_t)N);
        else                    i = (i + 1) % N;
    }

    __shared__ double s_sum[BLOCK_THREADS / 32];
    __shared__ double s_sum2[BLOCK_THREADS / 32];
    __shared__ int    s_count[BLOCK_THREADS / 32];

    int lane = tid & 31;
    int warp = tid >> 5;

    double dsum  = warp_reduce_sum_d((double)sum);
    double dsum2 = warp_reduce_sum_d((double)sum2);
    double dcnt  = warp_reduce_sum_d((double)count);

    if (lane == 0) {
        s_sum[warp]   = dsum;
        s_sum2[warp]  = dsum2;
        s_count[warp] = (int)dcnt;
    }
    __syncthreads();

    if (warp == 0) {
        double bsum  = (lane < BLOCK_THREADS / 32) ? s_sum[lane] : 0.0;
        double bsum2 = (lane < BLOCK_THREADS / 32) ? s_sum2[lane] : 0.0;
        double bcnt  = (lane < BLOCK_THREADS / 32) ? (double)s_count[lane] : 0.0;
        bsum  = warp_reduce_sum_d(bsum);
        bsum2 = warp_reduce_sum_d(bsum2);
        bcnt  = warp_reduce_sum_d(bcnt);
        if (lane == 0) {
            double mean = bsum / bcnt;
            double var  = (bsum2 - bcnt * mean * mean) / (bcnt - 1.0);
            sharpes[perm_id] = (var > 0.0) ? mean / sqrt(var) : 0.0;
        }
    }
}

// =============================================================================
// Multi-stream variant sweep
// =============================================================================

void run_sequential(
    const float* d_ret, int N, int B, double p, unsigned long long base_seed,
    int K, double* d_sharpes_all, float* ms_out)
{
    // 단일 stream으로 K번 launch (baseline)
    uint32_t p_threshold = (uint32_t)(p * 4294967296.0);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warmup
    bootstrap_block_fp32<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes_all, p_threshold, base_seed);
    CUDA_OK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int k = 0; k < K; ++k) {
        unsigned long long seed = base_seed + (unsigned long long)k * 1000003ULL;
        bootstrap_block_fp32<<<B, BLOCK_THREADS>>>(
            d_ret, N, d_sharpes_all + (size_t)k * B, p_threshold, seed);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(ms_out, t0, t1);
}

void run_multistream(
    const float* d_ret, int N, int B, double p, unsigned long long base_seed,
    int K, int NSTREAMS, double* d_sharpes_all, double* h_sharpes_pinned, float* ms_out)
{
    uint32_t p_threshold = (uint32_t)(p * 4294967296.0);

    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * NSTREAMS);
    for (int s = 0; s < NSTREAMS; ++s) cudaStreamCreate(&streams[s]);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warmup
    bootstrap_block_fp32<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes_all, p_threshold, base_seed);
    CUDA_OK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    for (int k = 0; k < K; ++k) {
        cudaStream_t s = streams[k % NSTREAMS];
        unsigned long long seed = base_seed + (unsigned long long)k * 1000003ULL;
        bootstrap_block_fp32<<<B, BLOCK_THREADS, 0, s>>>(
            d_ret, N, d_sharpes_all + (size_t)k * B, p_threshold, seed);
        // async D2H to pinned host
        cudaMemcpyAsync(
            h_sharpes_pinned + (size_t)k * B,
            d_sharpes_all + (size_t)k * B,
            sizeof(double) * B,
            cudaMemcpyDeviceToHost, s);
    }
    for (int s = 0; s < NSTREAMS; ++s) cudaStreamSynchronize(streams[s]);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(ms_out, t0, t1);

    for (int s = 0; s < NSTREAMS; ++s) cudaStreamDestroy(streams[s]);
    free(streams);
}

int main(int argc, char** argv) {
    if (argc < 8) {
        fprintf(stderr, "Usage: %s returns_fp32.bin N B block_len_inv seed K NSTREAMS\n"
                        "  K = variant 수 (e.g., 12)\n"
                        "  NSTREAMS = 동시 stream 수 (e.g., 4)\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int N = atoi(argv[2]);
    int B = atoi(argv[3]);
    double p = atof(argv[4]);
    unsigned long long base_seed = strtoull(argv[5], nullptr, 10);
    int K = atoi(argv[6]);
    int NSTREAMS = atoi(argv[7]);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    float* h_ret = (float*)malloc(sizeof(float) * N);
    size_t nread = fread(h_ret, sizeof(float), N, f);
    fclose(f);
    if ((int)nread != N) {
        fprintf(stderr, "expected %d float32 elements, got %zu\n", N, nread);
        return 1;
    }

    float*  d_ret;
    double* d_sharpes_all;
    CUDA_OK(cudaMalloc(&d_ret, sizeof(float) * N));
    CUDA_OK(cudaMalloc(&d_sharpes_all, sizeof(double) * (size_t)B * K));
    CUDA_OK(cudaMemcpy(d_ret, h_ret, sizeof(float) * N, cudaMemcpyHostToDevice));

    // Pinned host buffer for async D2H
    double* h_sharpes_pinned;
    CUDA_OK(cudaMallocHost(&h_sharpes_pinned, sizeof(double) * (size_t)B * K));

    // ---- Sequential baseline ----
    float ms_seq = 0.0f;
    run_sequential(d_ret, N, B, p, base_seed, K, d_sharpes_all, &ms_seq);

    // ---- Multi-stream ----
    float ms_ms = 0.0f;
    run_multistream(d_ret, N, B, p, base_seed, K, NSTREAMS,
                    d_sharpes_all, h_sharpes_pinned, &ms_ms);

    // 보고용 mean (multi-stream output 사용)
    double sum = 0.0;
    for (size_t i = 0; i < (size_t)B * K; ++i) sum += h_sharpes_pinned[i];

    printf("Stage5 multistream: N=%d B=%d K=%d NSTREAMS=%d\n", N, B, K, NSTREAMS);
    printf("  Sequential : %.3f ms (per strategy %.3f ms)\n", ms_seq, ms_seq / K);
    printf("  Multistream: %.3f ms (per strategy %.3f ms)\n", ms_ms, ms_ms / K);
    printf("  Speedup    : %.3fx\n", ms_seq / ms_ms);
    printf("  Throughput : seq %.3f Mperms/s, ms %.3f Mperms/s\n",
           (double)B * K / (ms_seq * 1e3), (double)B * K / (ms_ms * 1e3));
    printf("  null_mean (all K×B): %.6f\n", sum / ((double)B * K));

    free(h_ret);
    cudaFreeHost(h_sharpes_pinned);
    cudaFree(d_ret); cudaFree(d_sharpes_all);
    return 0;
}
