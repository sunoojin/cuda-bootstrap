// Stage 4a — Stage 3 + explicit __ldg() read-only cache hint.
//
// 목적: random access pattern (returns[idx])에 대해 sm_86의 read-only L1 cache (texture cache 회로)를
// explicit하게 지시. nvcc가 `const __restrict__`만 보고 자동 ldg할 수도 있으나, explicit 형태로
// 측정 비교 + 보고서에 metric 기록.
//
// 변경: returns[i] → __ldg(&returns[i]) (전체 3 위치)
//
// Build: nvcc -O3 -arch=sm_86 --use_fast_math -lineinfo stage4a_ldg.cu -o stage4a_ldg

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

#define BLOCK_THREADS 256

__inline__ __device__ double warp_reduce_sum_d(double v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

__global__ void bootstrap_block_fp32_ldg(
    const float* __restrict__ returns, int N,
    double* __restrict__ sharpes,
    uint32_t p_threshold,    // (uint32_t)(p * 2^32)
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

    // 초기 i — 1 uint 사용
    int i = (int)(curand(&state) % (uint32_t)N);

    // 누적은 fp32 (sub-segment 길이 ~217 → 안전 범위)
    float sum  = 0.0f;
    float sum2 = 0.0f;
    int count = seg_end - seg_start;

    // curand4 = Philox 1 round → uint4 (4 uints).
    // 매 iter는 (continuation 판정 1 uint) + (new index 1 uint) = 2 uints.
    // → 1 curand4 call = 2 iters.
    int t = 0;
    while (t + 1 < count) {
        uint4 r4 = curand4(&state);

        // iter t
        float v = __ldg(&returns[i]);
        sum  += v;
        sum2 += v * v;
        if (r4.x < p_threshold) i = (int)(r4.y % (uint32_t)N);
        else                    i = (i + 1) % N;

        // iter t+1
        v = __ldg(&returns[i]);
        sum  += v;
        sum2 += v * v;
        if (r4.z < p_threshold) i = (int)(r4.w % (uint32_t)N);
        else                    i = (i + 1) % N;

        t += 2;
    }
    // 홀수 길이 처리
    if (t < count) {
        uint4 r4 = curand4(&state);
        float v = __ldg(&returns[i]);
        sum  += v;
        sum2 += v * v;
        // i는 마지막 iter라 update 불필요 but 일관성
        if (r4.x < p_threshold) i = (int)(r4.y % (uint32_t)N);
        else                    i = (i + 1) % N;
    }

    // Block reduction (Stage 2와 동일 패턴, 단 final reduction은 double)
    __shared__ double s_sum[BLOCK_THREADS / 32];
    __shared__ double s_sum2[BLOCK_THREADS / 32];
    __shared__ int    s_count[BLOCK_THREADS / 32];

    int lane = tid & 31;
    int warp = tid >> 5;

    // float → double cast 후 reduce (warp 내 32 elements 누적도 double precision)
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

int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s returns_fp32.bin N B block_len_inv seed [dump_sharpes.bin]\n"
                        "  Note: input must be float32 binary (not double).\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];
    int N = atoi(argv[2]);
    int B = atoi(argv[3]);
    double p = atof(argv[4]);
    unsigned long long seed = strtoull(argv[5], nullptr, 10);

    // p_threshold = floor(p * 2^32). p < 1 가정.
    uint32_t p_threshold = (uint32_t)(p * 4294967296.0);

    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    float* h_ret = (float*)malloc(sizeof(float) * N);
    size_t nread = fread(h_ret, sizeof(float), N, f);
    fclose(f);
    if ((int)nread != N) {
        fprintf(stderr, "expected %d float32 elements, got %zu (file mismatch)\n", N, nread);
        return 1;
    }

    float*  d_ret;
    double* d_sharpes;
    CUDA_OK(cudaMalloc(&d_ret, sizeof(float) * N));
    CUDA_OK(cudaMalloc(&d_sharpes, sizeof(double) * B));
    CUDA_OK(cudaMemcpy(d_ret, h_ret, sizeof(float) * N, cudaMemcpyHostToDevice));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    // warmup
    bootstrap_block_fp32_ldg<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes, p_threshold, seed);
    CUDA_OK(cudaDeviceSynchronize());

    cudaEventRecord(t0);
    bootstrap_block_fp32_ldg<<<B, BLOCK_THREADS>>>(d_ret, N, d_sharpes, p_threshold, seed);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    double* h_sharpes = (double*)malloc(sizeof(double) * B);
    CUDA_OK(cudaMemcpy(h_sharpes, d_sharpes, sizeof(double) * B, cudaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < B; ++i) sum += h_sharpes[i];
    printf("Stage4a fp32+curand4+ldg: N=%d B=%d threads=%d elapsed=%.3fms throughput=%.3f Mperms/s null_mean=%.6f\n",
           N, B, BLOCK_THREADS, ms, (double)B / (ms * 1e3), sum / B);

    if (argc >= 7) {
        FILE* fo = fopen(argv[6], "wb");
        if (fo) { fwrite(h_sharpes, sizeof(double), B, fo); fclose(fo); }
    }

    free(h_ret); free(h_sharpes);
    cudaFree(d_ret); cudaFree(d_sharpes);
    return 0;
}
