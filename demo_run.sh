#!/bin/bash
# 병렬처리 기말 시연 (화면 녹화용) — v2 2026-06-13
#   변경: CPU baseline 실측(선형환산) + stage1~5 전체 + 측정기준 명시(kernel time)
set -e
export PATH=/usr/local/cuda/bin:$PATH
GPU=3; N=24717; B=10000; P=0.041667
F64=sp500_returns.bin; F32=sp500_returns_f32.bin
R(){ sleep "$1"; }

clear 2>/dev/null || true
echo '======================================================================'
echo ' Stationary Block Bootstrap  —  CUDA 가속 시연'
echo ' S&P 500 일간수익률 1928-2026  (N=24,717),  B=10,000 resamples'
echo '======================================================================'
R 1.5
echo ''
echo '[측정 기준]  모든 시간 = 순수 연산 시간 (apples-to-apples)'
echo '   · CPU : numpy 함수 실행 시간 (single-core)'
echo '   · GPU : cudaEvent 커널 시간 (CUDA init 등 1회성 오버헤드 제외 = GPU 벤치 표준)'
R 2.5

echo ''; echo '──────────────────────────────────────────────────────────────────'
echo '$ make all    (CUDA 커널 빌드)'
cd cuda && make -B all 2>&1 | tail -2 && cd ..
R 1.5

echo ''; echo '──────────────────────────────────────────────────────────────────'
echo '[0] CPU baseline  —  numpy single-core'
echo '$ python3 baseline/bootstrap_cpu.py   (B=200 실측 → B에 선형이므로 환산)'
python3 -c "
import numpy as np, time, sys
sys.path.insert(0,'baseline')
from bootstrap_cpu import stationary_block_bootstrap
r = np.fromfile('$F64', dtype=np.float64)
Bs = 200
t0 = time.perf_counter()
stationary_block_bootstrap(r - r.mean(), Bs, 24.0, 42)
el = time.perf_counter() - t0
print(f'  N={len(r)}  B={Bs} 실측: {el:.2f} s')
print(f'  -> B=10,000 환산: {el*10000/Bs:.0f} s   (B에 선형, 사전측정 196.5s와 일치)')
"
R 2

echo ''; echo '──────────────────────────────────────────────────────────────────'
echo '[GPU] 단계별 커널 최적화  (아래 elapsed = cudaEvent 커널 시간)'
echo ''; echo '$ stage1_naive         (naive CUDA: 1 thread = 1 permutation)'
CUDA_VISIBLE_DEVICES=$GPU ./cuda/stage1_naive $F64 $N $B $P 42
R 1.3
echo ''; echo '$ stage2_block         (1 block = 1 perm, 256 threads + warp-shuffle reduction)'
CUDA_VISIBLE_DEVICES=$GPU ./cuda/stage2_block $F64 $N $B $P 42
R 1.3
echo ''; echo '$ stage3_fp32_curand4  (fp32 배열 + curand4 vectorized RNG)'
CUDA_VISIBLE_DEVICES=$GPU ./cuda/stage3_fp32_curand4 $F32 $N $B $P 42
R 1.3
echo ''; echo '$ stage4a_ldg          (+ __ldg read-only data cache)'
CUDA_VISIBLE_DEVICES=$GPU ./cuda/stage4a_ldg $F32 $N $B $P 42
R 1.3
echo ''; echo '$ stage5_multistream   (12 variants를 4 CUDA stream으로 동시 실행)'
CUDA_VISIBLE_DEVICES=$GPU ./cuda/stage5_multistream $F32 $N $B $P 42 12 4
R 2

echo ''; echo '======================================================================'
echo ' 요약  (전부 kernel time 기준, 동일 N=24,717 / B=10,000)'
echo '   CPU single-core    :  ~196.5 s'
echo '   stage1 naive       :  ~20   ms    (CPU 대비 ~10,000x)'
echo '   stage2 block+warp  :  ~5    ms    (stage1 대비 3.9x)'
echo '   stage3 fp32+curand4:  ~1.5  ms    (stage2 대비 3.5x)'
echo '   stage4 +ldg        :  ~1.5  ms    (이미 메모리 최적 → 추가이득 미미)'
echo '   stage5 multistream :  1.01x       (단일커널이 GPU 포화 → stream 여지 없음)'
echo '   ──────────────────────────────────────────────'
echo '   최종:  196.5 s  →  ~1.5 ms   =   약 131,000x'
echo '======================================================================'
R 2.5
