#!/bin/bash
set -e
cd ~/cuda_bootstrap

echo "=== Building CUDA binaries ==="
cd cuda && make all 2>&1 | tail -3 && cd ..

N=24717
B=10000
P=0.041667  # 1/24

echo ""
echo "=== Stage 1 — naive CUDA ==="
CUDA_VISIBLE_DEVICES=3 ./cuda/stage1_naive sp500_returns.bin $N $B $P 42

echo ""
echo "=== Stage 2 — block + warp shuffle ==="
CUDA_VISIBLE_DEVICES=3 ./cuda/stage2_block sp500_returns.bin $N $B $P 42

echo ""
echo "=== Stage 3 — fp32 + curand4 ==="
CUDA_VISIBLE_DEVICES=3 ./cuda/stage3_fp32_curand4 sp500_returns_f32.bin $N $B $P 42

echo ""
echo "=== Stage 4a — explicit __ldg ==="
CUDA_VISIBLE_DEVICES=3 ./cuda/stage4a_ldg sp500_returns_f32.bin $N $B $P 42

echo ""
echo "=== Stage 5 — multi-stream ==="
CUDA_VISIBLE_DEVICES=3 ./cuda/stage5_multistream sp500_returns_f32.bin $N $B $P 42 4 2>&1 | tail -5

echo ""
echo "=== Stage 3 — 5 trials (median for solo) ==="
for i in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=3 ./cuda/stage3_fp32_curand4 sp500_returns_f32.bin $N $B $P $((42+i)) 2>&1 | grep -oE "elapsed=[0-9.]+ ms" | head -1
done
