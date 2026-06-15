#!/usr/bin/env bash
# GPU idle 감지 + solo 측정 자동 실행.
# 매 180s마다 GPU 0 utilization 확인 → < 20% 시 batch 측정 → 결과 누적 → exit.
# 사용: ssh server 'nohup bash ~/cuda_bootstrap/benchmark/polling_solo.sh > /tmp/polling.log 2>&1 &'

set -u
LOG=/tmp/solo_results.txt
BENCH_DIR=$HOME/cuda_bootstrap
IDLE_THRESHOLD=20      # %
POLL_INTERVAL=180      # sec
MAX_WAIT=21600         # 6 hours
ELAPSED=0

cd "$BENCH_DIR"

echo "=== Polling Solo Bench started at $(date) ===" | tee -a "$LOG"

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "[$(date +%H:%M:%S)] GPU0 util=${UTIL}% (waited ${ELAPSED}s)" | tee -a "$LOG"

    if [ "$UTIL" -lt "$IDLE_THRESHOLD" ]; then
        # 한 번 더 확인 (transient 회피)
        sleep 10
        UTIL2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "$UTIL2" -lt "$IDLE_THRESHOLD" ]; then
            echo "=== GPU IDLE confirmed (util=${UTIL2}%). Running solo benchmarks ===" | tee -a "$LOG"
            break
        fi
    fi
    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))
done

if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
    echo "=== Timeout waiting for GPU idle. Exit. ===" | tee -a "$LOG"
    exit 1
fi

echo "" | tee -a "$LOG"
echo "=== Solo bench: Stage 1 / 2 / 3 / 4a (N=55495 B=10000) — 5 trials each ===" | tee -a "$LOG"
for trial in 1 2 3 4 5; do
    echo "--- trial $trial ---" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 ./cuda/stage1_naive          btc_returns.bin     55495 10000 0.0417 42 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 ./cuda/stage2_block          btc_returns.bin     55495 10000 0.0417 42 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 ./cuda/stage3_fp32_curand4   btc_returns_f32.bin 55495 10000 0.0417 42 2>&1 | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0 ./cuda/stage4a_ldg           btc_returns_f32.bin 55495 10000 0.0417 42 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== Stage 5 multi-stream solo (B=1000 / B=100 sub-saturation) ===" | tee -a "$LOG"
for B in 100 1000 10000; do
    for K in 12 24; do
        for NS in 1 4 8; do
            CUDA_VISIBLE_DEVICES=0 ./cuda/stage5_multistream btc_returns_f32.bin 55495 $B 0.0417 42 $K $NS 2>&1 | tee -a "$LOG"
            echo "---" | tee -a "$LOG"
        done
    done
done

echo "" | tee -a "$LOG"
echo "=== Solo bench finished at $(date) ===" | tee -a "$LOG"
