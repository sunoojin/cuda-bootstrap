#!/bin/bash
# N × B sweep on stage 1 vs stage 2.
# Run on school server: bash benchmark/sweep.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p benchmark/results

OUT=benchmark/results/sweep_$(date +%Y%m%d_%H%M%S).txt
echo "# N B stage time_ms throughput_Mperms_s null_mean" > "$OUT"

for N in 1000 5000 10000 55495; do
    # Make truncated returns binary
    python3 -c "
import numpy as np
r = np.fromfile('btc_returns.bin', dtype=np.float64)
r[:$N].tofile('btc_returns_${N}.bin')
"
    for B in 1000 5000 10000 50000; do
        for STAGE in stage1_naive stage2_block; do
            OUT_LINE=$(CUDA_VISIBLE_DEVICES=1 ./cuda/$STAGE btc_returns_${N}.bin $N $B 0.0417 42)
            T=$(echo "$OUT_LINE" | grep -oP 'elapsed=\K[0-9.]+')
            TP=$(echo "$OUT_LINE" | grep -oP 'throughput=\K[0-9.]+')
            NM=$(echo "$OUT_LINE" | grep -oP 'null_mean=\K[+-]?[0-9.]+')
            echo "$N $B $STAGE $T $TP $NM" | tee -a "$OUT"
        done
    done
    rm -f btc_returns_${N}.bin
done

echo
echo "=== Summary saved to: $OUT ==="
