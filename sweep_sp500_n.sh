#!/bin/bash
set -e
cd ~/cuda_bootstrap

B=10000
P=0.041667

echo "# N B stage time_ms speedup_vs_stage1"
for N in 1000 5000 10000 24717; do
    for STAGE in stage1_naive stage2_block stage3_fp32_curand4; do
        BIN_FILE="sp500_returns.bin"
        if [ "$STAGE" == "stage3_fp32_curand4" ]; then
            BIN_FILE="sp500_returns_f32.bin"
        fi
        # 5 trials median for robustness
        TIMES=()
        for i in 1 2 3 4 5; do
            T=$(CUDA_VISIBLE_DEVICES=3 ./cuda/$STAGE $BIN_FILE $N $B $P $((42+i)) 2>&1 | grep -oE "elapsed=[0-9.]+ms" | grep -oE "[0-9.]+")
            TIMES+=($T)
        done
        # Compute median (5 values, sorted, take 3rd)
        MEDIAN=$(printf "%s\n" "${TIMES[@]}" | sort -n | sed -n "3p")
        echo "N=$N stage=$STAGE median_ms=$MEDIAN trials=(${TIMES[*]})"
    done
done
