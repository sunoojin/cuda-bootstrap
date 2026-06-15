"""CPU baseline ↔ CUDA stage 별 벤치마크 driver.

Stage 1만 구현된 시점에서 CPU vs GPU naive 비교. Stage 추가될 때마다 row 추가.
"""
import os
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def cpu_baseline(returns: np.ndarray, B: int, expected_block_len: float) -> tuple[float, float]:
    import sys
    sys.path.insert(0, os.path.join(ROOT, "baseline"))
    from bootstrap_cpu import stationary_block_bootstrap

    t0 = time.perf_counter()
    sharpes = stationary_block_bootstrap(returns, B, expected_block_len, seed=0)
    elapsed = time.perf_counter() - t0
    return elapsed, float(sharpes.mean())


def cuda_stage(stage_bin: str, returns: np.ndarray, B: int, expected_block_len: float) -> str:
    bin_path = os.path.join(ROOT, "cuda", stage_bin)
    if not os.path.exists(bin_path):
        return f"{stage_bin}: NOT BUILT (run `make` in cuda/)"

    bin_input = os.path.join(ROOT, "data", "_bench_input.bin")
    returns.astype(np.float64).tofile(bin_input)

    p = 1.0 / expected_block_len
    cmd = [bin_path, bin_input, str(len(returns)), str(B), f"{p:.10f}", "42"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}"


def main():
    returns_path = os.path.join(ROOT, "data", "btc_1h_ma_crossover_returns.npy")
    if not os.path.exists(returns_path):
        print(f"missing {returns_path} — run `python data/fetch_btc.py` first.")
        return
    returns = np.load(returns_path).astype(np.float64)
    returns_centered = returns - returns.mean()
    print(f"loaded returns: N={len(returns)}, sharpe={returns.mean()/returns.std():.4f}")

    B = 1000
    block_len = 24.0

    print("\n=== CPU baseline ===")
    cpu_elapsed, cpu_null_mean = cpu_baseline(returns_centered, B, block_len)
    print(f"  numpy: B={B} elapsed={cpu_elapsed:.2f}s null_mean={cpu_null_mean:.6f}")

    print("\n=== CUDA stages ===")
    for stage in ["stage1_naive"]:
        out = cuda_stage(stage, returns_centered, B, block_len)
        print(f"  {out}")


if __name__ == "__main__":
    main()
