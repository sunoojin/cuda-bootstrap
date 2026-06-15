"""Multi-core CPU baseline using joblib.

Stage 0b: 24-core EPYC 활용 시 single-core 대비 어디까지 가는지 측정.
GPU vs CPU 공정 비교를 위해 'best CPU effort'를 명시적으로 보고.
"""
import os
import time

import numpy as np
from joblib import Parallel, delayed


def _bootstrap_chunk(returns: np.ndarray, B_chunk: int, expected_block_len: float, seed: int) -> np.ndarray:
    """B_chunk개 permutation의 Sharpe 분포 (single worker)."""
    n = len(returns)
    p = 1.0 / expected_block_len
    rng = np.random.default_rng(seed)
    sharpes = np.empty(B_chunk, dtype=np.float64)

    for b in range(B_chunk):
        idx = np.empty(n, dtype=np.int64)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(n)
            else:
                i = (i + 1) % n
        resampled = returns[idx]
        mu = resampled.mean()
        sd = resampled.std(ddof=1)
        sharpes[b] = mu / sd if sd > 0 else 0.0

    return sharpes


def stationary_bootstrap_mp(returns: np.ndarray, B: int, expected_block_len: float, n_jobs: int = -1, seed: int = 0) -> np.ndarray:
    """Multi-process. Workers 사이 RNG 독립."""
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    chunks = [(B + n_jobs - 1) // n_jobs] * n_jobs
    chunks[-1] = B - sum(chunks[:-1])  # last chunk balances
    seeds = [seed * 10000 + i for i in range(n_jobs)]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_bootstrap_chunk)(returns, c, expected_block_len, s) for c, s in zip(chunks, seeds) if c > 0
    )
    return np.concatenate(results)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bootstrap_cpu import stationary_block_bootstrap

    rng = np.random.default_rng(42)
    n = 55495
    returns = rng.normal(0.0, 0.01, n).astype(np.float64)

    print(f"N={n}, cores={os.cpu_count()}")
    for B in [1000]:
        t0 = time.perf_counter()
        s_single = stationary_block_bootstrap(returns, B, 24.0, seed=0)
        e_single = time.perf_counter() - t0
        print(f"  single-core B={B}: {e_single:.2f}s  null_mean={s_single.mean():+.6f}")

        for n_jobs in [4, 8, 16, 24, 48]:
            t0 = time.perf_counter()
            s_mp = stationary_bootstrap_mp(returns, B, 24.0, n_jobs=n_jobs, seed=0)
            e_mp = time.perf_counter() - t0
            print(f"  mp({n_jobs:>2})    B={B}: {e_mp:.2f}s  speedup={e_single/e_mp:.1f}x  null_mean={s_mp.mean():+.6f}")
