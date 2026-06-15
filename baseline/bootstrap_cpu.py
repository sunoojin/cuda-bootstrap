"""Stationary block bootstrap — numpy CPU 참조 구현.

Politis & Romano (1994). The stationary bootstrap.

Test statistic: Sharpe ratio (annualized).
Bootstrap p-value: H0 (Sharpe=0) 하 분포에서 관측 Sharpe의 percentile.
"""
import numpy as np


def stationary_block_bootstrap(returns: np.ndarray, B: int, expected_block_len: float, seed: int = 0) -> np.ndarray:
    """B개 permutation의 Sharpe ratio 분포 반환.

    expected_block_len: stationary block의 평균 길이 (1/p).
                        time series autocorrelation에 따라 선택 (annual ~10-50 typical for 1h returns).
    """
    n = len(returns)
    p = 1.0 / expected_block_len
    rng = np.random.default_rng(seed)
    sharpes = np.empty(B, dtype=np.float64)

    for b in range(B):
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


def sharpe_ratio(returns: np.ndarray) -> float:
    mu = returns.mean()
    sd = returns.std(ddof=1)
    return mu / sd if sd > 0 else 0.0


def bootstrap_pvalue(returns: np.ndarray, B: int = 5000, expected_block_len: float = 24.0, seed: int = 0) -> dict:
    """관측 Sharpe vs bootstrap null 분포 p-value 계산."""
    observed = sharpe_ratio(returns)
    null_dist = stationary_block_bootstrap(returns - returns.mean(), B, expected_block_len, seed)
    p_two_sided = 2 * min((null_dist >= observed).mean(), (null_dist <= observed).mean())
    return {
        "observed_sharpe": observed,
        "null_mean": float(null_dist.mean()),
        "null_std": float(null_dist.std(ddof=1)),
        "p_value": float(p_two_sided),
        "ci_95": (float(np.quantile(null_dist, 0.025)), float(np.quantile(null_dist, 0.975))),
    }


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(42)
    n = 50_000
    returns = rng.normal(0.0001, 0.01, n).astype(np.float64)

    t0 = time.perf_counter()
    result = bootstrap_pvalue(returns, B=1000, expected_block_len=24.0)
    elapsed = time.perf_counter() - t0
    print(f"n={n}, B=1000, elapsed={elapsed:.2f}s")
    for k, v in result.items():
        print(f"  {k}: {v}")
