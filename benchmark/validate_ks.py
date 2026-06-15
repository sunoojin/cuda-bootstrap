"""
Stage 2 vs Stage 3 algorithmic equivalence — KS 2-sample test.

Both produce 'sharpes' arrays (length B) under H0 (no signal in returns).
Their distributions should be statistically indistinguishable if the
'1 chain of length N' vs '256 chains of length N/256' variant is asymptotically equivalent.

KS test:
    H0: Stage 2 and Stage 3 sharpes come from the same distribution
    Reject if p < 0.05.
    PASS = p > 0.05 (cannot reject equivalence).

Usage:
    1. Run Stage 2 with dump:
        ./stage2_block btc_returns.bin 55495 10000 0.0417 42 s2.bin
    2. Run Stage 3 with dump:
        ./stage3_fp32_curand4 btc_returns_f32.bin 55495 10000 0.0417 42 s3.bin
    3. python3 validate_ks.py s2.bin s3.bin
"""
import sys
import numpy as np
from scipy import stats


def main():
    if len(sys.argv) < 3:
        print("Usage: validate_ks.py s2.bin s3.bin", file=sys.stderr)
        sys.exit(1)

    s2 = np.fromfile(sys.argv[1], dtype=np.float64)
    s3 = np.fromfile(sys.argv[2], dtype=np.float64)

    print(f"Stage 2: N={len(s2):,} mean={s2.mean():.6f} std={s2.std():.6f}")
    print(f"Stage 3: N={len(s3):,} mean={s3.mean():.6f} std={s3.std():.6f}")
    print()

    # KS 2-sample test
    ks = stats.ks_2samp(s2, s3, alternative="two-sided")
    print(f"KS 2-sample test:")
    print(f"  D = {ks.statistic:.6f}")
    print(f"  p-value = {ks.pvalue:.6f}")
    print()

    if ks.pvalue > 0.05:
        print("✓ PASS — Cannot reject distributional equivalence (p > 0.05)")
    else:
        print(f"✗ FAIL — Distributions differ (p = {ks.pvalue:.4f})")

    # Additional: Anderson-Darling 2-sample (more sensitive in tails)
    ad = stats.anderson_ksamp([s2, s3])
    print()
    print(f"Anderson-Darling 2-sample:")
    print(f"  statistic = {ad.statistic:.4f}")
    print(f"  significance level = {ad.significance_level:.4f}")

    # Quick distributional summary
    print()
    print("Quantile comparison:")
    for q in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
        q2 = np.quantile(s2, q)
        q3 = np.quantile(s3, q)
        print(f"  q{q:.2f}: stage2={q2:+.4f}  stage3={q3:+.4f}  diff={q3-q2:+.4f}")


if __name__ == "__main__":
    main()
