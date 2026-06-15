"""S&P 500 daily returns 1928-2026 → binary (f64/f32) 변환.

원본 BTC와 동일한 파이프라인 (log returns, MA20 crossover로 strategy returns).
"""
import os
import numpy as np
import yfinance as yf
import pandas as pd

OUT_DIR = os.path.expanduser("~/cuda_bootstrap/")

print("Downloading ^GSPC daily 1928-01-01 to 2026-06-02 ...")
df = yf.download("^GSPC", start="1928-01-01", end="2026-06-02", auto_adjust=True, progress=False)
print(f"Rows: {len(df)}")
print(f"Range: {df.index[0]} to {df.index[-1]}")

close = df["Close"].values.flatten().astype(np.float64)
print(f"Close shape: {close.shape}")

# Log returns
log_ret = np.diff(np.log(close))
print(f"Log returns shape: {log_ret.shape}")

# MA(20) / MA(50) crossover strategy returns (BTC와 동일 baseline)
ma20 = pd.Series(close).rolling(20).mean().values
ma50 = pd.Series(close).rolling(50).mean().values

# signal: ma20 > ma50 then long else short
signal = np.where(ma20[:-1] > ma50[:-1], 1.0, -1.0)
# Strategy returns: signal[t-1] * log_ret[t]
strat_ret = signal[1:] * log_ret[1:]
# Remove NaN (early window before MA available)
strat_ret = strat_ret[~np.isnan(strat_ret)]

print(f"Strategy returns shape (after NaN removal): {strat_ret.shape}")
print(f"Mean: {strat_ret.mean():.6f}, Std: {strat_ret.std():.6f}")
print(f"Annualized Sharpe (sqrt 252): {strat_ret.mean()/strat_ret.std()*np.sqrt(252):.4f}")

# Save f64 + f32 binary
ret_f64 = strat_ret.astype(np.float64)
ret_f32 = strat_ret.astype(np.float32)

ret_f64.tofile(os.path.join(OUT_DIR, "sp500_returns.bin"))
ret_f32.tofile(os.path.join(OUT_DIR, "sp500_returns_f32.bin"))

print(f"\nSaved:")
print(f"  {OUT_DIR}sp500_returns.bin  ({ret_f64.nbytes:,} bytes, f64)")
print(f"  {OUT_DIR}sp500_returns_f32.bin  ({ret_f32.nbytes:,} bytes, f32)")
print(f"  N = {len(ret_f32)}")
