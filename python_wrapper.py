"""
Stationary Block Bootstrap — Python wrapper.

Single function `bootstrap_sharpe(returns, B, p, seed, K)` that auto-selects:
  - K == 1 → stage3 (fp32 + curand4, single-strategy throughput)
  - K  > 1 → stage5 (multi-stream variant sweep)

Returns dict with walltime, throughput, null_mean, and (for K==1) sharpes array.

Usage:
    import numpy as np
    from python_wrapper import bootstrap_sharpe
    ret = np.load('btc_returns.npy').astype(np.float32)
    out = bootstrap_sharpe(ret, B=10000, p=1/24, seed=42)
    print(out['throughput_Mperms_s'], out['sharpes'].shape)
"""
import os
import re
import subprocess
import tempfile
import numpy as np

BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cuda')
STAGE3 = os.path.join(BIN_DIR, 'stage3_fp32_curand4')
STAGE5 = os.path.join(BIN_DIR, 'stage5_multistream')


def _check_binary(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"binary not found: {path}. Run `make all` in cuda/.")


def _parse_stage3(stdout):
    m = re.search(
        r"Stage3.*?N=(\d+)\s+B=(\d+).*?elapsed=([\d.]+)ms\s+throughput=([\d.]+)\s+Mperms/s\s+null_mean=([-\d.]+)",
        stdout,
    )
    if not m:
        raise RuntimeError(f"stage3 output parse failed:\n{stdout}")
    return {
        'stage': 3, 'N': int(m.group(1)), 'B': int(m.group(2)),
        'elapsed_ms': float(m.group(3)),
        'throughput_Mperms_s': float(m.group(4)),
        'null_mean': float(m.group(5)),
    }


def _parse_stage5(stdout):
    m = re.search(r"N=(\d+)\s+B=(\d+)\s+K=(\d+)\s+NSTREAMS=(\d+)", stdout)
    seq = re.search(r"Sequential\s*:\s*([\d.]+)\s*ms", stdout)
    ms_ = re.search(r"Multistream:\s*([\d.]+)\s*ms", stdout)
    sp = re.search(r"Speedup\s*:\s*([\d.]+)x", stdout)
    null_ = re.search(r"null_mean.*?:\s*([-\d.]+)", stdout)
    if not (m and seq and ms_ and sp):
        raise RuntimeError(f"stage5 output parse failed:\n{stdout}")
    return {
        'stage': 5, 'N': int(m.group(1)), 'B': int(m.group(2)),
        'K': int(m.group(3)), 'NSTREAMS': int(m.group(4)),
        'sequential_ms': float(seq.group(1)),
        'multistream_ms': float(ms_.group(1)),
        'speedup': float(sp.group(1)),
        'null_mean': float(null_.group(1)) if null_ else None,
    }


def bootstrap_sharpe(returns, B=10000, p=1/24, seed=42, K=1, n_streams=4,
                    dump_sharpes=True, gpu_id=0):
    """
    Run stationary block bootstrap on GPU.

    Parameters
    ----------
    returns : np.ndarray (1D, float32 or float64)
        Daily/hourly returns. Auto-cast to float32.
    B : int
        Number of bootstrap permutations.
    p : float
        Block-length inverse (1/expected_block_length). Default 1/24 = avg block len 24.
    seed : int
        cuRAND seed.
    K : int
        Number of strategy variants. K=1 → stage3, K>1 → stage5 multi-stream.
    n_streams : int
        Streams for stage5. Ignored if K=1.
    dump_sharpes : bool
        If True (K=1 only), return per-permutation sharpes as np.ndarray.
    gpu_id : int
        CUDA_VISIBLE_DEVICES.

    Returns
    -------
    dict with stage info, walltime, throughput, null_mean, and optionally sharpes (K=1).
    """
    if returns.ndim != 1:
        raise ValueError("returns must be 1D")
    N = len(returns)
    ret_f32 = returns.astype(np.float32, copy=False)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    with tempfile.TemporaryDirectory() as tmp:
        bin_path = os.path.join(tmp, 'returns_f32.bin')
        ret_f32.tofile(bin_path)

        if K == 1:
            _check_binary(STAGE3)
            dump_path = os.path.join(tmp, 'sharpes.bin') if dump_sharpes else ''
            cmd = [STAGE3, bin_path, str(N), str(B), str(p), str(seed)]
            if dump_path:
                cmd.append(dump_path)
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"stage3 failed (rc={r.returncode}):\n{r.stderr}")
            out = _parse_stage3(r.stdout)
            if dump_sharpes and os.path.exists(dump_path):
                out['sharpes'] = np.fromfile(dump_path, dtype=np.float64)
            return out
        else:
            _check_binary(STAGE5)
            cmd = [STAGE5, bin_path, str(N), str(B), str(p), str(seed),
                   str(K), str(n_streams)]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(f"stage5 failed (rc={r.returncode}):\n{r.stderr}")
            return _parse_stage5(r.stdout)


def sharpe_pvalue(observed_sharpe, null_sharpes):
    """Two-sided p-value: P(|null| >= |observed|)."""
    return float(np.mean(np.abs(null_sharpes) >= abs(observed_sharpe)))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('returns_bin', help='float32 binary of returns')
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--p', type=float, default=1/24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--K', type=int, default=1)
    ap.add_argument('--streams', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    ret = np.fromfile(args.returns_bin, dtype=np.float32)[:args.N]
    out = bootstrap_sharpe(ret, B=args.B, p=args.p, seed=args.seed,
                           K=args.K, n_streams=args.streams, gpu_id=args.gpu)
    for k, v in out.items():
        if k == 'sharpes':
            print(f"  sharpes: shape={v.shape} mean={v.mean():.6f} std={v.std():.6f}")
        else:
            print(f"  {k}: {v}")
