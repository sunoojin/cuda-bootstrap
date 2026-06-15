# Stationary Block Bootstrap — CUDA 가속

2026-1 학기 병렬처리 기말 프로젝트.

## 한 줄 요약

시계열 Sharpe ratio significance test를 위한 Stationary Block Bootstrap
(Politis & Romano 1994)의 CUDA 구현.
CPU(numpy) baseline 대비 단계별 최적화로 **XXx speedup** 달성.

> 실제 측정값은 `benchmark/results/sweep_20260505_014802.txt` 참고.

## 왜 이 알고리즘?

- **Embarrassingly parallel**: 각 permutation 독립 → block-level parallelism 자연스러움
- **풍부한 아키텍처 적용 면적**: FP32 벡터 로드 / cuRAND / `__ldg()` 캐시 / multi-stream
- **명확한 baseline**: numpy + scipy.stats 단순 비교
- **현실 효용**: time series 통계 검정 표준 기법

## 단계별 최적화

| Stage | 구현                | 핵심 기법                | 실측 speedup |
| ----- | ------------------- | ------------------------ | ------------ |
| 0     | CPU numpy baseline  | —                        | 1.0x         |
| 0-mp  | CPU multiprocessing | Python multiprocessing   | Xx           |
| 1     | Naive CUDA          | 1 thread / permutation   | Xx           |
| 2     | Block-level         | 1 block / permutation    | Xx           |
| 3     | FP32 + cuRAND4      | float4 벡터 로드, cuRAND | Xx           |
| 4a    | LDG                 | `__ldg()` 읽기 캐시      | Xx           |
| 5     | Multi-stream        | CUDA 스트림 파이프라인   | Xx           |

> **TODO**: 벤치마크 결과값으로 `Xx` 채워주세요.

## 데이터

- **S&P 500**: `fetch_sp500.py`로 수집, `sp500_returns.bin` / `sp500_returns_f32.bin`
- **BTC**: Binance 공개 API BTCUSDT 1h OHLCV, `btc_returns.bin` / `btc_returns_f32.bin`
- 트레이딩 alpha 또는 proprietary 데이터 **미포함** (격리 원칙)
- `.bin` 파일은 `.gitignore` 처리 — `fetch_sp500.py`로 재생성 가능

## 디렉토리 구조

```
cuda_bootstrap/

├── README.md

├── fetch_sp500.py          # S&P 500 데이터 수집

├── python_wrapper.py       # CUDA 커널 Python 인터페이스

├── demo_run.sh             # 빠른 실행 데모

├── measure_sp500.sh        # S&P 500 벤치마크

├── sweep_sp500_n.sh        # N 스윕 실험

├── baseline/               # CPU 참조 구현

│   ├── bootstrap_cpu.py    # numpy 단일 프로세스

│   └── bootstrap_cpu_mp.py # multiprocessing 버전

├── cuda/                   # CUDA 커널 단계별 구현

│   ├── Makefile

│   ├── stage1_naive.cu

│   ├── stage2_block.cu

│   ├── stage3_fp32_curand4.cu

│   ├── stage4a_ldg.cu

│   └── stage5_multistream.cu

└── benchmark/              # 성능 측정

├── run_bench.py

├── sweep.sh

├── polling_solo.sh

├── validate_ks.py

└── results/

└── sweep_20260505_014802.txt
```

## 빠른 시작

### Requirements

- CUDA 12.x
- Python 3.10+

```bash
pip install numpy pandas yfinance scipy
```

### Build & Run

```bash
# 1. 데이터 수집
python fetch_sp500.py

# 2. CUDA 커널 빌드
cd cuda && make all && cd ..

# 3. 데모 실행
bash demo_run.sh

# 4. 전체 벤치마크 스윕
bash benchmark/sweep.sh
```

## 환경

- 실행/프로파일: RTX A6000 (CUDA 12.x, Nsight Compute)
- 개발: macOS (VSCode + Remote-SSH)

## 참고 문헌

- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. _JASA_, 89(428), 1303–1313.
- NVIDIA CUDA C++ Programming Guide
- _Programming Massively Parallel Processors_ (Hwu, Kirk, Hajj)

## 격리 원칙

- 트레이딩 전략 코드/데이터 업로드 **금지**
- Bootstrap 라이브러리는 generic, 공개 데이터로 시연만
