# LSTM Stock Forecast (NumPy from Scratch)

Single-layer LSTM with manual forward/BPTT, Adam updates, and gradient clipping, trained on adjusted closing prices from Yahoo Finance.

## Setup

```bash
pip install numpy yfinance scikit-learn matplotlib
```

Optional: use a virtual environment before installing.

## Train

Default data: **AAPL** adjusted close from **2015-01-01** through **2024** (`--end 2025-01-01`). The train/test split is **chronological** (no temporal leakage). Each epoch shuffles **training windows only**.

```bash
python train.py \
  --ticker AAPL \
  --hidden 64 \
  --seq_len 30 \
  --horizon 5 \
  --epochs 100 \
  --lr 0.001 \
  --split 0.8 \
  --exp_id run_aapl_baseline \
  --log experiment_log.csv
```

Artifacts:

- `outputs/{exp_id}_predictions.npy` — test predictions (original price scale)
- `outputs/{exp_id}_targets.npy` — matching test targets
- `outputs/{exp_id}_meta.npz` — horizon, seq_len, best epoch, etc.
- Appends one row per run to `experiment_log.csv` (path configurable with `--log`)

## Plot

After training, generate PNGs from saved arrays:

```bash
python plot_results.py --exp_id run_aapl_baseline --out_dir outputs --plot_dir plots
```

Outputs:

1. `{exp_id}_plot1_t1_series.png` — actual vs predicted **t+1** close over test windows  
2. `{exp_id}_plot2_rmse_by_step.png` — RMSE at each horizon step **t+1 … t+horizon**  
3. `{exp_id}_plot3_single_window.png` — one sample’s full multi-step curve  

## Suggested experiments (8 runs)

| # | Ticker | Hidden | `seq_len` | `horizon` | Learning rate | Notes |
|---|--------|--------|-----------|-----------|---------------|--------|
| 1 | AAPL | 32 | 10 | 5 | 0.001 | Small memory, short context |
| 2 | AAPL | 64 | 30 | 5 | 0.001 | Balanced default |
| 3 | AAPL | 128 | 60 | 5 | 0.001 | Large hidden + long input window |
| 4 | AAPL | 64 | 30 | 10 | 0.001 | Longer forecast horizon |
| 5 | AAPL | 64 | 30 | 5 | 0.0003 | Lower LR (more stable) |
| 6 | AAPL | 64 | 30 | 5 | 0.003 | Higher LR (faster, may diverge) |
| 7 | TSLA | 64 | 30 | 5 | 0.001 | Different ticker, same architecture |
| 8 | TSLA | 128 | 60 | 10 | 0.001 | Heavier model on volatile name |

Example commands:

```bash
python train.py --ticker AAPL --hidden 32 --seq_len 10 --horizon 5 --lr 0.001 --exp_id e01 --epochs 150
python train.py --ticker TSLA --hidden 128 --seq_len 60 --horizon 10 --lr 0.001 --exp_id e08 --epochs 150
python plot_results.py --exp_id e01
```

## Metrics

Training minimizes **MSE in normalized space**. Reported **RMSE** and **MAPE** use **inverse-transformed** prices (original scale), computed on the test set each epoch.

## Implementation notes

- Core LSTM math is **NumPy only** (no PyTorch/TensorFlow/autograd).
- Allowed extras: **yfinance**, **sklearn.preprocessing** (scaling), **matplotlib** (plots).
