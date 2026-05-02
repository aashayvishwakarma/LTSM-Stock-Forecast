# LSTM Stock Forecast (NumPy from Scratch)

Single-layer LSTM with manual forward/BPTT, Adam updates, and gradient clipping, trained on adjusted closing prices from Yahoo Finance.

## Setup

```bash
pip install numpy yfinance scikit-learn matplotlib
```

Optional: use a virtual environment before installing.

## Train

Hyperparameters (`TICKER`, `HIDDEN`, `SEQ_LEN`, `HORIZON`, `EPOCHS`, `LR`, `SPLIT`, dates, etc.) live at the **top of `train.py`** — edit there first.

Default data: **AAPL** adjusted close **2015-01-01** through **2024** (`END = "2025-01-01"`). Train/test split is **chronological**; each epoch shuffles **training windows only**.

CLI is minimal: pick a run label and optional log path.

```bash
python train.py --exp_id run_aapl_baseline --log experiment_log.csv
```

Artifacts:

- `outputs/{exp_id}_predictions.npy` — test predictions (original price scale)
- `outputs/{exp_id}_targets.npy` — matching test targets
- `outputs/{exp_id}_meta.npz` — horizon, seq_len, best epoch, etc.
- Appends one row per run to the CSV (`--log`, default `experiment_log.csv`)

## Plot

After training:

```bash
python plot_results.py run_aapl_baseline --out_dir outputs --plot_dir plots
```

`exp_id` is a **positional** argument (the label you passed to `--exp_id` when training).

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

Match each table row by editing the constants in `train.py`, then:

```bash
python train.py --exp_id e01
python train.py --exp_id e08
python plot_results.py e01
```

## Metrics

Training minimizes **MSE in normalized space**. Reported **RMSE** and **MAPE** use **inverse-transformed** prices (original scale), computed on the test set each epoch.

## Implementation notes

- Core LSTM math is **NumPy only** (no PyTorch/TensorFlow/autograd).
- Allowed extras: **yfinance**, **sklearn.preprocessing** (scaling), **matplotlib** (plots).
