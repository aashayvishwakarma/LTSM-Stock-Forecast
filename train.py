import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from lstm import LSTMModel

TICKER = "AAPL"
HIDDEN = 64
SEQ_LEN = 30
HORIZON = 5
EPOCHS = 100
LR = 1e-3
SPLIT = 0.8
SEED = 42
START = "2015-01-01"
END = "2025-01-01"
OUT_DIR = "outputs"


def fetch_close_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} to {end}).")
    return df["Close"].to_numpy(dtype=np.float64).reshape(-1, 1)


def build_windows(scaled, seq_len, horizon, start_idx, end_idx):
    inputs_batch = []
    targets_batch = []
    last_i = end_idx - seq_len - horizon
    for i in range(start_idx, last_i + 1):
        seq_in = [scaled[i + t : i + t + 1].copy() for t in range(seq_len)]
        tgt = scaled[i + seq_len : i + seq_len + horizon].copy()
        inputs_batch.append(seq_in)
        targets_batch.append(tgt)
    return inputs_batch, targets_batch


def inverse_matrix(scaler, arr):
    flat = arr.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(arr.shape)


def rmse_mape(pred, actual):
    diff = pred - actual
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = np.maximum(actual, 1e-8)
    mape = float(np.mean(np.abs(diff / denom)) * 100.0)
    return rmse, mape


def evaluate(model, inputs_list, targets_list, scaler):
    preds = []
    tgts = []
    for x_seq, y in zip(inputs_list, targets_list):
        y_hat, _ = model.forward(x_seq)
        preds.append(y_hat.copy())
        tgts.append(y.copy())
    pred_scaled = np.stack([p.reshape(-1) for p in preds], axis=0)
    tgt_scaled = np.stack([t.reshape(-1) for t in tgts], axis=0)
    pred_inv = inverse_matrix(scaler, pred_scaled)
    tgt_inv = inverse_matrix(scaler, tgt_scaled)
    return (*rmse_mape(pred_inv, tgt_inv), pred_inv, tgt_inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", default="default")
    ap.add_argument("--log", default="experiment_log.csv")
    ap.add_argument("--ticker", default=None, help="override TICKER in this file")
    args = ap.parse_args()

    ticker = args.ticker or TICKER

    os.makedirs(OUT_DIR, exist_ok=True)

    close = fetch_close_prices(ticker, START, END)
    n = len(close)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    train_end = int(n * SPLIT)
    train_inputs, train_targets = build_windows(scaled, SEQ_LEN, HORIZON, 0, train_end)
    test_inputs, test_targets = build_windows(scaled, SEQ_LEN, HORIZON, train_end, n)

    if not train_inputs:
        raise RuntimeError("No training windows.")
    if not test_inputs:
        raise RuntimeError("No test windows.")

    model = LSTMModel(1, HIDDEN, HORIZON, seed=SEED)
    train_indices = np.arange(len(train_inputs))

    best_rmse = float("inf")
    best_epoch = -1
    last_train_loss = 0.0
    last_test_rmse = 0.0
    last_test_mape = 0.0

    for epoch in range(1, EPOCHS + 1):
        np.random.default_rng(SEED + epoch).shuffle(train_indices)
        losses = []
        for idx in train_indices:
            x_seq = train_inputs[idx]
            y = train_targets[idx]
            pred, caches = model.forward(x_seq)
            losses.append(model.backward(pred, y, caches))
            model.apply_updates(LR)

        last_train_loss = float(np.mean(losses))
        test_rmse, test_mape, pred_inv, tgt_inv = evaluate(model, test_inputs, test_targets, scaler)
        last_test_rmse = test_rmse
        last_test_mape = test_mape
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch

        last_pred_inv = pred_inv.copy()
        last_tgt_inv = tgt_inv.copy()

        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch}/{EPOCHS}  train_loss={last_train_loss:.6f}  "
                f"test_rmse={test_rmse:.6f}  test_mape={test_mape:.4f}%"
            )

    pred_path = os.path.join(OUT_DIR, f"{args.exp_id}_predictions.npy")
    tgt_path = os.path.join(OUT_DIR, f"{args.exp_id}_targets.npy")
    np.save(pred_path, last_pred_inv)
    np.save(tgt_path, last_tgt_inv)

    meta_path = os.path.join(OUT_DIR, f"{args.exp_id}_meta.npz")
    np.savez(
        meta_path,
        horizon=HORIZON,
        seq_len=SEQ_LEN,
        best_epoch=best_epoch,
        best_test_rmse=best_rmse,
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_id": args.exp_id,
        "ticker": ticker,
        "hidden": HIDDEN,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
        "epochs": EPOCHS,
        "lr": LR,
        "split": SPLIT,
        "seed": SEED,
        "train_loss_final": last_train_loss,
        "test_rmse_final": last_test_rmse,
        "test_mape_final": last_test_mape,
        "test_rmse_best": best_rmse,
        "best_epoch": best_epoch,
        "pred_path": pred_path,
        "tgt_path": tgt_path,
    }

    log_path = args.log
    new_file = not os.path.isfile(log_path)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"Saved predictions to {pred_path} and log to {log_path}")


if __name__ == "__main__":
    main()
