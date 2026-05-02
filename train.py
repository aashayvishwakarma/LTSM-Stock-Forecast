"""
Train NumPy LSTM on stock closes (yfinance + sklearn scaling).
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from lstm import LSTMModel


def fetch_close_prices(ticker: str, start: str, end: str) -> np.ndarray:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} to {end}).")
    close = df["Close"].to_numpy(dtype=np.float64).reshape(-1, 1)
    return close


def build_windows(
    scaled: np.ndarray,
    seq_len: int,
    horizon: int,
    start_idx: int,
    end_idx: int,
) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
    """
    Sliding windows with indices i such that [i, i+seq_len+horizon) ⊆ [start_idx, end_idx).
    """
    inputs_batch: list[list[np.ndarray]] = []
    targets_batch: list[np.ndarray] = []
    last_i = end_idx - seq_len - horizon
    for i in range(start_idx, last_i + 1):
        seq_in = [scaled[i + t : i + t + 1].copy() for t in range(seq_len)]
        tgt = scaled[i + seq_len : i + seq_len + horizon].copy()
        inputs_batch.append(seq_in)
        targets_batch.append(tgt)
    return inputs_batch, targets_batch


def inverse_matrix(scaler: MinMaxScaler, arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(arr.shape)


def rmse_mape(pred: np.ndarray, actual: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    diff = pred - actual
    rmse = float(np.sqrt(np.mean(diff**2)))
    mape = float(np.mean(np.abs(diff / (actual + eps))) * 100.0)
    return rmse, mape


def evaluate(
    model: LSTMModel,
    inputs_list: list[list[np.ndarray]],
    targets_list: list[np.ndarray],
    scaler: MinMaxScaler,
) -> tuple[float, float, np.ndarray, np.ndarray]:
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
    r, m = rmse_mape(pred_inv, tgt_inv)
    return r, m, pred_inv, tgt_inv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NumPy LSTM stock forecaster")
    p.add_argument("--ticker", type=str, default="AAPL")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--split", type=float, default=0.8, help="Fraction of timesteps for training region.")
    p.add_argument("--exp_id", type=str, default="default")
    p.add_argument("--log", type=str, default="experiment_log.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, default="2015-01-01")
    p.add_argument("--end", type=str, default="2025-01-01")
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    close = fetch_close_prices(args.ticker, args.start, args.end)
    n = len(close)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    train_end = int(n * args.split)
    train_inputs, train_targets = build_windows(scaled, args.seq_len, args.horizon, 0, train_end)
    test_inputs, test_targets = build_windows(scaled, args.seq_len, args.horizon, train_end, n)

    if len(train_inputs) == 0:
        raise RuntimeError("No training windows; increase data or decrease seq_len/horizon.")
    if len(test_inputs) == 0:
        raise RuntimeError("No test windows; increase split or data length.")

    input_size = 1
    model = LSTMModel(input_size, args.hidden, args.horizon, seed=args.seed)

    train_indices = np.arange(len(train_inputs))

    best_rmse = float("inf")
    best_epoch = -1
    last_train_loss = 0.0
    last_test_rmse = 0.0
    last_test_mape = 0.0
    last_pred_inv: np.ndarray | None = None
    last_tgt_inv: np.ndarray | None = None

    for epoch in range(1, args.epochs + 1):
        np.random.default_rng(args.seed + epoch).shuffle(train_indices)
        epoch_losses: list[float] = []

        for idx in train_indices:
            x_seq = train_inputs[idx]
            y = train_targets[idx]
            pred, caches = model.forward(x_seq)
            loss = model.backward(pred, y, caches)
            model.apply_updates(args.lr)
            epoch_losses.append(loss)

        last_train_loss = float(np.mean(epoch_losses))

        test_rmse, test_mape, pred_inv, tgt_inv = evaluate(model, test_inputs, test_targets, scaler)
        last_test_rmse = test_rmse
        last_test_mape = test_mape

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch

        last_pred_inv = pred_inv.copy()
        last_tgt_inv = tgt_inv.copy()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch}/{args.epochs}  train_loss={last_train_loss:.6f}  "
                f"test_rmse={test_rmse:.6f}  test_mape={test_mape:.4f}%"
            )

    pred_path = os.path.join(args.out_dir, f"{args.exp_id}_predictions.npy")
    tgt_path = os.path.join(args.out_dir, f"{args.exp_id}_targets.npy")
    if last_pred_inv is None or last_tgt_inv is None:
        raise RuntimeError("No evaluation results to save.")
    np.save(pred_path, last_pred_inv)
    np.save(tgt_path, last_tgt_inv)

    meta_path = os.path.join(args.out_dir, f"{args.exp_id}_meta.npz")
    np.savez(
        meta_path,
        horizon=args.horizon,
        seq_len=args.seq_len,
        best_epoch=best_epoch,
        best_test_rmse=best_rmse,
        ticker=np.array(args.ticker),
    )

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exp_id": args.exp_id,
        "ticker": args.ticker,
        "hidden": args.hidden,
        "seq_len": args.seq_len,
        "horizon": args.horizon,
        "epochs": args.epochs,
        "lr": args.lr,
        "split": args.split,
        "seed": args.seed,
        "train_loss_final": last_train_loss,
        "test_rmse_final": last_test_rmse,
        "test_mape_final": last_test_mape,
        "test_rmse_best": best_rmse,
        "best_epoch": best_epoch,
        "pred_path": pred_path,
        "tgt_path": tgt_path,
    }

    file_exists = os.path.isfile(args.log)
    with open(args.log, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved predictions to {pred_path} and log to {args.log}")


if __name__ == "__main__":
    main()
