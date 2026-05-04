import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from lstm import LSTMModel

# yFinance History Window
START = "2015-01-01"
END = "2025-01-01"
TICKER = "AAPL"

# Model + Training Hyperparameters
HIDDEN = 64
SEQ_LEN = 30
HORIZON = 5
EPOCHS = 100
LR = 1e-3
SPLIT = 0.8
SEED = 42

OUT_DIR = "outputs"


def fetch_close_prices(ticker, start, end):
    # Pull closing prices from yFinance
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} ({start} to {end}).")
    return df["Close"].to_numpy(dtype=np.float64).reshape(-1, 1)


def build_windows(scaled, seq_len, horizon, start_idx, end_idx):
    # Build training windows
    xs = []
    ys = []
    last_i = end_idx - seq_len - horizon
    for i in range(start_idx, last_i + 1):
        window = [scaled[i + t : i + t + 1].copy() for t in range(seq_len)]
        future = scaled[i + seq_len : i + seq_len + horizon].copy()
        xs.append(window)
        ys.append(future)
    return xs, ys


def unscale(scaler, arr):
    # Bring predictions back to real dollar scale
    # Flattens array into 2D array and then reshapes back to original shape
    flat = arr.reshape(-1, 1)
    back = scaler.inverse_transform(flat)
    return back.reshape(arr.shape)


def rmse_mape(pred, actual):
    # Calculate RMSE and MAPE
    # RMSE is the square root of the mean of the squared differences between the predicted and actual values
    # MAPE is the mean of the absolute percentage errors between the predicted and actual values
    diff = pred - actual
    rmse = float(np.sqrt(np.mean(diff**2)))
    denom = np.maximum(actual, 1e-8)
    mape = float(np.mean(np.abs(diff / denom)) * 100.0)
    return rmse, mape


def evaluate(model, inputs_list, targets_list, scaler):
    # Run inference on the model for each input sequence and target
    preds = []
    tgts = []
    for x_seq, y in zip(inputs_list, targets_list):
        y_hat, _ = model.forward(x_seq)
        preds.append(y_hat.copy())
        tgts.append(y.copy())

    # Stack predictions and targets into 2D arrays
    pred_scaled = np.stack([p.reshape(-1) for p in preds], axis=0)
    tgt_scaled = np.stack([t.reshape(-1) for t in tgts], axis=0)

    # Unscale predictions and targets back to original dollar scale
    pred_inv = unscale(scaler, pred_scaled)
    tgt_inv = unscale(scaler, tgt_scaled)
    return (*rmse_mape(pred_inv, tgt_inv), pred_inv, tgt_inv)


def main():

    # Parsing and validating command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", default="default")
    parser.add_argument("--log", default="experiment_log.csv")
    parser.add_argument("--ticker", default=None, help="override TICKER in this file")
    args = parser.parse_args()

    ticker = args.ticker or TICKER

    os.makedirs(OUT_DIR, exist_ok=True)

    # Data preparation
    close = fetch_close_prices(ticker, START, END)
    n = len(close)

    # Fit scaler to closing prices
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    # Split data into training and test sets
    train_end = int(n * SPLIT)
    train_inputs, train_targets = build_windows(scaled, SEQ_LEN, HORIZON, 0, train_end)
    test_inputs, test_targets = build_windows(scaled, SEQ_LEN, HORIZON, train_end, n)

    if not train_inputs:
        raise RuntimeError("No training windows.")
    if not test_inputs:
        raise RuntimeError("No test windows.")

    # Initialize LSTM model
    model = LSTMModel(1, HIDDEN, HORIZON, seed=SEED)
    train_indices = np.arange(len(train_inputs))

    best_rmse = float("inf")
    best_epoch = -1
    last_train_loss = 0.0
    last_test_rmse = 0.0
    last_test_mape = 0.0

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        # Shuffle training indices
        rng = np.random.default_rng(SEED + epoch)
        rng.shuffle(train_indices)
        losses = []
        for idx in train_indices:
            x_seq = train_inputs[idx]
            y = train_targets[idx]

            # Forward pass through LSTM model
            pred, caches = model.forward(x_seq)

            # Backward pass through LSTM model
            losses.append(model.backward(pred, y, caches))

            # Apply updates to LSTM model
            model.apply_updates(LR)

        last_train_loss = float(np.mean(losses))

        # Evaluate model on test set every epoch
        test_rmse, test_mape, pred_inv, tgt_inv = evaluate(model, test_inputs, test_targets, scaler)
        last_test_rmse = test_rmse
        last_test_mape = test_mape

        # Update best model if test RMSE is lower
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_epoch = epoch

        # Keep track of last predictions and targets
        last_pred_inv = pred_inv.copy()
        last_tgt_inv = tgt_inv.copy()

        # Print progress every 10 epochs or at the end of training
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(
                f"epoch {epoch}/{EPOCHS}  train_loss={last_train_loss:.6f}  "
                f"test_rmse={test_rmse:.6f}  test_mape={test_mape:.4f}%"
            )

    # Save predictions and targets
    pred_path = os.path.join(OUT_DIR, f"{args.exp_id}_predictions.npy")
    tgt_path = os.path.join(OUT_DIR, f"{args.exp_id}_targets.npy")
    np.save(pred_path, last_pred_inv)
    np.save(tgt_path, last_tgt_inv)

    # Save metadata about the model
    meta_path = os.path.join(OUT_DIR, f"{args.exp_id}_meta.npz")
    np.savez(
        meta_path,
        horizon=HORIZON,
        seq_len=SEQ_LEN,
        best_epoch=best_epoch,
        best_test_rmse=best_rmse,
    )

    # Save log of training run
    # Appends one row per run to the CSV
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
            w.writeheader() # Write header if new file
        w.writerow(row)

    print(f"Saved predictions to {pred_path} and log to {log_path}")


if __name__ == "__main__":
    main()
