import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def per_step_rmse(predictions, actuals):
    # Calculate RMSE at each step
    err = predictions - actuals
    return np.sqrt(np.mean(err**2, axis=0))


def load_arrays(exp_id, out_dir):
    # Load predictions, targets, and metadata from files
    # Raise error if files are missing
    pred_path = os.path.join(out_dir, f"{exp_id}_predictions.npy")
    tgt_path = os.path.join(out_dir, f"{exp_id}_targets.npy")
    meta_path = os.path.join(out_dir, f"{exp_id}_meta.npz")
    if not os.path.isfile(pred_path) or not os.path.isfile(tgt_path):
        raise FileNotFoundError(f"Missing {pred_path} or {tgt_path}. Run train.py first.")
    pred = np.load(pred_path)
    tgt = np.load(tgt_path)
    return pred, tgt, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--plot_dir", default="plots")
    args = parser.parse_args()

    os.makedirs(args.plot_dir, exist_ok=True)

    pred, tgt, meta_path = load_arrays(args.exp_id, args.out_dir)

    # Raise error if predictions and targets have different shapes
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch pred {pred.shape} vs tgt {tgt.shape}")

    horizon = pred.shape[1]

    # Load metadata if it exists
    if os.path.isfile(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        if int(meta["horizon"]) != horizon:
            raise ValueError("meta horizon does not match prediction width")

    # Create array of steps from 1 to horizon
    steps = np.arange(1, horizon + 1)

    # Plot1: Actual vs Predicted Close (t+1)
    # Only plot the first time step (t+1)
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(tgt[:, 0], label="Actual (t+1)", color="tab:blue", alpha=0.85)
    ax1.plot(pred[:, 0], label="Predicted (t+1)", color="tab:orange", alpha=0.85)
    ax1.set_title(f"Test set: actual vs predicted close (t+1) - {args.exp_id}")
    ax1.set_xlabel("Test window index")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    p1 = os.path.join(args.plot_dir, f"{args.exp_id}_plot1_t1_series.png")
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # Plot2: RMSE by forecast step
    # Plot RMSE at each step from t+1 to t+horizon
    rmse_steps = per_step_rmse(pred, tgt)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(steps, rmse_steps, color="tab:green", alpha=0.85)
    ax2.set_title(f"RMSE by forecast step - {args.exp_id}")
    ax2.set_xlabel("Step (t+k)")
    ax2.set_ylabel("RMSE (same units as price)")
    ax2.set_xticks(steps)
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(args.plot_dir, f"{args.exp_id}_plot2_rmse_by_step.png")
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # Plot3: Single window multi-step forecast
    # Plot actual and predicted close for a single window
    # Show the actual and predicted close for a single window
    # Show the actual and predicted close for a single window
    sample_idx = pred.shape[0] // 2
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.plot(steps, tgt[sample_idx], marker="o", label="Actual", color="tab:blue")
    ax3.plot(steps, pred[sample_idx], marker="o", label="Predicted", color="tab:orange")
    ax3.set_title(f"Single window (index {sample_idx}): multi-step forecast - {args.exp_id}")
    ax3.set_xlabel("Horizon step")
    ax3.set_ylabel("Price")
    ax3.set_xticks(steps)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    p3 = os.path.join(args.plot_dir, f"{args.exp_id}_plot3_single_window.png")
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    print(f"Saved plots:\n  {p1}\n  {p2}\n  {p3}")


if __name__ == "__main__":
    main()
