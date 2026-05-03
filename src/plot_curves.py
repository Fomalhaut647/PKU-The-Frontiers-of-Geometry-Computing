"""画 train_loss / test_oa 曲线。读 logs/{model}/train_log.csv，输出到 report/figures/。"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: Path):
    rows = list(csv.DictReader(path.open()))
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "train_loss": [float(r["train_loss"]) for r in rows],
        "train_acc": [float(r["train_acc"]) for r in rows],
        "test_oa": [float(r["test_oa"]) for r in rows],
        "test_macc": [float(r["test_macc"]) for r in rows],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", default=["logs/pointnet/train_log.csv", "logs/pointnet2/train_log.csv"])
    ap.add_argument("--names", nargs="+", default=["PointNet", "PointNet++ (SSG)"])
    ap.add_argument("--out_dir", default="report/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for p, n in zip(args.logs, args.names):
        if Path(p).exists():
            data.append((n, read_log(Path(p))))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in data:
        ax[0].plot(d["epoch"], d["train_loss"], label=name)
        ax[1].plot(d["epoch"], d["test_oa"], label=f"{name} (OA)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("train loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("test accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.pdf")
    plt.savefig(out_dir / "training_curves.png", dpi=160)
    print(f"saved -> {out_dir}/training_curves.{{pdf,png}}")


if __name__ == "__main__":
    main()
