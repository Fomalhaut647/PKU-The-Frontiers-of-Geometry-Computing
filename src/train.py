"""通用训练脚本。

用法:
  pixi run python -m src.train --model pointnet  --epochs 250 --num_points 1024
  pixi run python -m src.train --model pointnet2 --epochs 250 --num_points 1024 --use_normals
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.data_utils import ModelNet40Cache

ROOT = Path(__file__).resolve().parents[1]


def build_model(name: str, num_classes: int, in_channels: int):
    if name == "pointnet":
        from src.pointnet.model import PointNetCls
        return PointNetCls(num_classes=num_classes, in_channels=in_channels, feature_transform=True)
    if name == "pointnet2":
        from src.pointnet2.model import PointNet2SSGCls
        return PointNet2SSGCls(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(name)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pointnet", "pointnet2"], required=True)
    ap.add_argument("--cache_dir", default="data/modelnet40_cache")
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--use_normals", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--decay", type=float, default=1e-4)
    ap.add_argument("--lr_step", type=int, default=20)
    ap.add_argument("--lr_gamma", type=float, default=0.7)
    ap.add_argument("--feat_reg", type=float, default=1e-3, help="feature transform regularizer (PointNet)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default=None, help="logs/{model} 默认")
    ap.add_argument("--ckpt_dir", default=None, help="checkpoints/{model} 默认")
    return ap.parse_args()


def train_epoch(model, loader, optimizer, device, feat_reg: float):
    model.train()
    n_seen = 0
    n_correct = 0
    loss_sum = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits, feat_trans = model(x)
        loss = F.cross_entropy(logits, y)
        if feat_reg > 0 and feat_trans is not None:
            from src.pointnet.model import feature_transform_regularizer
            loss = loss + feat_reg * feature_transform_regularizer(feat_trans)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n_seen += x.size(0)
        n_correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / n_seen, n_correct / n_seen


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    n_seen = 0
    n_correct = 0
    per_cls_correct = torch.zeros(num_classes)
    per_cls_total = torch.zeros(num_classes)
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        pred = logits.argmax(1)
        n_seen += x.size(0)
        n_correct += (pred == y).sum().item()
        for c in range(num_classes):
            mask = y == c
            per_cls_total[c] += mask.sum().item()
            per_cls_correct[c] += (pred[mask] == c).sum().item()
    overall = n_correct / max(n_seen, 1)
    per_cls = (per_cls_correct / per_cls_total.clamp(min=1)).mean().item()
    return overall, per_cls


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = 6 if args.use_normals else 3
    train_set = ModelNet40Cache(args.cache_dir, "train", num_points=args.num_points,
                                use_normals=args.use_normals, augment=True)
    test_set = ModelNet40Cache(args.cache_dir, "test", num_points=args.num_points,
                               use_normals=args.use_normals, augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, num_classes=40, in_channels=in_channels).to(device)
    print(f"model={args.model} params={sum(p.numel() for p in model.parameters()):,}")

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    out_dir = Path(args.out_dir or f"logs/{args.model}")
    ckpt_dir = Path(args.ckpt_dir or f"checkpoints/{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "train_log.csv"
    with log_csv.open("w", newline="") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "test_oa", "test_macc", "secs"])

    feat_reg = args.feat_reg if args.model == "pointnet" else 0.0
    best_oa = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, feat_reg)
        test_oa, test_macc = evaluate(model, test_loader, device, 40)
        scheduler.step()
        secs = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        with log_csv.open("a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr_now:.6f}", f"{train_loss:.4f}",
                                    f"{train_acc:.4f}", f"{test_oa:.4f}", f"{test_macc:.4f}", f"{secs:.1f}"])
        msg = (f"[{args.model}] ep {epoch:3d}/{args.epochs} "
               f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
               f"test_oa={test_oa:.4f} test_macc={test_macc:.4f} lr={lr_now:.5f} {secs:.1f}s")
        print(msg, flush=True)
        if test_oa > best_oa:
            best_oa = test_oa
            torch.save({"model": model.state_dict(), "epoch": epoch, "test_oa": test_oa,
                        "args": vars(args)}, ckpt_dir / "best.pth")
        torch.save({"model": model.state_dict(), "epoch": epoch, "test_oa": test_oa,
                    "args": vars(args)}, ckpt_dir / "last.pth")

    print(f"best test_oa={best_oa:.4f}")


if __name__ == "__main__":
    main()
