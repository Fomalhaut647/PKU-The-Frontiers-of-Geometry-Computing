"""加载训练好的 checkpoint，在 ModelNet40 test 上跑一次评估并报告 OA / mAcc。

用法: pixi run python -m src.test --model pointnet --ckpt checkpoints/pointnet/best.pth
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.utils.data_utils import ModelNet40Cache
from src.train import build_model, evaluate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pointnet", "pointnet2"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache_dir", default="data/modelnet40_cache")
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--use_normals", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--votes", type=int, default=1, help="多次随机采样投票，原 PointNet 用 12 次")
    return ap.parse_args()


@torch.no_grad()
def vote(model, loader, device, num_votes: int, num_classes: int):
    """Multi-vote evaluation per PointNet paper: average softmax over `num_votes`
    re-samplings of the test set, then argmax once."""
    model.eval()
    logits_sum = None
    labels_all = None
    for v in range(num_votes):
        batch_logits, batch_labels = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits, _ = model(x)
            batch_logits.append(torch.softmax(logits, dim=1).cpu())
            batch_labels.append(y)
        logits_v = torch.cat(batch_logits, 0)
        labels_v = torch.cat(batch_labels, 0)
        if logits_sum is None:
            logits_sum = logits_v
            labels_all = labels_v
        else:
            logits_sum = logits_sum + logits_v
    pred = logits_sum.argmax(1)
    n_seen = pred.size(0)
    n_correct = (pred == labels_all).sum().item()
    per_cls_correct = torch.zeros(num_classes)
    per_cls_total = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = labels_all == c
        per_cls_total[c] = mask.sum().item()
        per_cls_correct[c] = (pred[mask] == c).sum().item()
    return n_correct / max(n_seen, 1), (per_cls_correct / per_cls_total.clamp(min=1)).mean().item()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 6 if args.use_normals else 3

    test_set = ModelNet40Cache(args.cache_dir, "test", num_points=args.num_points,
                               use_normals=args.use_normals,
                               augment=(args.votes > 1))  # vote 时开 augment 让 random sampling 多样化
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, num_classes=40, in_channels=in_channels).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded {args.ckpt} (epoch {state.get('epoch')}, train-time test_oa={state.get('test_oa', 'NA')})")

    if args.votes <= 1:
        oa, macc = evaluate(model, test_loader, device, 40)
    else:
        oa, macc = vote(model, test_loader, device, args.votes, 40)
    out = {"model": args.model, "ckpt": args.ckpt, "votes": args.votes, "oa": oa, "macc": macc,
           "num_points": args.num_points, "use_normals": args.use_normals}
    print(json.dumps(out, indent=2))
    Path(args.ckpt).with_suffix(".result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
