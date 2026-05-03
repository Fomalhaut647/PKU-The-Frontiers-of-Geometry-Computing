"""对比 PointNet vs PointNet++ 的参数量、FLOPs、推理速度（batch=8）。

用法: pixi run python -m src.compare
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.train import build_model


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_speed(model, x: torch.Tensor, n_warmup: int = 10, n_iter: int = 50) -> float:
    device = x.device
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_iter):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_points", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default="logs/compare.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    configs = [
        ("pointnet", 3, False),
        ("pointnet2", 6, True),
    ]
    for name, in_channels, use_normals in configs:
        model = build_model(name, num_classes=40, in_channels=in_channels).to(device)
        x = torch.randn(args.batch_size, args.num_points, in_channels, device=device)
        params = count_params(model)
        sec_per_batch = measure_speed(model, x)
        sec_per_sample = sec_per_batch / args.batch_size
        rows.append({
            "model": name,
            "in_channels": in_channels,
            "use_normals": use_normals,
            "params_M": params / 1e6,
            "params_raw": params,
            "sec_per_batch": sec_per_batch,
            "sec_per_sample": sec_per_sample,
            "samples_per_sec": 1.0 / sec_per_sample,
        })
        print(json.dumps(rows[-1], indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
