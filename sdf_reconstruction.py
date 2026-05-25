#!/usr/bin/env python3
"""
SDF-based surface reconstruction using MLP.
Assignment 4 - Frontiers of Geometry Computing, PKU Spring 2026

Usage:
    python sdf_reconstruction.py --mode all         # train + reconstruct both variants
    python sdf_reconstruction.py --mode train       # train only
    python sdf_reconstruction.py --mode reconstruct # reconstruct only (needs trained models)
    python sdf_reconstruction.py --vanilla          # only vanilla MLP
    python sdf_reconstruction.py --fourier          # only Fourier-feature MLP
    python sdf_reconstruction.py --uid <UID>        # single shape
    python sdf_reconstruction.py --resolution 256   # higher resolution grid
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = "data"
RESULTS_DIR = "results"

UIDS = [
    "1a04e3eab45ca15dd86060f189eb133",
    "1a32f10b20170883663e90eaf6b4ca52",
    "1a54a2319e87bd4071d03b466c72ce41",
    "1a6ad7a24bb89733f412783097373bdc",
    "1a9b552befd6306cc8f2d5fe7449af61",
]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """8-layer MLP with skip connection at layer 4 for SDF prediction."""

    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8, skip_layer=4):
        super().__init__()
        self.skip_layer = skip_layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            elif i == skip_layer:
                in_dim = hidden_dim + input_dim
            else:
                in_dim = hidden_dim
            self.linears.append(nn.Linear(in_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = x
        for i, linear in enumerate(self.linears):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = self.relu(linear(h))
        return self.output_layer(h).squeeze(-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class FourierMLP(nn.Module):
    """MLP with random Fourier feature position encoding.

    Maps input x -> [sin(2pi B x), cos(2pi B x)] before the MLP,
    enabling the network to represent high-frequency detail.
    Reference: Tancik et al., NeurIPS 2020.
    """

    def __init__(self, num_freqs=256, sigma=10.0, hidden_dim=256,
                 num_layers=8, skip_layer=4):
        super().__init__()
        B = torch.randn(num_freqs, 3) * sigma
        self.register_buffer('B', B)          # fixed, not trained
        input_dim = 2 * num_freqs
        self.skip_layer = skip_layer
        self.linears = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            elif i == skip_layer:
                in_dim = hidden_dim + input_dim
            else:
                in_dim = hidden_dim
            self.linears.append(nn.Linear(in_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def encode(self, x):
        proj = 2 * np.pi * (x @ self.B.T)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x):
        x_enc = self.encode(x)
        h = x_enc
        for i, linear in enumerate(self.linears):
            if i == self.skip_layer:
                h = torch.cat([h, x_enc], dim=-1)
            h = self.relu(linear(h))
        return self.output_layer(h).squeeze(-1)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(uid, use_fourier=False, device="cuda",
                num_epochs=2000, batch_size=16384, lr=1e-3, lam=0.1):
    """Train a per-shape SDF network and save weights + training log."""
    variant = "fourier" if use_fourier else "vanilla"
    print(f"\n{'='*60}")
    print(f"Training [{variant}] for {uid[:12]}...")
    print(f"  epochs={num_epochs}  batch={batch_size}  lr={lr}  lambda={lam}")
    print(f"{'='*60}")

    # Load data
    sdf_data = np.load(os.path.join(DATA_DIR, uid, "sdf.npz"))
    pts_all  = torch.from_numpy(sdf_data["points"].astype(np.float32))
    sdf_all  = torch.from_numpy(sdf_data["sdf"].astype(np.float32))
    grad_all = torch.from_numpy(sdf_data["grad"].astype(np.float32))
    n = len(pts_all)

    # Build model
    if use_fourier:
        model = FourierMLP(num_freqs=256, sigma=10.0, hidden_dim=256).to(device)
    else:
        model = MLP(input_dim=3, hidden_dim=256).to(device)
    print(f"  Parameters: {model.num_params():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    log_entries = []
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        idx = torch.randperm(n)[:batch_size]
        pts  = pts_all[idx].to(device).requires_grad_(True)
        sdf  = sdf_all[idx].to(device)
        grad = grad_all[idx].to(device)

        sdf_pred = model(pts)
        loss_sdf = ((sdf_pred - sdf) ** 2).mean()

        # Compute ∇F̃ w.r.t. query points via autograd
        grad_pred = torch.autograd.grad(
            outputs=sdf_pred,
            inputs=pts,
            grad_outputs=torch.ones_like(sdf_pred),
            create_graph=True,
        )[0]
        loss_grad = ((grad_pred - grad) ** 2).mean()

        loss = loss_sdf + lam * loss_grad

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{epoch:4d}/{num_epochs}] loss={loss.item():.5f}  "
                  f"sdf={loss_sdf.item():.5f}  grad={loss_grad.item():.5f}  "
                  f"t={elapsed:.0f}s")
            log_entries.append({
                "epoch": epoch,
                "loss": round(loss.item(), 6),
                "loss_sdf": round(loss_sdf.item(), 6),
                "loss_grad": round(loss_grad.item(), 6),
            })

    train_time = time.time() - t0
    print(f"  Training done in {train_time:.1f}s")

    # Save
    save_dir = os.path.join(RESULTS_DIR, uid)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, f"model_{variant}.pt"))

    with open(os.path.join(save_dir, f"log_{variant}.json"), "w") as f:
        json.dump({
            "uid": uid,
            "variant": variant,
            "num_params": model.num_params(),
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "lambda": lam,
            "train_time_s": round(train_time, 2),
            "loss_curve": log_entries,
        }, f, indent=2)

    return model, train_time


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct_mesh(uid, model, use_fourier=False, device="cuda", resolution=128):
    """Evaluate SDF on a uniform grid and extract mesh via Marching Cubes."""
    from skimage.measure import marching_cubes
    import trimesh

    variant = "fourier" if use_fourier else "vanilla"
    grid_range = 0.55

    print(f"  Reconstructing [{variant}] res={resolution}...", end="", flush=True)
    t0 = time.time()

    # Build grid
    coords = np.linspace(-grid_range, grid_range, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Query model in chunks
    model.eval()
    sdf_vals = []
    chunk = 200_000
    with torch.no_grad():
        for i in range(0, len(grid_pts), chunk):
            pts = torch.from_numpy(grid_pts[i:i + chunk]).to(device)
            sdf_vals.append(model(pts).cpu().numpy())
    sdf_grid = np.concatenate(sdf_vals).reshape(resolution, resolution, resolution)

    # Marching Cubes
    step = 2 * grid_range / (resolution - 1)
    verts, faces, normals, _ = marching_cubes(sdf_grid, level=0.0,
                                              spacing=(step, step, step))
    verts -= grid_range  # shift from [0, 1.1] to [-0.55, 0.55]

    recon_time = time.time() - t0
    print(f" {len(verts)} verts, {len(faces)} faces  ({recon_time:.1f}s)")

    # Save PLY
    save_dir = os.path.join(RESULTS_DIR, uid)
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, f"mesh_{variant}.ply")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(ply_path)

    # Append recon time to log
    log_path = os.path.join(save_dir, f"log_{variant}.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        log["recon_time_s"] = round(recon_time, 2)
        log["mesh_verts"]   = len(verts)
        log["mesh_faces"]   = len(faces)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    return ply_path, recon_time


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_model(uid, use_fourier, device):
    variant = "fourier" if use_fourier else "vanilla"
    if use_fourier:
        model = FourierMLP(num_freqs=256, sigma=10.0, hidden_dim=256).to(device)
    else:
        model = MLP(input_dim=3, hidden_dim=256).to(device)
    path = os.path.join(RESULTS_DIR, uid, f"model_{variant}.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SDF reconstruction with MLP")
    parser.add_argument("--mode", choices=["train", "reconstruct", "all"],
                        default="all", help="Operation mode")
    parser.add_argument("--vanilla",    action="store_true", help="Only vanilla MLP")
    parser.add_argument("--fourier",    action="store_true", help="Only Fourier MLP")
    parser.add_argument("--epochs",     type=int,   default=2000)
    parser.add_argument("--batch",      type=int,   default=16384)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lam",        type=float, default=0.1,
                        help="Gradient loss weight lambda")
    parser.add_argument("--resolution", type=int,   default=128,
                        help="Marching Cubes grid resolution")
    parser.add_argument("--uid",        type=str,   default=None,
                        help="Process a single UID only")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    uids = [args.uid] if args.uid else UIDS

    if args.fourier and not args.vanilla:
        variants = [True]
    elif args.vanilla and not args.fourier:
        variants = [False]
    else:
        variants = [False, True]   # both by default

    t_total = time.time()

    for use_fourier in variants:
        for uid in uids:
            if args.mode in ("train", "all"):
                train_model(uid, use_fourier=use_fourier, device=device,
                            num_epochs=args.epochs, batch_size=args.batch,
                            lr=args.lr, lam=args.lam)

            if args.mode in ("reconstruct", "all"):
                model = load_model(uid, use_fourier=use_fourier, device=device)
                reconstruct_mesh(uid, model, use_fourier=use_fourier,
                                 device=device, resolution=args.resolution)

    print(f"\nAll done. Total elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
