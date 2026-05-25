#!/usr/bin/env python3
"""
Render comparison figures for the report.
Produces PNG images showing GT, vanilla-MLP, and Fourier-MLP reconstructions.

Usage:
    python render_results.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import trimesh

DATA_DIR    = "data"
RESULTS_DIR = "results"
FIGURE_DIR  = "report/figures"

UIDS = [
    "1a04e3eab45ca15dd86060f189eb133",
    "1a32f10b20170883663e90eaf6b4ca52",
    "1a54a2319e87bd4071d03b466c72ce41",
    "1a6ad7a24bb89733f412783097373bdc",
    "1a9b552befd6306cc8f2d5fe7449af61",
]

UID_LABELS = {
    "1a04e3eab45ca15dd86060f189eb133": "Shape 1",
    "1a32f10b20170883663e90eaf6b4ca52": "Shape 2",
    "1a54a2319e87bd4071d03b466c72ce41": "Shape 3",
    "1a6ad7a24bb89733f412783097373bdc": "Shape 4",
    "1a9b552befd6306cc8f2d5fe7449af61": "Shape 5",
}


def sample_surface(mesh, n=30000):
    """Sample points on mesh surface, return (points, normals)."""
    pts, face_idx = trimesh.sample.sample_surface(mesh, n)
    norms = mesh.face_normals[face_idx]
    return pts, norms


def plot_points(ax, pts, norms, title, color=None, s=0.3, azim=30, elev=20):
    """Scatter-plot surface points colored by normal z-component."""
    if color is None:
        c = (norms[:, 2] + 1) / 2  # map [-1,1] -> [0,1]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=c, cmap="coolwarm", s=s, alpha=0.6, linewidths=0)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=color, s=s, alpha=0.6, linewidths=0)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(-0.55, 0.55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1, 1, 1])


def render_comparison(uid, label):
    """Render GT vs vanilla vs Fourier for one shape, two views."""
    gt_path  = os.path.join(DATA_DIR, uid, f"{uid}.obj")
    v_path   = os.path.join(RESULTS_DIR, uid, "mesh_vanilla.ply")
    f_path   = os.path.join(RESULTS_DIR, uid, "mesh_fourier.ply")

    gt_mesh = trimesh.load(gt_path, force="mesh")
    v_mesh  = trimesh.load(v_path,  force="mesh")
    f_mesh  = trimesh.load(f_path,  force="mesh")

    gt_pts, gt_norms = sample_surface(gt_mesh, 30000)
    v_pts,  v_norms  = sample_surface(v_mesh,  20000)
    f_pts,  f_norms  = sample_surface(f_mesh,  20000)

    views = [(20, 30), (20, 120)]   # (elev, azim)
    fig = plt.figure(figsize=(12, 4 * len(views)))

    for row, (elev, azim) in enumerate(views):
        ax_gt = fig.add_subplot(len(views), 3, row * 3 + 1, projection="3d")
        ax_v  = fig.add_subplot(len(views), 3, row * 3 + 2, projection="3d")
        ax_f  = fig.add_subplot(len(views), 3, row * 3 + 3, projection="3d")

        plot_points(ax_gt, gt_pts, gt_norms, "Ground Truth",  azim=azim, elev=elev)
        plot_points(ax_v,  v_pts,  v_norms,  "Vanilla MLP",   azim=azim, elev=elev)
        plot_points(ax_f,  f_pts,  f_norms,  "Fourier MLP",   azim=azim, elev=elev)

    fig.suptitle(f"{label} ({uid[:12]}...)", fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, f"compare_{uid[:12]}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def render_all_overview():
    """One overview figure: 5 shapes × 3 methods, single view."""
    views = [(20, 30)]
    cols = 3
    rows = len(UIDS)
    fig = plt.figure(figsize=(12, 4.2 * rows))

    for r, uid in enumerate(UIDS):
        label = UID_LABELS[uid]
        gt_path = os.path.join(DATA_DIR, uid, f"{uid}.obj")
        v_path  = os.path.join(RESULTS_DIR, uid, "mesh_vanilla.ply")
        f_path  = os.path.join(RESULTS_DIR, uid, "mesh_fourier.ply")

        gt_mesh = trimesh.load(gt_path, force="mesh")
        v_mesh  = trimesh.load(v_path,  force="mesh")
        f_mesh  = trimesh.load(f_path,  force="mesh")

        gt_pts, gt_norms = sample_surface(gt_mesh, 20000)
        v_pts,  v_norms  = sample_surface(v_mesh,  15000)
        f_pts,  f_norms  = sample_surface(f_mesh,  15000)

        elev, azim = views[0]
        for c, (pts, norms, title) in enumerate([
            (gt_pts, gt_norms, f"{label} - GT"),
            (v_pts,  v_norms,  f"{label} - Vanilla"),
            (f_pts,  f_norms,  f"{label} - Fourier"),
        ]):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1, projection="3d")
            plot_points(ax, pts, norms, title, azim=azim, elev=elev, s=0.2)

    plt.tight_layout()
    out = os.path.join(FIGURE_DIR, "overview_all.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("Rendering per-shape comparisons...")
    for uid, label in UID_LABELS.items():
        render_comparison(uid, label)
    print("Rendering overview figure...")
    render_all_overview()
    print("Done.")


if __name__ == "__main__":
    main()
