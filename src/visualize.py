"""可视化工具：
    - plot_uv_layout: 画 2D UV 布局 + 三角网格连线
    - plot_3d_with_checker: 画 3D 网格，用 UV 采样棋盘纹理上色
    - plot_distortion_heatmap: 画面-扭曲热图
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_uv_layout(UV: np.ndarray, F: np.ndarray, title: str, out_path: str) -> None:
    """绘制 2D UV 三角网格布局。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    tri = mtri.Triangulation(UV[:, 0], UV[:, 1], F)
    ax.triplot(tri, color="#1f77b4", linewidth=0.2)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_uv_grid(uvs: dict[str, np.ndarray], F: np.ndarray, out_path: str) -> None:
    """把多个 UV 布局排到 2×2 网格中做对比。"""
    n = len(uvs)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).reshape(rows, cols)
    for idx, (name, UV) in enumerate(uvs.items()):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        tri = mtri.Triangulation(UV[:, 0], UV[:, 1], F)
        ax.triplot(tri, color="#1f77b4", linewidth=0.15)
        ax.set_aspect("equal")
        ax.set_title(name.replace("_", " + "))
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")
    fig.suptitle("Tutte parameterization — UV layout comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def sample_checker(UV: np.ndarray, freq: int = 10) -> np.ndarray:
    """根据 UV 坐标采样棋盘颜色（黑/白）。返回 (n, 3) RGB in [0, 1]。"""
    # 将 UV 拉伸到 [0,1]
    u_min, v_min = UV.min(axis=0)
    u_max, v_max = UV.max(axis=0)
    u = (UV[:, 0] - u_min) / max(u_max - u_min, 1e-16)
    v = (UV[:, 1] - v_min) / max(v_max - v_min, 1e-16)
    mask = ((np.floor(u * freq).astype(int) + np.floor(v * freq).astype(int)) % 2) == 0
    color = np.where(mask[:, None], 0.9, 0.2) * np.ones((1, 3))
    return color


def plot_3d_with_checker(
    V: np.ndarray, F: np.ndarray, UV: np.ndarray, title: str, out_path: str,
    freq: int = 20,
) -> None:
    """在 3D 网格上用 UV 采样的棋盘纹理上色，两个视角并排显示。"""
    colors = sample_checker(UV, freq=freq)
    face_colors = colors[F].mean(axis=1)

    # 简单基于法向 + 光源方向的明暗处理，提升立体感
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    normals = np.cross(b - a, c - a)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-16)
    light = np.array([0.5, 0.3, 1.0])
    light = light / np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.35, 1.0)
    face_colors = face_colors * shade[:, None]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), subplot_kw={"projection": "3d"})
    for ax, (elev, azim) in zip(axes, [(20, 45), (-60, 100)]):
        coll = Poly3DCollection(V[F], facecolors=face_colors, edgecolors="none")
        ax.add_collection3d(coll)
        mins = V.min(axis=0); maxs = V.max(axis=0)
        mid = (mins + maxs) / 2
        r = (maxs - mins).max() / 2 * 0.55
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
    fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distortion_heatmap(
    V: np.ndarray, F: np.ndarray, distortion: np.ndarray, title: str,
    out_path: str, vmin: float | None = None, vmax: float | None = None,
    cmap: str = "viridis",
) -> None:
    """把面扭曲度量以热图形式显示在 3D 网格上（两视角）。"""
    cmap_obj = plt.get_cmap(cmap)
    if vmin is None:
        vmin = float(np.percentile(distortion, 5))
    if vmax is None:
        vmax = float(np.percentile(distortion, 95))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    face_colors = cmap_obj(norm(distortion))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), subplot_kw={"projection": "3d"})
    for ax, (elev, azim) in zip(axes, [(20, 45), (-60, 100)]):
        coll = Poly3DCollection(V[F], facecolors=face_colors, edgecolors="none")
        ax.add_collection3d(coll)
        mins = V.min(axis=0); maxs = V.max(axis=0)
        mid = (mins + maxs) / 2
        r = (maxs - mins).max() / 2 * 0.55
        ax.set_xlim(mid[0] - r, mid[0] + r)
        ax.set_ylim(mid[1] - r, mid[1] + r)
        ax.set_zlim(mid[2] - r, mid[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.7, label="distortion")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
