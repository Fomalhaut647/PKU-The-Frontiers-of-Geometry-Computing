"""通用网格工具：边界检测、边界形状映射、扭曲度量。"""
from __future__ import annotations

import numpy as np


def find_boundary_loop(faces: np.ndarray) -> list[int]:
    """在纯三角网格中找到唯一的边界环，返回顶点索引的有序列表（首尾不重复）。

    Tutte 参数化要求输入是圆盘拓扑（单个边界环）。若存在多条边界环，此函数只返回最长的一条。
    """
    from collections import defaultdict, Counter

    edge_count: Counter[tuple[int, int]] = Counter()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_count[(min(a, b), max(a, b))] += 1
    boundary_edges = [e for e, c in edge_count.items() if c == 1]
    if not boundary_edges:
        raise ValueError("网格没有边界，不能用 Tutte 参数化")

    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, cur = -1, start
        while True:
            nxt = next((v for v in adj[cur] if v != prev), None)
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)

    loops.sort(key=len, reverse=True)
    return loops[0]


def map_boundary_to_circle(loop: list[int], V: np.ndarray) -> np.ndarray:
    """按原始3D 弧长比例将边界环映射到单位圆。返回 shape (len(loop), 2)。"""
    pts = V[loop]
    n = len(loop)
    # 计算每段弧长（含回到起点的那一段）
    seg = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    total = seg.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    theta = 2 * np.pi * cum / total
    uv = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return uv


def map_boundary_to_square(loop: list[int], V: np.ndarray) -> np.ndarray:
    """按弧长比例将边界映射到单位正方形（顶点 (±1, ±1)）。返回 shape (len(loop), 2)。"""
    pts = V[loop]
    n = len(loop)
    seg = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    total = seg.sum()
    cum = np.concatenate([[0.0], np.cumsum(seg[:-1])])
    t = cum / total  # [0, 1)

    uv = np.zeros((n, 2))
    for i, ti in enumerate(t):
        # 方形周长为 4，把 t ∈ [0,1) 分到四段：[0,1/4), [1/4,1/2), [1/2,3/4), [3/4,1)
        s = ti * 4.0
        if s < 1.0:
            uv[i] = [-1 + 2 * s, -1]
        elif s < 2.0:
            uv[i] = [1, -1 + 2 * (s - 1)]
        elif s < 3.0:
            uv[i] = [1 - 2 * (s - 2), 1]
        else:
            uv[i] = [-1, 1 - 2 * (s - 3)]
    return uv


def triangle_area(pts: np.ndarray) -> np.ndarray:
    """批量计算三角形面积。pts shape (F, 3, d)（d 可为2或3）。返回 (F,)。"""
    a, b, c = pts[:, 0], pts[:, 1], pts[:, 2]
    if pts.shape[-1] == 2:
        # 2D: 叉积的 z 分量绝对值除以 2
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                            - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    # 3D
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def compute_distortion(V3d: np.ndarray, UV: np.ndarray, F: np.ndarray) -> dict:
    """计算参数化扭曲度量：
       - area_ratio: 每面的 UV面积 / 3D面积 的（归一化后）倍数
       - conformal_distortion: σ_max/σ_min - 1（0 表示完美保角）
       - flipped: 以 UV 有号面积为基准，方向与多数派相反的面为真正折叠
    """
    V3d_tri = V3d[F]  # (F, 3, 3)
    UV_tri = UV[F]    # (F, 3, 2)

    area3d = triangle_area(V3d_tri)

    # 计算 UV 有号面积
    a2, b2, c2 = UV_tri[:, 0], UV_tri[:, 1], UV_tri[:, 2]
    signed_uv = 0.5 * (
        (b2[:, 0] - a2[:, 0]) * (c2[:, 1] - a2[:, 1])
        - (b2[:, 1] - a2[:, 1]) * (c2[:, 0] - a2[:, 0])
    )
    # 多数派符号
    majority_sign = np.sign(np.sum(np.sign(signed_uv)))
    if majority_sign == 0:
        majority_sign = 1.0
    eps = 1e-12
    # 退化：UV 面积接近 0（典型场景：方形边界上3个共线顶点）
    degenerate = np.abs(signed_uv) < eps
    # 真正翻转：有号面积符号与多数派相反且非零
    flipped = (np.sign(signed_uv) != majority_sign) & (~degenerate)

    area2d = np.abs(signed_uv)
    scale = area3d.sum() / max(area2d.sum(), 1e-16)
    area_ratio = (area2d * scale) / np.maximum(area3d, 1e-16)

    # 共形扭曲：把 3D 三角形投影到局部 2D，再求 Jacobian 的奇异值。
    # 这里我们让局部 2D 的方向与 UV 的多数派方向一致，以避免整体"全部翻转"的假阳性。
    a3, b3, c3 = V3d_tri[:, 0], V3d_tri[:, 1], V3d_tri[:, 2]
    e1 = b3 - a3
    e2 = c3 - a3
    x_axis = e1 / np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-16)
    dot = np.sum(e2 * x_axis, axis=1, keepdims=True)
    e2_perp = e2 - dot * x_axis
    y_axis = e2_perp / np.maximum(np.linalg.norm(e2_perp, axis=1, keepdims=True), 1e-16)

    P = np.zeros_like(UV_tri)
    P[:, 1, 0] = np.linalg.norm(e1, axis=1)
    P[:, 2, 0] = np.sum(e2 * x_axis, axis=1)
    P[:, 2, 1] = np.sum(e2 * y_axis, axis=1)

    # 若 UV 多数派为顺时针（signed<0），对局部 2D 做一次 y 翻转对齐
    if majority_sign < 0:
        P[:, :, 1] *= -1

    M3d = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)
    M2d = np.stack([UV_tri[:, 1] - UV_tri[:, 0], UV_tri[:, 2] - UV_tri[:, 0]], axis=2)
    J = M2d @ np.linalg.inv(M3d)

    sv = np.linalg.svd(J, compute_uv=False)
    sigma_max = sv[:, 0]
    sigma_min = np.maximum(sv[:, 1], 1e-16)
    conformal = sigma_max / sigma_min - 1.0

    return {
        "area_ratio": area_ratio,
        "conformal": conformal,
        "flipped": flipped,
        "degenerate": degenerate,
        "num_flipped": int(flipped.sum()),
        "num_degenerate": int(degenerate.sum()),
        "area3d": area3d,
        "area2d": area2d * scale,
        "signed_uv": signed_uv,
    }
