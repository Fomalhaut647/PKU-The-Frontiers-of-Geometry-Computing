"""Tutte 参数化 —— numpy / scipy 风格实现。

核心思路：
    1. 找边界环，把边界映射到凸域（单位圆/正方形）。
    2. 对内部顶点建立离散 Laplace 方程 L·x = 0（u, v 分量各自独立求解）。
    3. 把边界 UV 作为 Dirichlet 边界条件移到右端项，解稀疏线性系统。

两种权重：
    - uniform: 每个邻居权重相等（经典 Tutte, 1963）。
    - cotangent: w_{ij} = 0.5*(cot α + cot β)，α/β 为边 (i,j) 两侧对角角度。
                 这对应离散调和映射（Pinkall & Polthier 1993），保角性更好。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_adjacency(faces: np.ndarray, n_verts: int) -> list[set[int]]:
    """返回每个顶点的 1-ring 邻居集合。"""
    adj: list[set[int]] = [set() for _ in range(n_verts)]
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            adj[a].add(b)
            adj[b].add(a)
    return adj


def build_uniform_laplacian(faces: np.ndarray, n_verts: int) -> sp.csr_matrix:
    """返回对称的均匀图拉普拉斯 L，满足 (L·x)_i = Σ_{j∈N(i)} (x_i - x_j)。

    L = D - A，D 是度数对角阵，A 是邻接矩阵。
    """
    adj = build_adjacency(faces, n_verts)
    rows, cols, data = [], [], []
    for i, nbrs in enumerate(adj):
        rows.append(i)
        cols.append(i)
        data.append(float(len(nbrs)))
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            data.append(-1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_verts, n_verts))


def build_cotangent_laplacian(V: np.ndarray, faces: np.ndarray) -> sp.csr_matrix:
    """构造 cotangent 拉普拉斯。

    对每个三角形 (i, j, k)，对边 (j, k) 的对角 i 的角为 α_i：
        cot(α_i) = ((v_j - v_i) · (v_k - v_i)) / |(v_j - v_i) × (v_k - v_i)|
    然后把 0.5 * cot(α_i) 加到边 (j, k) 的权重上（每条边有两个对角贡献，对称）。
    """
    n = V.shape[0]
    rows, cols, data = [], [], []

    tri = V[faces]  # (F, 3, 3)
    for li in range(3):
        i_idx = faces[:, li]
        j_idx = faces[:, (li + 1) % 3]
        k_idx = faces[:, (li + 2) % 3]
        vi = V[i_idx]
        vj = V[j_idx]
        vk = V[k_idx]
        e1 = vj - vi
        e2 = vk - vi
        # cot(α_i) at vertex i, opposite edge (j,k)
        cot = np.sum(e1 * e2, axis=1) / np.maximum(
            np.linalg.norm(np.cross(e1, e2), axis=1), 1e-16
        )
        w = 0.5 * cot
        # 为边 (j, k) 及 (k, j) 各贡献 -w，对角线 j 和 k 各贡献 +w
        rows.extend(j_idx.tolist()); cols.extend(k_idx.tolist()); data.extend((-w).tolist())
        rows.extend(k_idx.tolist()); cols.extend(j_idx.tolist()); data.extend((-w).tolist())
        rows.extend(j_idx.tolist()); cols.extend(j_idx.tolist()); data.extend(w.tolist())
        rows.extend(k_idx.tolist()); cols.extend(k_idx.tolist()); data.extend(w.tolist())

    L = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    return L


def tutte_parameterize(
    V: np.ndarray,
    F: np.ndarray,
    boundary_loop: list[int],
    boundary_uv: np.ndarray,
    weight: str = "uniform",
) -> np.ndarray:
    """计算 Tutte 参数化。

    Args:
        V: (n, 3) 顶点坐标
        F: (m, 3) 三角面索引
        boundary_loop: 边界顶点索引序列
        boundary_uv: (len(boundary_loop), 2) 边界 UV 坐标
        weight: "uniform" 或 "cotangent"

    Returns:
        UV: (n, 2) 全部顶点的 UV 坐标
    """
    n = V.shape[0]
    boundary_set = set(boundary_loop)
    interior = np.array([i for i in range(n) if i not in boundary_set])
    boundary = np.array(boundary_loop)

    if weight == "uniform":
        L = build_uniform_laplacian(F, n)
    elif weight == "cotangent":
        L = build_cotangent_laplacian(V, F)
    else:
        raise ValueError(f"unknown weight: {weight}")

    # Solve L_II · x_I = - L_IB · x_B
    L_II = L[interior][:, interior]
    L_IB = L[interior][:, boundary]

    UV = np.zeros((n, 2))
    UV[boundary] = boundary_uv
    rhs = -L_IB @ boundary_uv  # (|I|, 2)

    # 两个线性系统同时解：scipy spsolve 支持多右端项
    x = spla.spsolve(L_II.tocsc(), sp.csc_matrix(rhs))
    if sp.issparse(x):
        x = x.toarray()
    UV[interior] = np.asarray(x)
    return UV
