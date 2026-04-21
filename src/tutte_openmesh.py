"""Tutte 参数化 —— OpenMesh / 半边结构风格实现。

本实现刻意多使用半边遍历来展示半边结构的表达力：
    - is_boundary(halfedge) 判断边界
    - next_halfedge_handle 沿着边界环前进
    - 对每个面遍历其三个半边来累加 cotangent 权重
"""
from __future__ import annotations

import numpy as np
import openmesh as om
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def load_mesh(path: str) -> om.TriMesh:
    return om.read_trimesh(path)


def vertices_as_array(mesh: om.TriMesh) -> np.ndarray:
    """提取顶点坐标为 numpy 数组。"""
    return np.array([mesh.point(v) for v in mesh.vertices()])


def faces_as_array(mesh: om.TriMesh) -> np.ndarray:
    """提取面为顶点索引数组。"""
    faces = []
    for f in mesh.faces():
        vs = [v.idx() for v in mesh.fv(f)]
        faces.append(vs)
    return np.array(faces, dtype=np.int64)


def find_boundary_loop_halfedge(mesh: om.TriMesh) -> list[int]:
    """用半边结构找到边界环：
       1. 找到任意一个边界半边；
       2. 沿着 next_halfedge_handle 遍历整个边界环（边界半边的 next 仍是边界半边）。
    """
    start_he = None
    for he in mesh.halfedges():
        if mesh.is_boundary(he):
            start_he = he
            break
    if start_he is None:
        raise ValueError("网格没有边界")

    loop: list[int] = []
    he = start_he
    while True:
        v_from = mesh.from_vertex_handle(he).idx()
        loop.append(v_from)
        he = mesh.next_halfedge_handle(he)
        if he == start_he:
            break
    return loop


def build_uniform_laplacian_hm(mesh: om.TriMesh) -> sp.csr_matrix:
    """用半边遍历构造均匀图拉普拉斯。

    每个顶点 v：
        (L·x)_v = deg(v) * x_v - Σ_{u∈N(v)} x_u
    通过遍历 v 的出半边得到邻居。
    """
    n = mesh.n_vertices()
    rows, cols, data = [], [], []
    for v in mesh.vertices():
        i = v.idx()
        deg = 0
        for he in mesh.voh(v):
            j = mesh.to_vertex_handle(he).idx()
            rows.append(i); cols.append(j); data.append(-1.0)
            deg += 1
        rows.append(i); cols.append(i); data.append(float(deg))
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def build_cotangent_laplacian_hm(mesh: om.TriMesh) -> sp.csr_matrix:
    """用半边遍历构造 cotangent 拉普拉斯。

    对每个（非边界）面 (i, j, k)，遍历其三个半边，对对边累加 0.5*cot(对角)。
    """
    n = mesh.n_vertices()
    rows, cols, data = [], [], []

    for f in mesh.faces():
        vs = [v.idx() for v in mesh.fv(f)]
        pts = [mesh.point(mesh.vertex_handle(i)) for i in vs]
        i, j, k = vs
        vi, vj, vk = pts

        # 对顶点 i 的角 α_i，其对边是 (j, k)
        for a, b, c, va, vb, vc in [
            (i, j, k, vi, vj, vk),
            (j, k, i, vj, vk, vi),
            (k, i, j, vk, vi, vj),
        ]:
            e1 = vb - va
            e2 = vc - va
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-16:
                continue
            cot_a = float(np.dot(e1, e2) / cross_norm)
            w = 0.5 * cot_a
            # b, c 是对边顶点
            rows.append(b); cols.append(c); data.append(-w)
            rows.append(c); cols.append(b); data.append(-w)
            rows.append(b); cols.append(b); data.append(w)
            rows.append(c); cols.append(c); data.append(w)

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def tutte_parameterize_om(
    mesh: om.TriMesh,
    boundary_loop: list[int],
    boundary_uv: np.ndarray,
    weight: str = "uniform",
) -> np.ndarray:
    """OpenMesh 风格的 Tutte 参数化求解。"""
    n = mesh.n_vertices()
    if weight == "uniform":
        L = build_uniform_laplacian_hm(mesh)
    elif weight == "cotangent":
        L = build_cotangent_laplacian_hm(mesh)
    else:
        raise ValueError(f"unknown weight: {weight}")

    boundary_set = set(boundary_loop)
    interior = np.array([i for i in range(n) if i not in boundary_set])
    boundary = np.array(boundary_loop)

    L_II = L[interior][:, interior]
    L_IB = L[interior][:, boundary]

    UV = np.zeros((n, 2))
    UV[boundary] = boundary_uv
    rhs = -L_IB @ boundary_uv
    x = spla.spsolve(L_II.tocsc(), sp.csc_matrix(rhs))
    if sp.issparse(x):
        x = x.toarray()
    UV[interior] = np.asarray(x)
    return UV
