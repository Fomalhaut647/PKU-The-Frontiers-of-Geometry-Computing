"""主运行脚本：跑完 Tutte 参数化的所有组合并生成可视化。"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mesh_utils import (
    find_boundary_loop,
    map_boundary_to_circle,
    map_boundary_to_square,
    compute_distortion,
)
from tutte_numpy import tutte_parameterize
from tutte_openmesh import (
    load_mesh,
    vertices_as_array,
    faces_as_array,
    find_boundary_loop_halfedge,
    tutte_parameterize_om,
)
from visualize import (
    plot_uv_layout,
    plot_3d_with_checker,
    plot_distortion_heatmap,
    plot_uv_grid,
)


def write_obj_with_uv(path: str, V: np.ndarray, F: np.ndarray, UV: np.ndarray) -> None:
    """把 UV 坐标作为 `vt` 写入 obj，便于外部软件加载纹理。"""
    lines = ["# Tutte parameterization result"]
    for v in V:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for uv in UV:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    for f in F:
        a, b, c = f[0] + 1, f[1] + 1, f[2] + 1
        lines.append(f"f {a}/{a} {b}/{b} {c}/{c}")
    Path(path).write_text("\n".join(lines) + "\n")


def main() -> None:
    mesh_path = str(ROOT / "parameterization.obj")
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ---------- 加载 ----------
    print("[1/5] 加载网格...")
    tm = trimesh.load(mesh_path, process=False)
    V = np.asarray(tm.vertices, dtype=np.float64)
    F = np.asarray(tm.faces, dtype=np.int64)
    n = len(V)
    print(f"      顶点 {n}, 面 {len(F)}")

    # ---------- 边界 ----------
    print("[2/5] 找边界环（numpy + OpenMesh 两种方式）...")
    loop_np = find_boundary_loop(F)

    mesh_om = load_mesh(mesh_path)
    loop_om = find_boundary_loop_halfedge(mesh_om)

    print(f"      numpy loop len:    {len(loop_np)}")
    print(f"      openmesh loop len: {len(loop_om)}")
    # 两种方式找出的边界应该是同一个环（可能起点/方向不同）
    assert set(loop_np) == set(loop_om), "边界集合应相同"

    # 以 numpy 版本为准
    loop = loop_np

    uv_circle = map_boundary_to_circle(loop, V)
    uv_square = map_boundary_to_square(loop, V)

    # ---------- 求解 ----------
    print("[3/5] 求解 Tutte 线性系统（4 种组合 + OpenMesh 验证）...")
    combos = [
        ("uniform", "circle", uv_circle),
        ("uniform", "square", uv_square),
        ("cotangent", "circle", uv_circle),
        ("cotangent", "square", uv_square),
    ]

    results: dict[str, dict] = {}
    for weight, shape, bd_uv in combos:
        key = f"{weight}_{shape}"
        t0 = time.time()
        UV = tutte_parameterize(V, F, loop, bd_uv, weight=weight)
        dt = time.time() - t0
        dist = compute_distortion(V, UV, F)
        valid = ~dist["degenerate"]
        conf_valid = dist["conformal"][valid]
        ar_valid = dist["area_ratio"][valid]
        results[key] = {
            "UV": UV,
            "time": dt,
            "num_flipped": dist["num_flipped"],
            "num_degenerate": dist["num_degenerate"],
            "conformal_mean": float(np.mean(conf_valid)) if conf_valid.size else float("nan"),
            "conformal_p95": float(np.percentile(conf_valid, 95)) if conf_valid.size else float("nan"),
            "area_ratio_p05": float(np.percentile(ar_valid, 5)) if ar_valid.size else float("nan"),
            "area_ratio_p95": float(np.percentile(ar_valid, 95)) if ar_valid.size else float("nan"),
            "distortion": dist,
        }
        print(
            f"      {key:22s}  t={dt:.3f}s  "
            f"flipped={dist['num_flipped']:4d}  "
            f"degen={dist['num_degenerate']:4d}  "
            f"conformal(mean|p95)={results[key]['conformal_mean']:.3f}|"
            f"{results[key]['conformal_p95']:.3f}"
        )

    # 验证：用 OpenMesh 风格算同样的 uniform+circle，检查数值一致
    print("      验证 OpenMesh 实现 (uniform+circle) 与 numpy 版本一致性...")
    UV_om = tutte_parameterize_om(mesh_om, loop, uv_circle, weight="uniform")
    diff = np.linalg.norm(UV_om - results["uniform_circle"]["UV"])
    print(f"      numpy vs openmesh diff: {diff:.2e}")
    assert diff < 1e-6, f"两种实现数值不一致: {diff}"

    # cotangent 版本也验证
    UV_om_cot = tutte_parameterize_om(mesh_om, loop, uv_circle, weight="cotangent")
    diff_cot = np.linalg.norm(UV_om_cot - results["cotangent_circle"]["UV"])
    print(f"      cotangent numpy vs openmesh diff: {diff_cot:.2e}")
    assert diff_cot < 1e-6, f"cotangent 两种实现数值不一致: {diff_cot}"

    # ---------- 可视化 ----------
    print("[4/5] 生成可视化...")
    # 选择最"好"的一个（cotangent_circle）同时做原始 3D 棋盘贴图
    for key, res in results.items():
        UV = res["UV"]
        plot_uv_layout(
            UV, F, f"UV layout: {key.replace('_', ' + ')}",
            str(results_dir / f"uv_{key}.png"),
        )
        plot_3d_with_checker(
            V, F, UV, f"3D checker: {key.replace('_', ' + ')}",
            str(results_dir / f"checker_{key}.png"),
        )
        plot_distortion_heatmap(
            V, F, res["distortion"]["conformal"],
            f"Conformal distortion: {key.replace('_', ' + ')}",
            str(results_dir / f"distortion_{key}.png"),
            vmin=0.0, vmax=float(np.percentile(res["distortion"]["conformal"], 95)),
        )

    # 生成 2×2 对比图
    plot_uv_grid(
        {k: r["UV"] for k, r in results.items()},
        F,
        str(results_dir / "uv_comparison_grid.png"),
    )

    # ---------- 输出 UV-obj ----------
    best_key = "cotangent_circle"
    write_obj_with_uv(
        str(results_dir / "parameterization_uv.obj"), V, F, results[best_key]["UV"]
    )
    print(f"      带UV的obj → results/parameterization_uv.obj ({best_key})")

    # ---------- 保存摘要 ----------
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk not in ("UV", "distortion")}
        for k, v in results.items()
    }
    summary["mesh"] = {
        "n_vertices": int(n),
        "n_faces": int(len(F)),
        "n_boundary": int(len(loop)),
        "euler_number": int(n - len(tm.edges_unique) + len(F)),
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[5/5] 完成。结果在 results/")


if __name__ == "__main__":
    main()
