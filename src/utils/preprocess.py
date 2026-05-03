"""离线把 ModelNet40 OFF mesh -> 采样点云缓存。一次预处理后 dataloader 直接读 .npy。"""
import argparse
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def load_off_robust(path: Path) -> trimesh.Trimesh:
    text = path.read_text()
    # ModelNet40 known bug: first line is "OFFnnn fff eee" instead of "OFF\nnnn fff eee".
    if not text.startswith("OFF\n") and text.startswith("OFF"):
        text = "OFF\n" + text[3:]
    return trimesh.load(io.StringIO(text), file_type="off", force="mesh")


def sample_off(path: Path, num_points: int) -> np.ndarray:
    mesh = load_off_robust(path)
    if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
        raise ValueError(f"empty mesh: {path}")
    pts, face_idx = mesh.sample(num_points, return_index=True)
    normals = mesh.face_normals[face_idx]
    return np.concatenate([pts.astype(np.float32), normals.astype(np.float32)], axis=1)


def _process_one(args_tuple):
    f_str, label, out_path_str, num_points = args_tuple
    try:
        pts = sample_off(Path(f_str), num_points)
    except Exception as e:
        return (f_str, label, False, str(e))
    np.save(out_path_str, pts)
    return (Path(f_str).stem, label, True, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ModelNet40")
    ap.add_argument("--dst", default="data/modelnet40_cache")
    ap.add_argument("--num_points", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = ap.parse_args()

    src = ROOT / args.src
    dst = ROOT / args.dst
    dst.mkdir(parents=True, exist_ok=True)

    classes = sorted([p.name for p in src.iterdir() if p.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    (dst / "shape_names.txt").write_text("\n".join(classes) + "\n")

    for split in ("train", "test"):
        out_dir = dst / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob(f"*/{split}/*.off"))
        tasks = []
        for f in files:
            cls = f.parent.parent.name
            tasks.append((str(f), cls_to_idx[cls], str(out_dir / f"{f.stem}.npy"), args.num_points))
        rows = []
        bad = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for stem_or_path, label, ok, err in tqdm(
                ex.map(_process_one, tasks, chunksize=8),
                total=len(tasks), desc=split,
            ):
                if ok:
                    rows.append(f"{stem_or_path}.npy,{label}\n")
                else:
                    bad += 1
                    print(f"WARN {stem_or_path}: {err}", file=sys.stderr)
        (dst / f"{split}.csv").write_text("".join(rows))
        print(f"{split}: {len(rows)} good, {bad} bad")


if __name__ == "__main__":
    main()
