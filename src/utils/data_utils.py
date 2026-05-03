"""ModelNet40 dataloader：从预处理好的 npy 缓存读 10k 点，运行时随机采 N 点 + 增广。"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40Cache(Dataset):
    def __init__(
        self,
        cache_dir: str,
        split: str,
        num_points: int = 1024,
        use_normals: bool = False,
        augment: bool = True,
    ):
        self.cache_dir = Path(cache_dir) / split
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment = augment
        csv = (Path(cache_dir) / f"{split}.csv").read_text().strip().splitlines()
        self.items: list[tuple[str, int]] = []
        for line in csv:
            name, label = line.split(",")
            self.items.append((name, int(label)))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def pc_normalize(pc: np.ndarray) -> np.ndarray:
        center = pc.mean(axis=0)
        pc = pc - center
        m = np.max(np.linalg.norm(pc, axis=1))
        return pc / (m + 1e-9)

    def __getitem__(self, idx: int):
        name, label = self.items[idx]
        data = np.load(self.cache_dir / name)  # (10000, 6)
        # 随机采 num_points 点
        choice = np.random.choice(data.shape[0], self.num_points, replace=False)
        pts = data[choice]  # (N, 6)
        xyz = pts[:, :3]
        normals = pts[:, 3:6]
        xyz = self.pc_normalize(xyz).astype(np.float32)

        if self.augment:
            # 绕 z 轴随机旋转
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0.0],
                [np.sin(theta),  np.cos(theta), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            xyz = xyz @ R.T
            normals = normals @ R.T
            # 随机缩放 + 平移
            scale = np.random.uniform(0.8, 1.25)
            shift = np.random.uniform(-0.1, 0.1, size=(1, 3)).astype(np.float32)
            xyz = xyz * scale + shift
            # jitter（仅坐标）
            xyz = xyz + np.clip(0.01 * np.random.randn(*xyz.shape), -0.05, 0.05).astype(np.float32)

        if self.use_normals:
            feat = np.concatenate([xyz, normals.astype(np.float32)], axis=1)
        else:
            feat = xyz
        return torch.from_numpy(feat).float(), int(label)
