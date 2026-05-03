"""PointNet++ SSG (Qi et al., NeurIPS 2017). 三层 SetAbstraction + FC 分类头。

Architecture (与原论文一致):
  SA1: npoint=512, radius=0.2, nsample=32, mlp=[64, 64, 128]
  SA2: npoint=128, radius=0.4, nsample=64, mlp=[128, 128, 256]
  SA3: 全局 group, mlp=[256, 512, 1024]
  FC : 1024 -> 512 -> 256 -> num_classes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pointnet2.ops import sample_and_group, sample_and_group_all


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int | None, radius: float | None, nsample: int | None,
                 in_channel: int, mlp: list[int], group_all: bool):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last = in_channel
        for c in mlp:
            self.mlp_convs.append(nn.Conv2d(last, c, 1))
            self.mlp_bns.append(nn.BatchNorm2d(c))
            last = c

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None):
        # xyz (B, N, 3), points (B, N, D) or None
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_points (B, S, K, C) -> (B, C, K, S)
        new_points = new_points.permute(0, 3, 2, 1).contiguous()
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, dim=2)[0]  # (B, C, S)
        new_points = new_points.transpose(1, 2)  # (B, S, C)
        return new_xyz, new_points


class PointNet2SSGCls(nn.Module):
    def __init__(self, num_classes: int = 40, in_channels: int = 3):
        super().__init__()
        extra = in_channels - 3
        self.sa1 = SetAbstraction(512, 0.2, 32, 3 + extra, [64, 64, 128], False)
        self.sa2 = SetAbstraction(128, 0.4, 64, 3 + 128, [128, 128, 256], False)
        self.sa3 = SetAbstraction(None, None, None, 3 + 256, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor):
        # x (B, N, C). 头三维 = xyz；其余通道 = 法向量等额外特征
        xyz = x[:, :, :3]
        feats = x[:, :, 3:] if x.size(2) > 3 else None
        l1_xyz, l1_pts = self.sa1(xyz, feats)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        x = l3_pts.view(l3_pts.size(0), -1)  # (B, 1024)
        x = self.dp1(F.relu(self.bn1(self.fc1(x))))
        x = self.dp2(F.relu(self.bn2(self.fc2(x))))
        return self.fc3(x), None
