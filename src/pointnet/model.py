"""PointNet (Qi et al., CVPR 2017). 输入 BxNx3，输出 Bx40 logits。

包含两个 T-Net（3x3 输入对齐 + 64x64 特征对齐），最大池化 + FC 分类头。
返回 (logits, feat_trans) 用于特征对齐正则化。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        iden = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(B, 1)
        return (x + iden).view(B, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.tnet_input = TNet(in_channels)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.tnet_feat = TNet(64) if feature_transform else None
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor):
        # x: (B, N, C)
        x = x.transpose(1, 2)  # (B, C, N)
        T = self.tnet_input(x)
        x = torch.bmm(T, x)
        x = F.relu(self.bn1(self.conv1(x)))
        feat_trans = None
        if self.tnet_feat is not None:
            feat_trans = self.tnet_feat(x)
            x = torch.bmm(feat_trans, x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]  # (B, 1024)
        return x, feat_trans


class PointNetCls(nn.Module):
    def __init__(self, num_classes: int = 40, in_channels: int = 3, feature_transform: bool = True):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels=in_channels, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(0.4)
        self.dp2 = nn.Dropout(0.4)

    def forward(self, x: torch.Tensor):
        feat, feat_trans = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dp2(x)
        x = self.fc3(x)
        return x, feat_trans


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    diff = torch.bmm(trans, trans.transpose(1, 2)) - I
    return torch.mean(torch.norm(diff, dim=(1, 2)))
