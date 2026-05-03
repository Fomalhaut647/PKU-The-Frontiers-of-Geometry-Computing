"""纯 PyTorch 实现 PointNet++ 三个核心 op：FPS、Ball Query、Group。

不依赖 CUDA 扩展，避开 cu130 ABI 兼容性风险。1024 点规模下速度可接受。
"""
import torch


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    # src (B, N, 3), dst (B, M, 3) -> (B, N, M)
    return torch.cdist(src, dst, p=2.0).pow(2)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # points (B, N, C), idx (B, ...) -> (B, ..., C)
    B = points.size(0)
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_idx = torch.arange(B, device=points.device).view(view_shape).expand_as(idx)
    return points[batch_idx, idx]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    # xyz (B, N, 3) -> idx (B, npoint)
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((B, N), 1e10, device=xyz.device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=xyz.device)
    batch_idx = torch.arange(B, dtype=torch.long, device=xyz.device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].unsqueeze(1)  # (B, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    # xyz (B, N, 3), new_xyz (B, S, 3) -> idx (B, S, nsample)
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    sqrdist = square_distance(new_xyz, xyz)  # (B, S, N)
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, S, 1)
    mask = sqrdist > radius ** 2
    group_idx[mask] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    # 把超 radius 的（=N）替换为该 query 的第一个 neighbor（典型 PointNet++ trick）
    first = group_idx[:, :, 0:1].repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor | None):
    """对 xyz 做 FPS，对每个中心做 ball query，返回分组特征。

    xyz: (B, N, 3); points: (B, N, D) or None
    return new_xyz (B, S, 3), new_points (B, S, K, 3+D)
    """
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, S, K)
    grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # 局部坐标
    if points is not None:
        grouped_points = index_points(points, idx)  # (B, S, K, D)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz: torch.Tensor, points: torch.Tensor | None):
    """全局分组：把所有点合到一个虚拟中心。"""
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C, device=xyz.device)
    grouped_xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.unsqueeze(1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points
