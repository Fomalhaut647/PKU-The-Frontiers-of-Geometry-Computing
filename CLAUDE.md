# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

北京大学《几何计算前沿》课程作业仓库。不同作业存放在不同的 git branch 中。

- `hw1` 分支：TSDF Fusion 作业（项目根目录）

## 环境管理

使用 Pixi 管理 Python 环境。在 `pixi.toml` 中添加依赖后运行：

```bash
pixi install
pixi run python <script>
```

目前 `pixi.toml` 的 `[dependencies]` 为空，需要根据作业需要添加（如 `numpy`、`opencv`、`trimesh`、`scikit-image`、`tqdm`）。

## Homework1: TSDF Fusion

**入口**：`demo.py`（在项目根目录运行）

```bash
pixi run python demo.py
```

**核心文件**：
- `fusion.py`：主要实现文件，包含 `TSDFVolume` 类和 `cam_to_world` 函数
- `demo.py`：驱动脚本，读取 1000 帧深度图和相机位姿，调用 `fusion.py` 中的函数

**数据**：`data/`，包含 1000 帧的深度图（`.depth.png`，16-bit，单位毫米）、彩色图（`.color.jpg`）、相机位姿（`.pose.txt`，4×4 矩阵）和相机内参（`camera-intrinsics.txt`）。

**任务分解**：
- Task 1：在 `cam_to_world` 中将深度图转换为世界坐标点云
- Task 2：在 `TSDFVolume.__init__` 和 `integrate` 中建立体素格点坐标并投影到像素坐标采样深度
- Task 3：在 `integrate` 中计算单帧 TSDF 值（公式：`min(1.0, (depth - z) / t)`）
- Task 4：在 `integrate` 中加权平均融合 TSDF，在 `demo.py` 中用 Marching Cubes 抽取 mesh
