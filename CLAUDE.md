# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

北京大学《几何计算前沿》课程作业仓库。不同作业存放在不同的 git branch 中。

- `hw1` 分支：TSDF Fusion 作业（项目根目录）

## 环境管理

使用 Pixi 管理 Python 环境。用命令行安装依赖，不要手动编辑 `pixi.toml`：

```bash
pixi add python                             # 先装 Python（conda-forge）
pixi add numpy scikit-image tqdm opencv     # conda-forge 包
pixi add --pypi trimesh                     # PyPI 包（conda-forge 没有时）
pixi run python <script>                    # 在环境中运行脚本
```

hw1 环境已配置完成：python、numpy、scikit-image、tqdm、opencv、tectonic（conda-forge），trimesh、matplotlib（PyPI）。

### Pixi 常用操作

```bash
pixi add <pkg>            # 添加 conda 依赖
pixi add --pypi <pkg>     # 添加 PyPI 依赖
pixi remove <pkg>         # 移除依赖
pixi install              # 按 lockfile 安装/同步环境
pixi update               # 更新 lockfile 到最新兼容版本
pixi list                 # 列出已安装包
pixi run <cmd>            # 在环境中执行命令
pixi shell                # 进入激活了环境的交互式 shell
pixi task add <name> <cmd>  # 定义任务
```

> conda 包优先于 PyPI（含预编译二进制），仅在 conda-forge 没有时才用 `--pypi`。

## Homework1: TSDF Fusion

**入口**：`demo.py`（在项目根目录运行）

```bash
pixi run python demo.py
```

**核心文件**：
- `fusion.py`：主要实现文件，包含 `TSDFVolume` 类和 `cam_to_world` 函数
- `demo.py`：驱动脚本，读取 1000 帧深度图和相机位姿，调用 `fusion.py` 中的函数
- `render_results.py`：生成报告用可视化图片
- `report/`：报告目录，含 `report.tex`、`report.pdf`、PLY 结果和可视化图片

**编译报告**：`pixi run tectonic report/report.tex`（需在 report/ 下运行或调整路径）

**数据**：`data/`，包含 1000 帧的深度图（`.depth.png`，16-bit，单位毫米）、彩色图（`.color.jpg`）、相机位姿（`.pose.txt`，4×4 矩阵）和相机内参（`camera-intrinsics.txt`）。

**任务分解**：
- Task 1：在 `cam_to_world` 中将深度图转换为世界坐标点云
- Task 2：在 `TSDFVolume.__init__` 和 `integrate` 中建立体素格点坐标并投影到像素坐标采样深度
- Task 3：在 `integrate` 中计算单帧 TSDF 值（公式：`min(1.0, (depth - z) / t)`）
- Task 4：在 `integrate` 中加权平均融合 TSDF，在 `demo.py` 中用 Marching Cubes 抽取 mesh
- 附加任务：带颜色的 TSDF Fusion（已完成）

**hw1 状态**：所有任务已完成（Task 1-4 + 附加任务），报告在 `report/report.pdf`。
