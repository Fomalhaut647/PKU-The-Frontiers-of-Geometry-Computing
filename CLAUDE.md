# CLAUDE.md (hw3 branch)

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 当前分支：hw3 — ModelNet40 三维形状分类

第三次作业要求复现两个 3D Classification 论文在 ModelNet40 上的结果。本分支选择了 **PointNet** 与 **PointNet++ SSG** 两个经典方法。截止日期 2026-05-12。

## 源码组织

```
src/
├── pointnet/model.py        # PointNet (CVPR'17): T-Net + 共享 MLP + 全局 max-pool
├── pointnet2/
│   ├── ops.py               # 纯 PyTorch 实现 FPS / Ball Query / Group（无 CUDA C++ 扩展）
│   └── model.py             # PointNet++ SSG (NeurIPS'17): 三层 SetAbstraction + FC
├── utils/
│   ├── preprocess.py        # OFF mesh → 10000 点 + 法向量 npy 缓存（多进程，~90s）
│   └── data_utils.py        # ModelNet40Cache dataset + augmentation
├── train.py                 # 通用训练入口（`--model {pointnet,pointnet2}`）
├── test.py                  # 评估入口（含可选 12-vote）
├── compare.py               # 参数量 / 推理速度对比
└── plot_curves.py           # 训练曲线图
```

报告 `report/report.tex`（XeLaTeX，沿用 hw1 风格的 `ctex` 包）。

## 关键设计决策

- **纯 PyTorch 实现 PointNet++ 的 op**。原作者 PointNet++ 仓库需要 `setup.py build_ext` 编 4 个 .cu 文件，新版 CUDA（13.0）工具链下常因 NVCC ABI 改动失败。我用 `torch.cdist` + `topk` + 高级索引重写 FPS / Ball Query / Group，速度只比 CUDA 慢 3-5 倍但**完全无编译失败风险**。H20 算力盈余够吃这个损失。
- **数据 pipeline 分离 mesh sampling 与 epoch sampling**。离线一次性把 OFF mesh 采到 10000 点（`src/utils/preprocess.py`，多进程），训练时再从 10000 点中随机采 1024（`ModelNet40Cache.__getitem__`），大幅压缩每 epoch I/O。
- **PointNet 仅用 xyz；PointNet++ 用 xyz + 法向量**。这是原论文推荐配置，让两种方法的对比贴合论文 baseline。

## 数据准备

ModelNet40 来自 `/root/autodl-pub/ModelNet/ModelNet40.zip`（autodl 公共数据集挂载）：

```bash
unzip /root/autodl-pub/ModelNet/ModelNet40.zip -d data/
pixi run python -m src.utils.preprocess --workers 32 --num_points 10000
```

预处理产出 `data/modelnet40_cache/{train,test}/<stem>.npy`，每个 `(10000, 6)`。约 2.8 GB。`data/` 已在 `.gitignore` 中。

## 训练 / 评估

```bash
# 训练（H20: PointNet ~16 min, PointNet++ ~1.5 h）
pixi run python -m src.train --model pointnet --epochs 250 --batch_size 32
pixi run python -m src.train --model pointnet2 --use_normals --epochs 250 --batch_size 32

# 评估（best checkpoint）
pixi run python -m src.test --model pointnet  --ckpt checkpoints/pointnet/best.pth
pixi run python -m src.test --model pointnet2 --use_normals --ckpt checkpoints/pointnet2/best.pth --votes 12

# 对比 + 画图
pixi run python -m src.compare
pixi run python -m src.plot_curves
```

训练日志写到 `logs/<model>/train_log.csv`（每 epoch 一行：epoch, lr, train_loss, train_acc, test_oa, test_macc, secs）；checkpoint 写到 `checkpoints/<model>/{best,last}.pth`。

## 报告

`report/report.tex` 使用 XeLaTeX + ctex 包。**实验数字以 `\newcommand` 在文件开头集中定义**（如 `\PNoa`, `\PNNoa`），训练完成后只需改这一处即可全文同步。编译：

```bash
cd report && xelatex -interaction=nonstopmode report.tex
```

## 与全局规则的关系

- 报告作者格式遵循全局 `~/.claude/CLAUDE.md`（`\author{刘智琦 2300012860}` / `\date{\today}`）
- 全局规则要求 Pixi 管理 Python 环境
- 仓库 git author `Fomalhaut647 <fomalhaut@stu.pku.edu.cn>`（pseudonym 对应中文"刘智琦"）

## 当前依赖

主要 PyPI 依赖（通过 `pixi add --pypi` 安装）：
- `torch>=2.11`（CUDA 13.0 wheel）
- `torchvision`
- `numpy`, `h5py`, `matplotlib`, `tqdm`, `scikit-learn`, `tensorboard`
- `trimesh`（OFF 加载 + 表面采样）

Python 3.11，conda-forge channel。
