# 几何计算前沿 第三次作业 — 3D 形状分类

复现 **PointNet** 与 **PointNet++ (SSG)** 两种经典点云分类方法在 ModelNet40 上的结果，并对比分析参数量、运行速度与分类精度。

## 目录结构

```
hw3/
├── src/
│   ├── pointnet/
│   │   └── model.py            # PointNet (CVPR'17) 实现：T-Net + 共享 MLP + 全局最大池化
│   ├── pointnet2/
│   │   ├── ops.py              # 纯 PyTorch 实现 FPS / Ball Query / Group（无 CUDA 扩展）
│   │   └── model.py            # PointNet++ SSG (NeurIPS'17) 实现
│   ├── utils/
│   │   ├── preprocess.py       # OFF mesh -> 10000 点采样缓存（多进程）
│   │   └── data_utils.py       # ModelNet40 dataloader + 数据增广
│   ├── train.py                # 训练入口
│   ├── test.py                 # 测试入口（含可选 multi-vote 测试）
│   ├── compare.py              # 参数量 / 推理速度对比
│   └── plot_curves.py          # 训练曲线绘制
├── checkpoints/                # 训练好的模型权重（训完会有）
├── logs/                       # 训练日志 csv
├── report/
│   ├── report.tex              # XeLaTeX 报告源
│   ├── figures/                # 训练曲线、混淆矩阵
│   └── report.pdf              # 渲染后的 PDF（提交用）
├── data/                       # gitignored；ModelNet40 原始 OFF + 预处理点云缓存
├── pixi.toml / pixi.lock       # 环境锁
└── README.md
```

## 环境

- 用 [Pixi](https://pixi.sh/) 管理依赖。`pixi install` 安装；`pixi run python ...` 自动激活环境。
- 主要依赖：`python=3.11`、`torch>=2.11`（CUDA 13.0 wheel）、`numpy`、`h5py`、`matplotlib`、`tqdm`、`scikit-learn`、`tensorboard`、`trimesh`。
- 实测平台：单卡 NVIDIA H20（97 GB 显存）+ CUDA 13.0。

```bash
pixi install
```

## 数据准备

1. 把官方 [ModelNet40](https://modelnet.cs.princeton.edu/) 的 zip 解压到 `data/ModelNet40/`，得到 40 个类别子目录，每个含 `train/` 与 `test/`，文件为 `.off` mesh。AutoDL 用户可直接 `unzip /root/autodl-pub/ModelNet/ModelNet40.zip -d data/`。
2. 离线预处理（mesh 按面积均匀采 10000 点 + 法向量，缓存为 `.npy`，约 2-3 分钟，多进程）：

```bash
pixi run python -m src.utils.preprocess --workers 32 --num_points 10000
```

输出位于 `data/modelnet40_cache/{train,test}/<stem>.npy`，每文件形状 `(10000, 6)`（xyz + nx+ny+nz）。

## 训练

PointNet（仅 xyz，250 epoch）：

```bash
pixi run python -m src.train --model pointnet --epochs 250 --batch_size 32 \
                              --lr 1e-3 --optimizer adam
```

PointNet++ SSG（xyz + 法向量，250 epoch）：

```bash
pixi run python -m src.train --model pointnet2 --use_normals --epochs 250 \
                              --batch_size 32 --lr 1e-3 --optimizer adam
```

训练日志写入 `logs/<model>/train_log.csv`；最佳与最后 checkpoint 保存为 `checkpoints/<model>/{best,last}.pth`。

## 测试

单次评估：

```bash
pixi run python -m src.test --model pointnet  --ckpt checkpoints/pointnet/best.pth
pixi run python -m src.test --model pointnet2 --use_normals --ckpt checkpoints/pointnet2/best.pth
```

加 `--votes 12` 启用多次随机采样投票评估（与 PointNet 论文一致）。

## 对比与画图

```bash
pixi run python -m src.compare                                 # 参数量 + 推理速度
pixi run python -m src.plot_curves                             # 训练曲线 PDF/PNG
```

## 模型权重下载

训练完成后 `checkpoints/{pointnet,pointnet2}/best.pth` 体积均 < 50 MB，已随提交压缩包附上。如教学网附件大小受限，请通过北大网盘下载：

- 链接：（训练完成后填入）
- 提取码：（同上）

## 复现结果

详见 `report/report.pdf`。
