# 第四次作业：基于 MLP 的点云表面重建

## 文件结构

```
.
├── sdf_reconstruction.py   # 主训练与重建脚本
├── render_results.py       # 可视化脚本（生成报告图表）
├── README_HW4.md           # 本文件
├── results/                # 重建结果
│   ├── <uid>/
│   │   ├── mesh_vanilla.ply   # Vanilla MLP 重建 mesh
│   │   └── mesh_fourier.ply   # Fourier MLP 重建 mesh
│   └── ...
└── report/
    ├── report.pdf          # 报告
    ├── report.tex          # 报告 LaTeX 源码
    └── figures/            # 报告图表
```

## 环境依赖

```bash
pip install torch numpy scikit-image trimesh matplotlib
```

## 数据准备

将数据放在 `data/` 目录下，结构为：

```
data/
└── <uid>/
    ├── pointcloud.npz
    ├── sdf.npz
    └── <uid>.obj
```

## 运行方式

### 训练 + 重建（推荐）

```bash
# 训练两种方法并重建所有 5 个形状（默认 2000 epoch，128^3 网格）
python sdf_reconstruction.py --mode all

# 仅训练 Vanilla MLP
python sdf_reconstruction.py --mode all --vanilla

# 仅训练 Fourier MLP
python sdf_reconstruction.py --mode all --fourier

# 单个形状
python sdf_reconstruction.py --mode all --uid 1a04e3eab45ca15dd86060f189eb133

# 更高分辨率重建（更细节，更慢）
python sdf_reconstruction.py --mode all --resolution 256
```

### 仅训练

```bash
python sdf_reconstruction.py --mode train
```

### 仅重建（需已训练好的模型）

```bash
python sdf_reconstruction.py --mode reconstruct
```

### 生成可视化图表

```bash
python render_results.py
# 输出到 report/figures/
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 2000 | 训练轮数 |
| `--batch` | 16384 | 批量大小 |
| `--lr` | 1e-3 | 学习率（余弦退火至 1e-5）|
| `--lam` | 0.1 | 梯度损失权重 λ |
| `--resolution` | 128 | Marching Cubes 网格分辨率 |

## 结果输出

- 重建 mesh 保存为 `results/<uid>/mesh_vanilla.ply` 和 `results/<uid>/mesh_fourier.ply`
- 训练日志保存为 `results/<uid>/log_vanilla.json` 和 `results/<uid>/log_fourier.json`
