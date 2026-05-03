# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 仓库性质

北京大学 2026 春季学期《几何计算前沿》（Frontiers of Geometry Computing，缩写 FGC）课程的作业仓库，**仅此一门课**。Pixi 项目名 `fgc`。主要交付物是源码 + report PDF。

## 分支模型

- **`main`：脚手架/工具分支**——承载项目级配置（本 `CLAUDE.md`、`README.md`、`pixi.toml`/`pixi.lock` 模板、`.claude/skills/autopku/`），不放作业产物
- **`hwN` 分支：每次作业一个独立 branch**——基于 main 长出，自包含一个完整的 Pixi 子项目（在 main 脚手架之上加该次作业的依赖 + 源码 + `report/` + 数据），结构由该次作业决定，不与其它分支统一
- 所有分支共享 root commit `43c7f45 (first commit)`，`git merge` / `git rebase main hwN` 都正常工作（不存在 unrelated histories）
- 新作业流程：`git switch -c hwN main`（从最新 main 长出，自动获得 autopku skill / 项目级 CLAUDE.md 等），随后 `pixi add` 装作业依赖。每个 hw 分支可以重写自己的 README.md / CLAUDE.md 描述该次作业（rebase 时用 `-Xtheirs` 自动保留 hw 版本）

## 已有作业现状

| 分支 | 主题 | 源码组织 | report 格式 |
|------|------|----------|-------------|
| `hw1` | TSDF Fusion（含 color fusion bonus） | `fusion.py` + `demo.py` + `render_results.py` 平铺在根目录 | `report/report.tex`（XeLaTeX）|
| `hw2` | Tutte 网格参数化 | `src/{main,mesh_utils,tutte_numpy,tutte_openmesh,visualize,build_report}.py` | `report/report.md` |
| `hw3` | 占位（仅 `.gitattributes` / `.gitignore` / `README.md` / `pixi.toml`） | — | — |

新作业开始时按需要从 hw1/hw2 借鉴模式，或自定。**不要强制统一**——hw 之间无依赖。

## Python 环境

每个 hw 分支带自己的 Pixi 锁定环境：

```bash
pixi install                 # 装当前 branch 的环境
pixi add <pkg>               # 加包（自动更新 pixi.toml + pixi.lock）
pixi run python <script>     # 跑脚本（自动激活环境）
```

`main` 上的 `pixi.toml` 是空模板（`[dependencies]` 为空，`pixi.lock` 中 `packages: []`）；开 hw 分支后再 `pixi add` 实际依赖（numpy、open3d、openmesh、scipy 等具体由作业决定）。

## AutoPku Skill（仅在 `main` 上存在）

`.claude/skills/autopku/` 是项目级 skill，来自上游 [ICUlizhi/AutoPku](https://github.com/ICUlizhi/AutoPku) 的 clone（已剥离 repo meta：`.git/`、`.gitignore`、`README.md`、`CHANGELOG.md`、`LICENSE`、`images/`，并把 `skill.md` 重命名为 `SKILL.md`）。Claude 在用户提到"同步通知 / 看作业 / 完成作业 / 写笔记"等意图时自主触发，通过 `pku3b` CLI 操作 PKU 教学网。

**`pku3b` 当前**未**装在这台机器上**（`which pku3b` 为空）。第一次用时按 skill 内 `sub-skills/tools/pku3b-setup.md` 的步骤安装，或参考上游 [sshwy/pku3b](https://github.com/sshwy/pku3b)（推荐 v0.12.1+）。装好后常用：

| 命令 | 用途 |
|------|------|
| `pku3b a ls` | 当前学期未完成作业 |
| `pku3b a -f ls -a` | 强刷缓存列所有作业（提交后查状态用） |
| `pku3b a download <ID> -d <dir>` | 下载附件（ID = 16 位 hex） |
| `pku3b a submit <ID> <file>` | 提交单文件；多文件先 `zip` 打包再 submit |

详细工作流见 `.claude/skills/autopku/SKILL.md`。

## 上游 AutoPku 同步流程

skill 不再走 dotfiles 同步路径，以 GitHub 上游为准。更新时：

```bash
git clone git@github.com:ICUlizhi/AutoPku.git
rm -rf .claude/skills/autopku
mv AutoPku .claude/skills/autopku
cd .claude/skills/autopku
rm -rf .git images
rm -f .gitignore CHANGELOG.md LICENSE README.md
mv skill.md SKILL.md
```

## 与全局规则的关系

- 全局 `~/.claude/CLAUDE.md` 已规定课程作业 LaTeX PDF 的作者/日期格式（`\author{刘智琦 2300012860}` / `\date{\today}`）、Pixi 偏好、commit 粒度等——这些**不在本文件复述**，避免漂移
- 仓库 git author 是 `Fomalhaut647 <fomalhaut@stu.pku.edu.cn>`（与全局规则中的"刘智琦"为同一人，pseudonym vs 中文名）
