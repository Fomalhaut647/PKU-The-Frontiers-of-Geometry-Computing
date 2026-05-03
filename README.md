# PKU-The-Frontiers-of-Geometry-Computing

北京大学 2026 春季学期《几何计算前沿》课程的作业仓库。

## 仓库约定

- `main`：只维护项目级配置（`CLAUDE.md`、`.claude/skills/autopku/`、`pixi.lock`），不放作业交付物
- 每次作业开一个 branch，作业 PDF / 源码 / 数值实验产物等都在对应 branch 内
- 查看历次作业：`git branch -a` 列出所有作业 branch，`git switch <branch>` 切换查看

## 项目级工具

`.claude/skills/autopku/` 是项目独有的 [AutoPku](https://github.com/ICUlizhi/AutoPku) skill，部署后 Claude Code 会在你提到"同步通知 / 完成作业 / 写笔记"等意图时自主触发，自动调用 `pku3b` 完成 PKU 教学网相关流程。详细工作指引见 `CLAUDE.md`。

## Python 环境

某些作业需要 Python（数值优化、画图等），用 [Pixi](https://pixi.sh) 管理：

```bash
pixi install               # 装锁定环境
pixi run python <script>   # 跑脚本
```
