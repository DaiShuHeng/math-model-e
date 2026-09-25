# 2026 数学建模 E 题：修订与优化训练

私有队内工作仓库。当前已修复连续缺失训练协议、填充掩码、模型加载与打包漏件，并加入成对监督、一致性约束及五组验证集对照实验。**优化模型还未完成正式多轮训练，不存在已证实的准确率提升。** 历史结果和历史提交目录不能当作新协议结果。

## 在服务器开始训练

先激活服务器上已经配置好 CUDA 的 PyTorch 环境。在仓库根目录执行：

```bash
git pull --ff-only origin main
python -m pip install -r requirements-training.txt
# 服务器已具备完整数据和 Tiny 模型时，无需重新下载。
python scripts/fetch_training_assets.py --asset tiny cache
bash scripts/train_server.sh --out weights/optimization_s2026 --seed 2026 --epochs 30
```

下载脚本使用已有 Git 登录或 `GITHUB_TOKEN` 环境变量；不要把令牌写进脚本。已有文件默认拒绝覆盖。只缺对齐数据时用 `--asset aligned`；完整原始素材或非对齐版本可使用已修复数据资产的 `download_all.sh`，但它是全量覆盖式下载，应先检查已有文件。原始数据及大权重均不进入 Git。

脚本默认使用 `E题/E题数据` 与 `models/bert-tiny`；服务器数据保留在其他目录时：

```bash
export MATH_E_DATA=/home/daishuheng/math_competition/E题/E题数据
export MATH_E_BERT=/home/daishuheng/math_competition/models/bert-tiny
bash scripts/train_server.sh --out weights/optimization_s2026 --seed 2026 --epochs 30
```

输出路径相对 `solution/`。五组试验串行运行，每组独立保存模型与日志；对照表在 `solution/weights/optimization_s2026/validation_comparison.json`。模型按验证集选取，脚本不运行 test 或专项集推断。服务器断开会话前请使用你习惯的持久终端会话。

只查看计划：

```bash
cd solution
python -m src.optimize_valid --out weights/optimization_s2026 --bert-dir ../models/bert-tiny --plan
python -m tests.run_tests
python -m src.preflight --smoke
```

`preflight --smoke` 会在临时模型上用 4 个真实 train 样本做一次梯度更新，不保存权重，不是准确率评估。当前本机通过 10 项测试、合成端到端训练和真实 train 梯度检查；GPU 路径需服务器执行验证。

## 当前文件状态

2026-09-25 本机核验：完整 aligned_50.pkl（993,842,861 字节）、Tiny 配置/权重/词表、Q3 归一化缓存、100 条 Q1 特征、30 条附件 3 对齐样本、20 条附件 4 对齐样本均已具备，**修订 Q2 训练输入齐全**。预训练 Tiny 模型的完整 Q2 参数约 18.84 MB（FP32），正式打包仍需计入全部其他文件。

本机仍未安装 BERT-base 预训练目录和非对齐数据；它们不阻塞当前 Tiny + aligned 路线。重新生成 Q1 原始特征还需原始音视频工具链及 CTC 模型；已有 Q1 特征不等于该链已全部重跑。

## 文档

- [模型优化方案与实验说明](模型优化方案与实验说明.md)
- [服务器输入核验](服务器输入核验.json)
- [审查与修订报告](审查与修订报告.md)
- [修订运行指南](修订运行指南.md)
- [论文工作稿](solution/paper/论文.md)

`solution/paper/历史原稿/` 与 `历史说明/` 保存修订前内容。`E题_submission/`、`E题_队友评审包/` 为历史导出快照，不代表修订完成。当前无最终可提交的新协议预测包。
