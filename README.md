# 2026 数学建模 E 题：修订与优化训练

私有队内工作仓库。v3 已完成连续缺失协议下的训练与三种子核验；v4 完成强教师蒸馏与
第三问解释对照；**v5 完成缺失结构审计、三处实现缺陷修复与特征级鲁棒模型，是本轮推荐入口**。

## v5：缺失结构审计 + 特征级模型（本轮）

诊断结论：**性能上限由文本表征决定，不由融合结构决定**。同一融合结构与训练配方下，把
50 token 的 BERT-Tiny 表征换成附件 2 提供的 768 维文本特征，验证集 clean Macro-F1 由
0.5616 升至 0.6296；12 次独立训练（4 配置 × 3 种子）均值 0.6239±0.0070；按同一选择分数在验证集上选出的
规模为 4 的部署集成达 clean Macro-F1 0.6356 / 连续缺失 r20 0.6300，选择分数 0.6328，
相对第一轮交付模型的 0.5585 提升 7.4 个百分点。

同时发现并修复三处实现缺陷：最大值池化的 $-10^4$ 哨兵、池化未排除跨度外位置、文本
标准化掩码与池化掩码不一致。新增 16 项回归测试（合计 30 项）全部通过。

审计结论：赛题全部特征文件中音视频零行 **100% 落在文本 token 跨度之外**，附件 3 的
30 条里语音与视觉观测掩码 29/30 完全相同。详见
[缺失结构审计](solution/docs/缺失结构审计.md) 与 [实验记录 v5](solution/docs/实验记录_v5.md)。

```bash
cd solution
python -m tests.run_tests
python -m src.audit_missing_structure
python -m src.explain_att4_v5 \
  --ckpt weights/v5_deploy/c0_s2026.pt weights/v5_deploy/c2_s2026.pt \
         weights/v5_deploy/c2_s42.pt weights/v5_deploy/c3_s2026.pt \
  --out results/附件4_预测与解释_v5.csv
```

论文工作稿：[solution/paper/论文_v5.md](solution/paper/论文_v5.md)

## 与队友方案的对比（2026-09-25）

队友包 `E题_最终优化版` 报出的问题二 test 0.6754 / 0.6485、问题三 0.6327 / 0.5874
**已用他包自己的代码与权重复现，精确到 1e-6**。同条件正面对比（同一 test 划分）：

| 模型 | valid Macro-F1 | test acc | test Macro-F1 | test MAE |
|---|---:|---:|---:|---:|
| 队友 p2（3 种子，校准后） | 0.6348 | **0.6754** | **0.6485** | **0.6145** |
| 本方案 v5 部署集成（4 模型，校准后） | **0.6356** | 0.6657 | 0.6433 | 0.6500 |

**结论：队友的模型整体更好，主要赢在强度回归（MAE 差 0.0355）。分类上两者同级。**

他的优势来自**重建辅助目标 + 他的增广族 + 他的选择准则**这一整套组合。
我把四个想法逐个移植到本方案，**没有一个产生可迁移的收益**，跨队集成也没有收益。
完整分析、被证伪的结论与整合建议见
[两队方案对比与整合分析](solution/docs/两队方案对比与整合分析.md)；
复现脚本在 `solution/scripts/comparison/`。

## v4 入口：蒸馏优化

```bash
git pull --ff-only origin main
python -m pip install -r requirements-training.txt
bash scripts/train_distill_server.sh --seeds 2026  # 三组试跑
bash scripts/train_distill_server.sh               # 补齐三种子，共九组
```

完整说明、输入路径、缺失诊断和第三问命令见 [蒸馏优化训练指南](蒸馏优化训练指南.md)。该入口不运行 test 或附件 3/4，也不覆盖已有提交包。以下为此前 v3 的训练入口，保留用于复现。

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

- [论文 v5 工作稿](solution/paper/论文_v5.md)
- [两队方案对比与整合分析](solution/docs/两队方案对比与整合分析.md)
- [缺失结构审计 v5](solution/docs/缺失结构审计.md)
- [实验记录 v5](solution/docs/实验记录_v5.md)
- [模型优化方案与实验说明](模型优化方案与实验说明.md)
- [服务器输入核验](服务器输入核验.json)
- [审查与修订报告](审查与修订报告.md)
- [修订运行指南](修订运行指南.md)
- [论文工作稿](solution/paper/论文.md)

`solution/paper/历史原稿/` 与 `历史说明/` 保存修订前内容。`E题_submission/`、`E题_队友评审包/` 为历史导出快照，不代表修订完成。`E题_submission_v3/` 是已有候选快照；v4 实验不会自动覆盖它。
