# 队友原始交付代码（仓库此前缺失的部分）

这些是**实际产出仓库中已交付结果**的脚本，此前只在队友工作区，未入仓库。
`E题_复核修订工作包/02_模型与代码/code_最终模型/` 里已有其中一部分；
这里补齐的是**之前缺失的 9 个脚本 + 1 个测试**。

## 为什么入仓库

仓库中所有交付数字（test acc 0.6754 / Macro-F1 0.6485 / MAE 0.6145 / Pearson 0.6912、
决策偏置 `[-0.2,0,0]`、附件3/4 的预测与 Shapley）都由这些脚本产生。
没有它们，交付件不可复现。

## 文件

| 文件 | 作用 |
|---|---|
| `integrated_inference.py` | **统一推理入口**：用 p2 单一模型同时跑附件3与附件4 |
| `inference.py` | 三种子概率集成 + checkpoint 加载 |
| `io_utils.py` | 对齐特征读取、掩码构造、标准化统计 |
| `explain_v2.py` | 逐位置遮蔽效应 + 词表/CTC 对齐映射 |
| `explanation_validation.py` | 解释验证（关键位置删除 vs 随机删除） |
| `shapley_exact.py` | 8 联盟精确 Shapley 分解 |
| `build_final_tables.py` / `build_final_paper.py` | 生成最终 CSV 与论文表格 |
| `paper_extra.py` | 论文配图与补充统计 |
| `test_protocol.py` | 协议回归测试 |

## 相关新增

- `../models/linear_*.joblib`：单模态与融合线性基线（train 上拟合，无法由脚本再生）
- `../../01_交付结果/队友原始预测/`：队友原始版本的附件3/4预测与 Shapley，
  与修订版的差异可逐行对照
