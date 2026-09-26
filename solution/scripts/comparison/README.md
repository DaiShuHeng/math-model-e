# 两队方案对比脚本

这些脚本复现 `docs/两队方案对比与整合分析.md` 的全部数字。它们**不是**提交包的一部分，
放在这里是为了让对比结论可被独立重跑。

| 脚本 | 作用 |
|---|---|
| `verify_teammate.py` | 用队友包自己的代码与权重复现他报出的 test 指标（精确到 1e-6） |
| `head2head.py` | 在同一个 test 划分、同一套指标下对比两队模型（含 valid 拟合的校准） |
| `exp12_integrate.py` | 队友增广族下逐个移植他的四个想法（重建项 / 缺失令牌 / 标签平滑 / 门控先验） |
| `exp13_calib.py` | 校准迁移与缺失令牌在另一套训练配置下的复核 |
| `exp14_seq.py` | 把队友的"模态序列按 α 融合 + Transformer + 重建"结构搬进本方案 |
| `exp15_ensemble.py` | 跨队概率集成（贪心 + 全部组合，只用 valid 选择） |
| `exp16_criterion.py` | 选择准则移植（他的 `f1 − 0.5·mae/3` vs 本方案的 clean+r20 均值） |
| `exp17_evidence.py` | 证据定位正面对比（双方证据位置在同一模型上一起删除） |

运行前提：

```bash
export MATH_E_DATA=<包含 附件1..附件4 的目录>
export PYTHONPATH=<math-model-e>/solution:<队友包>/code
```

脚本里的绝对路径常量需要按实际位置调整。所有脚本只读赛题数据；除 `exp17` 读取队友的
解释 CSV 外，不读取任何外部标签。
