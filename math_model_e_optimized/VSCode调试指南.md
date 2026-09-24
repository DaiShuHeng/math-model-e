# 在 VS Code 查看三个问题的结果

在 VS Code 用“文件 → 打开文件夹”打开本项目根目录，即含有 `code`、`data`、`models`、`results` 的目录。左侧选择“运行和调试”，从顶部列表选择对应任务，按 F5 运行。调试配置位于 `.vscode/launch.json`，题目数据路径位于 `.vscode/.env`；若原始数据被移动，只需改 `.env` 中的 `MATH_E_DATA`。

## 先看现成结果

不需要重新运行代码，直接在资源管理器中打开下列文件。CSV适合用 VS Code 的表格扩展或 Excel 查看；含图的正文用浏览器打开 `paper/论文修订稿.html`。

| 问题 | 关键结果 | 辅助结果 |
|---|---|---|
| 1 特征提取与时序对齐 | `data/p1_summary_v2.csv`：100条样本全量概览；`data/p1_features/*.npz`：每条50个时间窗的三模态特征与掩码 | `results/p1_grouped_cv.csv`：按原视频分组的交叉验证；`results/figures/p1_alignment_example.png`：原视频时间对应示例 |
| 2 模态缺失预测 | `results/附件3_优化预测.csv`：30条无标签预测；`results/model_metrics.csv` 中 `model=p2` 的验证和测试指标 | `results/robustness_valid.csv`：缺失类型、缺失率、段数和位置；`results/robustness_mask_audit.csv`：实际遮蔽审计；`results/p2_confusion_valid.csv`：验证集混淆矩阵 |
| 3 可解释预测 | `results/附件4_优化预测与解释.csv`：20条结果及每个模态的证据；`results/model_metrics.csv` 中 `model=p3` 的指标 | `results/evidence_time_mapping.csv`：词和原视频秒数；`results/explanation_validation.csv`：删除实验与区间；`results/figures/explanation_card.png`：典型解释图 |

`results/final_metrics.json` 同时保存问题2和问题3的完整验证集、测试集指标。附件3和附件4没有真实标签，因此各自预测CSV里没有真实正确率；有标签的准确率、F1、MAE来自附件2验证集和测试集。

## 运行与逐步调试

1. 选择左侧“运行和调试”中的任务，按 **F5**。程序打印的进度和指标显示在下方**终端**；运行结束后查看上表文件。
2. 想看中间变量时，在代码行号左侧单击添加红点断点，再按F5。暂停后查看左侧“变量”，或在“监视”中输入表达式。**F10**执行下一行，**F11**进入函数。
3. 问题1提取过程使用子进程；配置已启用 `subProcess`，可在 `p1_refine.py` 的 `media_features` 函数内设置断点。主流程可在 `main` 中保存NPZ和汇总CSV的语句处设置断点。
4. 问题2可在 `evaluate_final.py` 的 `calibrate` 或 `res=calibrated(...)` 处暂停，查看 `res['metrics']`、`res['p_cls'][:10]`、`res['p_reg'][:10]`。研究缺失实验时选“问题2：缺失率与位置实验”，在 `robustness_evaluation.py` 的 `corruption` 或 `metrics=` 处查看 `masked`、`aud` 和 `metrics`。
5. 问题3可在 `explain_v2.py` 的 `contribution`、`effects` 赋值之后暂停，查看模态删除影响和局部位置影响；`mapping_audit` 记录了可映射到词时间的情况。

请按需选择脚本。提取100条视频会重新写入问题1的特征和汇总文件；“基础指标与附件3预测”会重新写入两类模型的指标及附件3预测；解释任务会重新写入附件4的CSV。只想阅读已有结果时直接打开文件即可。
