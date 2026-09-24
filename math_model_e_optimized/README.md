# E题多模态情感预测优化版

本项目包含已完成的100条自建特征、两类专项模型、30条缺失预测和20条可回查解释，以及依据真实实验结果生成的论文修订稿。

## 先查看结果

1. 双击 `paper/论文修订稿.html`，用桌面浏览器阅读图表和完整正文。修改正文可编辑同目录的Markdown文件。
2. 用Excel打开 `results/附件3_优化预测.csv` 与 `results/附件4_优化预测与解释.csv`。导入时把sample_id列设为文本，以保留01等编号的前导零。
3. 阅读根目录 `优化结果说明.md`，了解前后对比和已知局限。文稿仍需按比赛统一格式排版，自动词时间需要结合原视频复核。

## 环境和数据路径

实际验证环境见 `environment.json`；依赖版本见 `requirements.txt`。推荐使用独立Python环境，运行命令中的 `python` 应指向装有这些依赖的解释器。无需显卡。需要可用的内存空间，重型步骤顺序执行。

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
$env:MATH_E_DATA='D:\题目数据\E题数据'
$env:PYTHONIOENCODING='utf-8'
```

`MATH_E_DATA`须指向直接包含“附件1-数据集原始多模态样本”“附件2-数据集特征文件”“附件3-模态缺失特征样本”“附件4-可解释专项视频样本与特征文件”的目录。原始附件不装入本压缩包，目录结构与题目提供的数据保持一致。输入PKL仅限自己信任的题目文件。

## 只复现专项预测

```powershell
python code/predict_only.py
```

此命令加载已保存的模型、训练统计量、验证决策偏置和CTC时间记录，不重新训练，也不重新选择阈值。附件3直接BERT特征缓存随包提供。附件4解释映射需要BERT分词器；若机器没有对应预训练缓存，请先按下一节下载并设置路径。模型输出的confidence是最大分类概率，尚未验证其概率校准质量。

## 获取冻结特征提取模型

```powershell
python download_models.py --directory external_models
$env:MATH_E_BERT=(Resolve-Path 'external_models/bert-base-uncased').Path
```

网络访问官方站点不便时可显式指定 `--endpoint https://hf-mirror.com`。下载包含冻结BERT与wav2vec2预训练参数；其体积较大，不纳入提交附件。未使用额外情感数据进行训练、微调或选择阈值。

## 从原始视频重新生成特征和时间记录

```powershell
python code/align_media.py --model external_models/wav2vec2-base-960h --only all --threads 3
python run_pipeline.py --data-root $env:MATH_E_DATA --start features
```

第一条命令保留已有对齐记录，缺少时生成。若需要重新计算已有对齐，请在项目副本中先移走 `data/alignment/a1` 和 `data/alignment/a4` 两个记录目录。第二条依次执行特征提取、按原视频分组验证、线性基线、固定模型评价、缺失实验、解释与删除检验。每步日志在logs中，失败会停下并指出日志文件；可用 `--start 步骤名` 从某一步恢复，也可用 `--only 步骤名` 单独运行。

## 重新训练与生成报告

```powershell
python code/train_experiments.py --force
python run_pipeline.py --data-root $env:MATH_E_DATA --start baselines
python code/analyze_results.py
python code/build_deliverables.py
python -m unittest discover -s tests -v
```

`--force`会在当前项目副本中覆盖现有训练输出；需要保留本次结果时，请先复制项目。默认不带该参数时，训练跳过已经完成的种子。Python或底层库版本不同、不同硬件线程实现可能带来轻微数值差异。

`build_deliverables.py`需要原始附件1生成典型视频帧示意图，并使用已完成的全部结果生成正文。Markdown转HTML额外使用 `markdown`，表格使用 `tabulate`，均记录在环境清单。

## 数据和评价约定

- 情感模型仅在附件2训练集3395条上学习；验证集728条选轮次与分类偏置；测试集727条只作固定模型最终评价。
- 问题1自建物理时间窗特征与问题2、3的官方词元位置接口分开使用，不能以维度相同声称特征语义相同。
- 三类标签编号为0负向、1中性、2正向。回归标签等于零时只有中性。辅助二分类把零纳入非负类。
- 内容支持掩码与观测掩码分开；标准化只使用训练观测；人工缺失相对于当前有效观测定义。
- 人工连续段按有序有效位置生成，并审计存储轴上的实际段数。附件3缺失前文本长度未知，不推断其真实原始缺失率。
- CTC词时间为自动估计。`alignment_quality.csv`给出需要优先回查的记录；`evidence_mapping_audit.csv`核对词元编号与可定位位置。
- 删除作用为有符号概率变化；负值表示移除后原预测类概率上升。归一化绝对作用只表示敏感性大小，不是因果概率。完全模态删除是解释诊断，不改变题目所要求的局部缺失预测任务。
- 分类与回归独立任务头可能出现符号不一致，结果保持原模型输出；`head_sign_agreement`可用于定位复核。

## 文件索引

| 路径 | 内容 |
|---|---|
| data/p1_features | 100条NPZ，含三类特征、三类掩码、51个物理时间边界、6项meta及schema_version |
| data/p1_summary_v2.csv | 100条样本的编号、时长、有效窗、对齐与人脸检测质量 |
| data/alignment | 100条问题1及20条问题3的自动词时间，含原文本字符偏移 |
| data/cache | 训练统计量与已核实接口的附件3直接BERT特征缓存 |
| models/p2_* 和 models/p3_* | 各三个种子权重、验证指标与逐轮历史 |
| models/p2_no_reconstruction_2026 | 去掉重建任务的单种子消融 |
| models/decision_calibration.json | 验证集选择后固定的分类对数概率偏置 |
| results | 全量预测、评价表、审计表、解释与图 |
| paper | 依据本次真实实验结果生成的可编辑正文和离线HTML |
| tests | 缺失率、段数、填充、全缺失数值稳定性、源数据不变性及损失尺度测试 |

NPZ的meta顺序为：音频有效时长秒、音频帧数、视频帧数、词数、情感强度标签、人脸检出比例。读取示例：

```python
import numpy as np
from pathlib import Path
p = next(Path('data/p1_features').glob('*.npz'))
with np.load(p) as z:
    print(z['text'].shape, z['audio'].shape, z['vision'].shape)
    print(z['time_edges'], z['mask_text'], z['meta'])
```

正文引用的官方模型与方法资料均在论文末尾列出。整个结果包未修改原始题目文件或原工程。
