"""Build figures and a paper draft strictly from saved experimental outputs."""
import json,html
from pathlib import Path
import config as C
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'savefig.dpi':170,'text.parse_math':False})

def table(df):return df.to_markdown(index=False,floatfmt='.4f',disable_numparse=True)
def read(name):return pd.read_csv(C.RESULT_DIR/name,dtype={'sample_id':str})
def figure(name):
    plt.savefig(C.FIG_DIR/name,bbox_inches='tight');plt.close()

def figures():
    robust=read('robustness_valid.csv');fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,d in robust[robust.family=='rate'].groupby('modality',sort=False):
        for ax,metric,title in zip(axes,['f1_macro','mae'],['宏平均 F1 越高越好','MAE 越低越好']):
            ax.errorbar(d.rate,d[metric],yerr=d[metric+'_std'],marker='o',markersize=3,label=name,capsize=2);ax.set(xlabel='新增缺失率',ylabel=title);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8,ncol=2);fig.suptitle('验证集连续缺失实验 每种情况重复 3 次');fig.tight_layout();figure('robustness.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,tag in zip(axes,['p2','p3']):
        df=pd.read_csv(C.RESULT_DIR/f'{tag}_confusion_valid.csv',index_col=0);a=df.to_numpy();ax.imshow(a,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(a[i,j]),ha='center',va='center',color='white' if a[i,j]>a.max()/2 else 'black')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=['负向','中性','正向'],yticklabels=['负向','中性','正向'],xlabel='预测类别',ylabel='真实类别',title=tag+' 验证集')
    fig.tight_layout();figure('confusion_valid.png')
    cv=read('p1_grouped_cv.csv');fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for ax,m in zip(axes,['f1_macro','mae']):
        g=cv.groupby('feature')[m].agg(['mean','std']);ax.bar(g.index,g['mean'],yerr=g['std'],capsize=4,color=['#5799ad','#3869a0','#df9c4c','#8c79ac']);ax.set(ylabel=m,title='问题1 按原视频分组交叉验证');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();figure('p1_grouped_cv.png')
    exp=read('附件4_优化预测与解释.csv');effects=np.load(C.RESULT_DIR/'explanation_effects.npz');i=0;r=exp.iloc[i];m=r.main_reference_modality;mi=list(C.MODALITIES).index(m)
    fig,axes=plt.subplots(2,1,figsize=(10,6),gridspec_kw={'height_ratios':[1,1.5]})
    values=[r[k+'_probability_drop_on_removal'] for k in C.MODALITIES];axes[0].bar(['文本','语音','视觉'],values,color=['#3869a0','#5799ad','#df9c4c']);axes[0].axhline(0,color='gray',lw=.8);axes[0].set(ylabel='移除后原预测类概率下降',title=f'样本 {r.sample_id}  {r.pred_polarity}  强度 {r.pred_intensity:.3f}  主要参考 {m}')
    e=effects['local_effect'][i,mi];valid=np.flatnonzero(np.isfinite(e));axes[1].bar(valid,e[valid],color='#3869a0');axes[1].set(xlabel='官方 BERT 对齐位置索引',ylabel='单位置移除概率下降',title='主要参考模态的局部扰动作用');axes[1].axhline(0,color='gray',lw=.8)
    fig.tight_layout();figure('explanation_card.png')
    # One original sample with wave, three feature traces, CTC word times and real frames.
    summary=pd.read_csv(C.DATA_DIR/'p1_summary_v2.csv',dtype={'clip_id':str});candidates=summary[(summary.duration_s>4)&(summary.duration_s<15)&(summary.transcript_similarity>.85)&(summary.face_detection_rate>.5)]
    row=(candidates if len(candidates) else summary).iloc[0];sid=row.sample_id.replace(C.SEP,'__');al=json.loads((C.DATA_DIR/'alignment/a1'/f'{sid}.json').read_text(encoding='utf-8'))
    from p1_extract import decode_audio
    import cv2
    path=C.A1_DIR/row.video_id/f'{row.clip_id}.mp4';audio=decode_audio(path);z=np.load(C.P1_DIR/f'{sid}.npz');mid=(z['time_edges'][:-1]+z['time_edges'][1:])/2
    fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(4,3,height_ratios=[1.2,1,1.2,1.4]);ax=fig.add_subplot(gs[0,:]);ax.plot(np.arange(0,len(audio),40)/16000,audio[::40],lw=.5,color='#5799ad');ax.set(xlim=(0,row.duration_s),ylabel='音频振幅',title=f'问题1 真实素材时间对应  {row.sample_id}')
    ax=fig.add_subplot(gs[1,:]);
    for n,w in enumerate(al['words']):
        ax.broken_barh([(w['start'],w['end']-w['start'])],(n%2,.7),facecolors='#adcadd');ax.text((w['start']+w['end'])/2,n%2+.35,w['word'],ha='center',va='center',fontsize=6,rotation=25)
    ax.set(xlim=(0,row.duration_s),ylim=(-.3,2),yticks=[],ylabel='CTC 词区间')
    ax=fig.add_subplot(gs[2,:]);
    for m,color in zip(C.MODALITIES,['#3869a0','#5799ad','#df9c4c']):
        a=np.linalg.norm(z[m],axis=1);a=a/max(a.max(),1e-8);ax.plot(mid,a,label=m,color=color);ax.scatter(mid[z[f'mask_{m}']==0],np.zeros(sum(z[f'mask_{m}']==0)),marker='x',s=10,color=color)
    ax.set(xlim=(0,row.duration_s),xlabel='原视频秒',ylabel='归一化特征范数');ax.legend(ncol=3,fontsize=8)
    cap=cv2.VideoCapture(str(path));frames=[]
    for j,frac in enumerate([.2,.5,.8]):
        t=frac*row.duration_s;cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=cap.read();ax=fig.add_subplot(gs[3,j]);ax.axis('off')
        if ok:ax.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        ax.set_title(f'原视频帧 {t:.2f} 秒',fontsize=9);frames.append(round(t,3))
    cap.release();fig.tight_layout();figure('p1_alignment_example.png')
    (C.RESULT_DIR/'typical_sample.json').write_text(json.dumps({'sample_id':row.sample_id,'transcript':row.text,'frame_times_s':frames,'duration_s':row.duration_s},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    C.PAPER_DIR.mkdir(exist_ok=True);figures()
    metrics=read('model_metrics.csv');cv=read('p1_grouped_cv.csv');base=read('linear_baselines_valid.csv');robust=read('robustness_valid.csv');expl=read('explanation_validation.csv');a3=read('附件3_优化预测.csv');a4=read('附件4_优化预测与解释.csv');p1=pd.read_csv(C.DATA_DIR/'p1_summary_v2.csv',dtype={'clip_id':str});quality=read('alignment_quality.csv')
    runs=pd.DataFrame(json.loads((C.RESULT_DIR/'validation_training_runs.json').read_text()));integrity=json.loads((C.RESULT_DIR/'integrity_checks.json').read_text(encoding='utf-8'));versions=json.loads((C.ROOT/'environment.json').read_text());typical=json.loads((C.RESULT_DIR/'typical_sample.json').read_text(encoding='utf-8'))
    final=json.loads((C.RESULT_DIR/'final_metrics.json').read_text());cols=['acc','f1_macro','mae','pearson'];display=metrics[['model','split',*cols]]
    old=pd.DataFrame([dict(model='p2',split='test',acc=.6492434663,f1_macro=.6185629137,mae=.6499853730,pearson=.6680502680),dict(model='p3',split='test',acc=.6492434663,f1_macro=.6227409623,mae=.6755879521,pearson=.6505095737)])
    comparison=pd.concat([old.assign(version='原版复现'),metrics[metrics.split=='test'].assign(version='优化版')]);comparison=comparison[['version','model',*cols]];comparison.to_csv(C.RESULT_DIR/'before_after_test.csv',index=False,encoding='utf-8-sig')
    p1table=p1[['sample_id','duration_s','time_bin_s','text_valid','audio_valid','vision_valid','face_detection_rate','processing_status']].copy()
    a4table=a4[['sample_id','pred_polarity','pred_intensity','main_reference_modality','text_normalized_abs_effect','audio_normalized_abs_effect','vision_normalized_abs_effect','alignment_status']]
    rate=robust[(robust.family=='rate')&(robust.rate.isin([0,.4,.8]))][['modality','rate','f1_macro','f1_macro_std','mae','mae_std']]
    summary_cv=cv.groupby('feature')[cols].mean().reset_index();cal={k:v['log_probability_bias'] for k,v in final.items()}
    needs=quality[quality.needs_review].copy();needs_a4=needs[needs.dataset=='a4'].sample_id.tolist()
    class_rows=[]
    for tag in ['p2','p3']:
        conf=pd.read_csv(C.RESULT_DIR/f'{tag}_confusion_valid.csv',index_col=0).to_numpy()
        for i,name in enumerate(['Negative','Neutral','Positive']):class_rows.append(dict(model=tag,label=name,support=int(conf[i].sum()),recall=conf[i,i]/max(conf[i].sum(),1)))
    class_table=pd.DataFrame(class_rows)
    errors=read('p2_validation_details.csv').nlargest(4,'absolute_error').copy();errors['text']=errors.text.str.slice(0,110)
    high_missing=robust[(robust.family=='rate')&(robust.rate==.8)].sort_values('f1_macro');worst=high_missing.iloc[0]
    paper=f'''# 基于有效位置掩码和输入扰动解释的多模态情感预测

## 摘要

针对原始视频时序组织、局部模态缺失和预测依据复核三个问题，建立以可核验输入接口为基础的建模流程。问题1使用冻结 BERT、声学描述符和视觉像素网格描述符，并用冻结语音识别模型的 CTC 强制对齐结果将词映射到物理时间窗；完整保存100条样本及其有效掩码。问题2区分填充位置和观测缺失，通过模态专属 Transformer、有效比例门控与人工遮蔽重建学习三分类及强度回归。问题3采用输入删除实验估计模态和局部片段对预测的作用，将关键位置映射回词区间。模型仅使用附件2训练集学习情感参数，验证集选择训练轮次与决策偏置，独立测试集用于最终评价。问题2测试集宏平均 F1 为 {final['p2']['test']['f1_macro']:.4f}、MAE 为 {final['p2']['test']['mae']:.4f}；问题3相应为 {final['p3']['test']['f1_macro']:.4f} 与 {final['p3']['test']['mae']:.4f}。自动时间定位和删除作用都存在解释边界，因此另行报告对齐质量与配对删除检验。

关键词：多模态情感预测；有效掩码；连续缺失；CTC 时间对齐；输入扰动

## 一 问题分析与数据约定

附件1为100个视频片段、来自37个原视频；附件2按官方划分使用训练3395条、验证728条、测试727条；附件3含30个无标签缺失样本，附件4含20个无标签解释样本。全程使用 aligned 版本。标签定义为 y<0 负向、y=0 中性、y>0 正向，回归范围为[-3,3]。正文主报告三分类 Accuracy 和宏平均 F1；CSV中的辅助二分类将零归入非负类。

预训练 BERT 和 wav2vec2 仅作冻结特征提取与时间定位，不在额外情感数据集上训练或微调。问题1自建的74维音频和35维视觉特征，与附件2同维但定义不同，不能宣称等价于其官方特征，故分别进行评估。问题2和问题3直接读取附件2的数值特征；附件3的整数 text_bert 通过相同 BERT 的最后一层直接编码。8条训练或验证样本的接口核查中，直接编码与官方 text 的 RMSE 约为10⁻⁶；沿词轴插值会破坏这一对应。

主要假设为：附件2 aligned 三模态的位置与有效文本词元相对应；给定转写描述对应音轨；人工连续遮蔽能够近似局部信息不可用。前两项通过字段核对、词元编号比对和对齐质量记录检查；第三项仍有外部有效性局限。音频噪声与遮挡也可能造成非零损坏，本研究的置零实验不能覆盖全部真实故障。

## 二 问题1 特征提取与物理时间对齐

### 2.1 特征定义

文本使用冻结 bert-base-uncased 的768维最后层隐状态。对转写的每个词，依据字符偏移聚合其 BERT 子词。音频以16 kHz单声道解码，窗长25 ms、步长10 ms，构造 MFCC 及一二阶差分39维、对数能量及差分与过零率3维、对数基频及一二阶差分3维、谱形7维、谱对比7维、色度12维、谱统计3维，共74维。音高在无声或噪声区可能不可靠，相关输出不解释为精确生理测量。

视觉以10 fps抽帧并检测人脸，使用样本内检测框的中位数形成固定裁剪区域，缩放至56×40灰度图。以有效帧的逐像素中位数图为参考，计算7×5网格内平均绝对像素差，并以90分位值归一化、截断到[0,5]。未检出人脸的帧不进入时间窗均值。该35维量是外观变化描述符，不能等同于面部动作单元或精确表情强度；头动和光照仍会影响它。

### 2.2 对齐模型与存储

对冻结 wav2vec2-base-960h 在16 kHz音轨上的 CTC 对数概率，建立插入 blank 的字符状态序列。动态规划递推为 D(t,s)=log pₜ(cₛ)+max(D(t−1,s),D(t−1,s−1),D(t−1,s−2))，跨两状态转移仅在非 blank 且标签不重复时允许。回溯最优单调路径获得词的起止区间。数字逐位展开并保留原文本字符偏移，可能与真实读法不同，列为自动对齐局限。

对视频时长T建立50个物理时间窗 Iⱼ=[jT/50,(j+1)T/50)。音视频特征对落入该窗且有效的帧取均值。文本按词区间与时间窗的交叠时长加权：xⱼᵗ=Σᵢ|Iⱼ∩Wᵢ|hᵢ / Σᵢ|Iⱼ∩Wᵢ|。无有效帧或无交叠词的窗为零，同时有效掩码为0；不通过重复特征伪造观测。这里50个时间窗与附件2的50个词元存储位置是两种接口。

每条NPZ保留 text(50,768)、audio(50,74)、vision(50,35)，三类mask，51个time_edges和6项meta（音频有效时长、音频帧数、视频帧数、词数、标签、人脸检出比例）；schema_version 为 physical_time_bins_v2。原始标签不改动，失败样本仍保留并记录原因。本次100条处理异常数为 {integrity['p1_media_failures']}。有效时长由实际16 kHz解码音轨计算，范围 {p1.duration_s.min():.3f}～{p1.duration_s.max():.3f} 秒；题面描述的时长范围与这一测量口径不同，本研究保留实测值，不修改源文件。

### 2.3 典型样本和有效性检查

典型样本编号为 {typical['sample_id']}，转写为“{typical['transcript']}”。下图同时展示音频波形、自动词区间、时间窗特征范数和原视频帧，零范数及叉号提示该窗没有相应观测。时间定位为自动估计，尚未逐词人工核验。

![原始素材与特征的时间对应](../results/figures/p1_alignment_example.png)

对100条样本，以原视频video_id为分组，使用随机种子2026、2027、2028的五折 StratifiedGroupKFold。每折仅在训练折拟合标准化、逻辑回归和Ridge，分类C=0.3，回归α=10；同一原视频不会同时进入训练与验证。以下为三次完整折外预测指标的平均，误差条展示不同分组种子的标准差，不解释为独立置信区间。

{table(summary_cv)}

![分组交叉验证](../results/figures/p1_grouped_cv.png)

100条样本较少且视觉描述符较粗，这一实验反映内部有效性，不能据此断言与官方特征等效。原版随机片段划分的结果与本表协议不同，不作直接提升比例比较。

### 2.4 全量特征汇总

下表所有样本均含文本、语音、视觉三模态，维度统一为50×768、50×74、50×35；time_bin_s为对齐粒度，三个valid列为非填充有效时间窗数。该表与 data/p1_summary_v2.csv 及100条NPZ逐一对应。

{table(p1table)}

## 三 问题2 局部缺失鲁棒预测

### 3.1 有效掩码与标准化

令 Sₘⱼ 表示内容支持位置、Oₘⱼ 表示实际观测、Aₘⱼ 表示本轮人工保留。支持掩码从BERT注意力掩码并排除[CLS]、[SEP]和padding得到，实际观测还要求对应数值有限且非全零。模型输入掩码 M=O·A，填充不计入缺失率。均值和标准差仅使用训练集的实际观测位置估计，缺失位置标准化后仍置零。

新增缺失率r以人工遮蔽前可见位置数L为分母，精确删除round(rL)个位置。连续区块由正整数长度和非负间隙组成，内部间隙至少1；只有几何上不可能时将段数截断至min(k,n,L−n+1)。连续性首先定义在有序有效位置上；遇到天然空洞时，存储轴上的实际段数可能不同，因此同时记录实际段数。短序列取整导致实际比率与目标比率不同，CSV保留这一差异。

附件3已删除文本的原始词元总数不可由当前文件确定，故报告的是现存词元支持序列上的观测情况，不声称恢复真实原始文本缺失率。

### 3.2 模型与目标函数

各模态线性映射至96维，加入位置编码，使用一层四头Transformer，前馈维度192、dropout=0.25。注意力池化得到模态表示zₘ，实际观测比例aₘ=ΣMₘ/ΣSₘ参与门控：α=softmax(g(zₘ)+log aₘ)。完全不可用模态在融合中屏蔽；全部不可用时使用有限的退化输出，保证数值稳定，但不将其视为可靠预测。

融合表示z=Σαₘzₘ连接三分类头与回归头。强度为3 tanh(f(z)/3)。目标 L=L_CE+0.6 L_Huber+0.1 L_rec，分类含训练集频数倒数权重与0.05标签平滑，Huber参数β=0.5。重建只针对人工遮蔽且原本可见的位置，先按每模态位置数×特征维数平均，再在参与模态间平均，避免768维文本隐含获得更大的损失权重。

AdamW初始学习率6×10⁻⁴、权重衰减10⁻⁴，batch=64，最多35轮、10%预热和余弦衰减，梯度裁剪1；验证分数F1_macro−MAE/6选择轮次，7轮无改善停止。问题2每条训练样本以0.6概率实施0.1～0.8新增缺失，随机选模态组合和1～3段。固定训练三个种子并等权平均分类概率和回归输出，不依据测试表现挑选种子。单模型参数量为462546。

最终对数分类概率偏置仅在验证集从{{−0.2,0,0.2}}的负向、中性二维网格中选择，正向固定为0。保存的p2和p3偏置分别为 {cal['p2']} 与 {cal['p3']}。所有测试指标在这一选择之后计算，未使用专项测试集的标签或伪标签。

### 3.3 基础结果与对照

{table(display)}

线性基线使用各模态有效位置均值，训练集拟合标准化；逻辑回归C在0.01、0.1、1中选择，Ridge α在1、10、100、1000中选择，二者分别以验证F1和MAE选择。其验证结果如下；线性基线与主模型的能力应按实际指标判断，不能预设复杂网络更优。

{table(base[['model','C','ridge_alpha',*cols]])}

训练种子与重建消融结果如下。去掉重建的对照仅运行一个种子，不能据单次结果宣称统计显著提升；三种子集成与单种子消融也不属于等计算量比较。

{table(runs[['tag','seed','best_epoch',*cols]])}

![验证集混淆矩阵](../results/figures/confusion_valid.png)

两类模型在验证集的逐类召回率如下，可直接检查中性类别和类别不平衡影响：

{table(class_table)}

独立测试集与原版复现的比较如下。这是整条建模流程的前后对比，掩码、损失、集成和输入处理同时变化，不能归因到单一改动。测试集重采样区间另见 results/test_bootstrap_intervals.csv；普通样本bootstrap未校正同原视频片段之间的相关性，不作显著性结论。

{table(comparison)}

问题2四项主指标均优于原版复现。问题3回归MAE和相关系数改善，但准确率与宏平均F1下降；验证集的改善未完整转化为测试分类泛化。因此不能宣称问题3预测质量全面提升，本轮对其最确定的改进在于可核验的解释和输入处理。结果冻结后不再利用测试表现重新挑选偏置或种子。

### 3.4 缺失规律与误差分析

在完整728条验证集上，固定新增缺失种子11、22、33，考察7种模态组合、0～0.8缺失率，另外在0.4缺失率下考察2/4/8段及前/中/后位置。表中std为三次遮蔽重复的标准差，不是训练不确定性。

{table(rate)}

![缺失率与性能](../results/figures/robustness.png)

不同位置和段数的完整结果保存在 robustness_valid.csv，实际缺失审计保存在 robustness_mask_audit.csv。缺失率越高可见证据通常越少，但有限样本上的指标不必严格单调；遮蔽偶尔也会去掉干扰输入。语义、韵律、外观的作用具有样本依赖性，不能仅凭总体门控平均值推断因果重要性。验证集逐样本真实值、预测值和文本见 p2_validation_details.csv；需重点检查零附近强度的类别边界、短文本以及模态冲突样本。

在0.8新增缺失率下，本次宏平均F1最低的组合为 {worst.modality}，F1为 {worst.f1_macro:.4f}、MAE为 {worst.mae:.4f}。固定0.4缺失率改变段数与位置的结果如下；前、中、后删除是确定性位置，同一模型下重复种子不增加独立信息。

{table(robust[robust.family!='rate'][['family','modality','segments','position','f1_macro','mae']])}

问题2验证集强度绝对误差最大的4条样本如下。片段级情感标签与文字字面极性未必一致，此处展示原始错误，不依据这些个案重新选择测试模型。

{table(errors[['sample_id','true_class','pred_class','true_intensity','pred_intensity','absolute_error','text']])}

### 3.5 附件3全量预测

以下为冻结模型的最终推理，confidence为分类概率最大值，未作概率可信度校准，不等同于真实正确率。

{table(a3)}

## 四 问题3 可量化作用与证据回查

### 4.1 预测模型与解释量

问题3与问题2共用网络结构，人工缺失概率降为0.1，关闭重建和互补正则；其余训练及三种子集成规则相同。分类和回归使用两个任务头，类别与强度符号可能不一致，CSV保留该状态，不为表面一致而用测试信息改写输出。

对原预测类别c，模态删除作用定义为Δₘ=p(c|x)−p(c|x去掉模态m)。正值表示该模态支持当前类别，负值表示移除后当前类更有把握；主要参考模态取|Δₘ|最大者，归一化作用为|Δₘ|/Σ|Δ|。同时保存回归强度变化和门控α。由于完全移除可能产生分布外输入、模态间存在交互，该量是模型扰动敏感性，不能解释为唯一因果贡献，三模态作用也不具可加性。

对局部位置逐一删除并计算原预测类别概率变化，按下降量排序选取最多3个不同词的证据，保留有符号数值；如果所有下降量均非正，不应把它们称为正向支持证据。BERT词元编号与原文本重新分词逐位校验，只对一致位置或一致前缀做时间映射。CTC给出原词区间，同一对齐位置的音频和视觉证据映射到该词区间，而非机械使用jT/50。没有确认映射的位置保留“无时间戳”，不会编造时间。

### 4.2 解释检验与典型卡片

在事先固定的128条验证样本上，分别删除10%、20%、40%的高注意力位置，并与5次等量随机删除作配对比较。置信区间来自2000次样本bootstrap。该检验测试注意力是否能辅助筛选候选位置，不构成对逐位置删除解释的独立全面验证。

{table(expl)}

配对差值大于0表示注意力候选删除造成更大概率下降；只有区间整体高于0才提供该样本协议下的支持证据，否则不作优于随机的结论。注意力β与门控α保留为内部权重，最终证据以实际扰动结果为依据。

![典型解释卡及局部作用分布](../results/figures/explanation_card.png)

典型样本 {a4.iloc[0].sample_id} 的文本为“{a4.iloc[0].text}”。其预测为 {a4.iloc[0].pred_polarity}，强度 {a4.iloc[0].pred_intensity:.4f}，主要参考模态为 {a4.iloc[0].main_reference_modality}。各模态前三个候选证据如下，时间单位秒：

'''
    for m in C.MODALITIES:
        entries=json.loads(a4.iloc[0][m+'_evidence']);paper+=f'\n{m}\n\n'+table(pd.DataFrame(entries)[[c for c in ['position','word','start_s','end_s','probability_drop','alignment_confidence'] if c in pd.DataFrame(entries)]])+'\n'
    paper+=f'''
### 4.3 附件4全量预测与解释

下表三类effect列为归一化绝对删除作用，方向见结果CSV中的probability_drop_on_removal列。每条样本每种模态的词、区间、局部作用与池化权重完整保存在 附件4_优化预测与解释.csv 和 evidence_time_mapping.csv。以贪心识别文本和提供转写的字符相似度0.65作为人工回查提示；该阈值仅用于展示质量标记，不用于训练或挑选预测。附件4需优先人工复核的编号为 {', '.join(needs_a4) if needs_a4 else '无'}，所有其余定位也仍是自动结果。

{table(a4table)}

### 4.4 关键证据全量索引

每行按文本、音频、视觉给出最高排序证据及起止秒，便于直接在原视频回看；多词详细输出在CSV中保留。

'''
    evidence_rows=[]
    for _,r in a4.iterrows():
        er={'sample_id':r.sample_id}
        for m in C.MODALITIES:
            entries=json.loads(r[m+'_evidence']);e=entries[0] if entries else {};er[m]=f"{e.get('word','未验证位置')} [{e.get('start_s','?')}, {e.get('end_s','?')}] Δ={e.get('probability_drop',0):.4f}"
        evidence_rows.append(er)
    paper+=table(pd.DataFrame(evidence_rows))
    paper+=f'''

## 五 局限与改进方向

首先，100条自建特征评估规模较小，按原视频分组后结果更接近对新原视频的泛化，但分组波动仍明显。视觉像素网格无法稳定排除头动和光照，后续可在保留相同评价协议的条件下引入冻结人脸关键点或动作单元工具，并作同样本对照。其次，CTC依赖转写与音轨相符，数字读法、口语缩略和背景声会引起位置偏移；本次120条自动定位中有 {len(needs)} 条达到优先回查条件，不应把高对齐分数解释为人工确认的正确时间。

再次，置零缺失不能完全模拟噪声、识别错误和遮挡等非零损坏。附件3缺失前的文本长度不可恢复，不能计算原始真实缺失比例。模型预测概率尚未做可靠性校准；独立任务头也可能给出类别与回归方向不一致的输出。最后，删除实验估计模型响应，不能替代真实因果归因，多种子集成虽减少单次训练偶然性，但扩大了推理成本。下一阶段应在训练及验证集上预先确定比较方案，优先改善弱模态特征与跨模态冲突样本，并将新的测试评价与本轮保持分离。

## 六 复现与文件说明

详细命令见项目根目录README.md。预测使用保存的6份主模型权重、训练统计量与验证偏置。完整复现按时间对齐、问题1特征、分组交叉验证、训练、线性基线、冻结评价、鲁棒实验、解释生成顺序运行。桌面环境按顺序执行重型步骤，避免同时加载多份附件2与预训练模型。

实测运行环境如下：

{table(pd.DataFrame(list(versions.items()),columns=['组件','版本']))}

100条NPZ位于data/p1_features，120条对齐记录位于data/alignment，训练轮次与损失记录位于models各子目录的history.json。results目录包含全量预测、验证明细、混淆矩阵、鲁棒审计和解释明细。模型和代码之外的原始数据需从题目附件读取。冻结特征提取预训练权重不装入提交压缩包；download_models.py提供获取入口，问题1已生成特征及时间记录随包提供。

## 参考资料

1. 赛题组提供的2026年E题题面与四份附件，数据接口与提交要求以本地题面为准。
2. Devlin J, et al. BERT Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019. https://aclanthology.org/N19-1423/
3. Facebook AI. wav2vec2-base-960h model card. https://huggingface.co/facebook/wav2vec2-base-960h （冻结模型，16 kHz输入）
4. Jain S, Wallace B C. Attention is not Explanation. NAACL, 2019. https://aclanthology.org/N19-1357/
5. scikit-learn. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html
'''
    (C.PAPER_DIR/'论文修订稿.md').write_text(paper,encoding='utf-8')
    import markdown
    body=markdown.markdown(paper,extensions=['tables','fenced_code']);style='body{max-width:1100px;margin:40px auto;padding:0 28px;font-family:Microsoft YaHei,sans-serif;line-height:1.8;color:#202c3a}h1{font-size:28px}h2{margin-top:40px;border-bottom:1px solid #dae1e6}table{border-collapse:collapse;font-size:12px;display:block;overflow:auto;margin:18px 0}th,td{padding:5px 9px;border:1px solid #dce3e8}th{background:#eef3f7}img{max-width:100%}code{font-family:Consolas,monospace}@media print{body{margin:0;padding:0}table{font-size:9px;display:table}tr,img{break-inside:avoid}h2{break-after:avoid}}'
    (C.PAPER_DIR/'论文修订稿.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>多模态情感预测论文修订稿</title><style>'+style+'</style><body>'+body+'</body></html>',encoding='utf-8')
    improvement=f'''# E题优化结果与使用说明

本次已完成代码修复、7份模型重训（两类主模型各3个种子及1份重建消融）、100条物理时间特征重建、30条缺失样本预测和20条预测解释。原项目保留，优化结果在本目录中独立提供。

## 核心变化

- 修正text模态名被逐字符重复遮蔽的问题，缺失率和连续段数都有实际掩码审计。
- 分离填充和观测缺失，用真实内容长度计算可用比例；修正附件3的直接BERT接口。
- 重建损失按特征维数平均，避免高维模态主导；加入多种子及线性对照。
- 问题1使用CTC词时间和物理时间窗，补全掩码、元数据和100条全量表；交叉验证按原视频分组，标准化在折内拟合。
- 解释使用有符号输入删除作用，核对词元映射，保留自动定位质量标记；撤去未经验证的注意力忠实性断言。

## 真实指标对比

以下是727条同一官方测试集上的整体流程结果，不能归因到某一项改动，也不代表每项指标都提高。

{table(comparison)}

问题2四项主指标均改善。问题3的MAE从0.6756降到 {final['p3']['test']['mae']:.4f}，但宏平均F1从0.6227降到 {final['p3']['test']['f1_macro']:.4f}，这一分类泛化退步尚未解决，不能把本版描述为所有任务全面优于原版。为保持本轮测试评价有效，未据此反复调参。

## 阅读顺序

先打开 paper/论文修订稿.html，浏览方法、真实表格与图；可编辑源为 paper/论文修订稿.md。然后查看results中的两份中文预测CSV。完整命令在README.md，既可只预测，也可从原始数据重新计算。文稿是依据已运行结果生成的修订稿，仍需按比赛统一格式排版。

## 仍需注意的具体问题

附件4优先回查自动词时间的编号：{', '.join(needs_a4) if needs_a4 else '无'}。其余时间也未逐词人工核验。视觉仍使用较粗的网格外观特征；删除解释不等于因果归因；专项集无标签，无法报告真实准确率。完整实验和局限已写入论文，未用编造数据补齐结论。
'''
    (C.ROOT/'优化结果说明.md').write_text(improvement,encoding='utf-8')
    print('REPORTS COMPLETE',len(paper),flush=True)
if __name__=='__main__':main()
