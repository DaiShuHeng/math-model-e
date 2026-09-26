"""CTC forced alignment of the supplied transcript to its original audio.

Frozen facebook/wav2vec2-base-960h; no emotion labels or external training data.
Dynamic programming uses the standard blank/repeat/advance CTC transitions.
Generated times are automatic estimates; confidence and transcript match are saved.
"""
from __future__ import annotations
import argparse,json,re,os,difflib
from pathlib import Path
import config as C
import numpy as np,torch,pandas as pd
from p1_extract import decode_audio

def normalize_words(text):
    rows=[];digits=['ZERO','ONE','TWO','THREE','FOUR','FIVE','SIX','SEVEN','EIGHT','NINE']
    for match in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+",text):
        word=match.group();spoken=word.upper().replace('’',"'")
        if word.isdigit():spoken='|'.join(digits[int(c)] for c in word)
        rows.append({'word':word,'char_start':match.start(),'char_end':match.end(),'spoken':spoken})
    return rows

def ctc_path(logp,labels,blank):
    T,V=logp.shape;ext=np.full(len(labels)*2+1,blank,int);ext[1::2]=labels
    S=len(ext);prev=np.full(S,-np.inf,np.float32);prev[0]=logp[0,blank]
    if S>1:prev[1]=logp[0,ext[1]]
    trace=np.zeros((T,S),np.uint8)
    allow=(ext!=blank)&np.r_[False,False,ext[2:]!=ext[:-2]]
    for t in range(1,T):
        step1=np.r_[-np.inf,prev[:-1]];step2=np.r_[-np.inf,-np.inf,prev[:-2]];step2[~allow]=-np.inf
        options=np.stack([prev,step1,step2]);choice=options.argmax(0)
        prev=options[choice,np.arange(S)]+logp[t,ext];trace[t]=choice
    state=S-1 if prev[-1]>=prev[-2] else S-2
    if not np.isfinite(prev[state]):raise ValueError('Transcript cannot be aligned within emission length')
    states=np.empty(T,int)
    for t in range(T-1,-1,-1):states[t]=state;state-=int(trace[t,state]) if t else 0
    return states,ext

def align_one(text,audio,processor,model):
    words=normalize_words(text)
    transcript='|'.join(w['spoken'] for w in words)
    vocab=processor.tokenizer.get_vocab();labels=[vocab[c] for c in transcript]
    inputs=processor(audio,sampling_rate=16000,return_tensors='pt')
    with torch.no_grad():emission=model(**inputs).logits[0].log_softmax(-1).numpy()
    states,ext=ctc_path(emission,labels,model.config.pad_token_id)
    ratio=len(audio)/16000/len(emission);cursor=0
    for w in words:
        positions=[];scores=[]
        for j in range(len(w['spoken'])):
            state=2*(cursor+j)+1;frames=np.flatnonzero(states==state)
            if len(frames):positions.extend(frames.tolist());scores.extend(np.exp(emission[frames,ext[state]]).tolist())
        if not positions:raise ValueError('Word has no aligned frames')
        w.update(start=round(min(positions)*ratio,4),end=round((max(positions)+1)*ratio,4),confidence=round(float(np.mean(scores)),4))
        cursor+=len(w['spoken'])+1
    decoded=processor.batch_decode(emission.argmax(-1)[None,:])[0]
    return {'duration':len(audio)/16000,'words':words,'normalized_transcript':transcript.replace('|',' '),'greedy_transcript':decoded,'transcript_similarity':difflib.SequenceMatcher(None,transcript.replace('|',' '),decoded).ratio(),'mean_confidence':float(np.mean([w['confidence'] for w in words])),'time_source':'CTC forced alignment; automatic estimates, not human verified'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--only',choices=['a1','a4','all'],default='all');ap.add_argument('--threads',type=int,default=3)
    args=ap.parse_args();torch.set_num_threads(args.threads)
    from transformers import Wav2Vec2Processor,Wav2Vec2ForCTC
    processor=Wav2Vec2Processor.from_pretrained(args.model,local_files_only=True)
    model=Wav2Vec2ForCTC.from_pretrained(args.model,local_files_only=True).eval()
    tasks=[]
    if args.only in ('a1','all'):
        for r in pd.read_excel(C.A1_LABEL,dtype={'video_id':str,'clip_id':str}).itertuples(index=False):
            tasks.append(('a1',f'{r.video_id}__{r.clip_id}',str(r.text),C.A1_DIR/r.video_id/f'{r.clip_id}.mp4'))
    if args.only in ('a4','all'):
        from io_utils import _load_pkl
        for p in sorted(C.A4_ALIGNED_DIR.glob('*.pkl')):
            tasks.append(('a4',p.stem,str(_load_pkl(p)['raw_text']),C.A4_VIDEO_DIR/f'{p.stem}.mp4'))
    failures=[]
    for n,(group,sid,text,video) in enumerate(tasks,1):
        dest=C.DATA_DIR/'alignment'/group/f'{sid}.json';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():continue
        try:
            y=decode_audio(video);result=align_one(text,y,processor,model)
            result.update(sample_id=sid,transcript=text,model='facebook/wav2vec2-base-960h')
            dest.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{n}/{len(tasks)} {group}/{sid} words={len(result["words"])} confidence={result["mean_confidence"]:.3f} match={result["transcript_similarity"]:.3f}',flush=True)
        except Exception as e:
            failures.append({'group':group,'id':sid,'error':str(e)});print('FAILED',sid,str(e),flush=True)
    (C.DATA_DIR/'alignment'/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    if failures:raise RuntimeError(f'{len(failures)} alignment failures; samples retained')

if __name__=='__main__':main()
