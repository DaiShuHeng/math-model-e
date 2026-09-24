"""Generate all 100 time-aligned features using supplied transcripts and CTC times."""
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
import config as C
import numpy as np,pandas as pd
from p1_extract import decode_audio,acoustic_features,visual_features

def pool_frames(feats,times,edges,observed=None):
    observed=np.ones(len(feats),bool) if observed is None else np.asarray(observed,bool)
    out=np.zeros((len(edges)-1,feats.shape[1]),np.float32);mask=np.zeros(len(out),np.float32)
    for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        pick=(times>=a)&(times<b)&observed
        if pick.any():out[j]=feats[pick].mean(0);mask[j]=1
    return out,mask

def media_features(job):
    sid,path=job;y=decode_audio(path);duration=len(y)/16000
    audio=acoustic_features(y);vision,vfps,detected=visual_features(path,return_details=True)
    edges=np.linspace(0,duration,51)
    a,am=pool_frames(audio,np.arange(len(audio))*.01,edges)
    v,vm=pool_frames(vision,(np.arange(len(vision))+.5)/vfps,edges,detected)
    return sid,dict(audio=a,vision=v,mask_audio=am,mask_vision=vm,time_edges=edges.astype(np.float32),duration=duration,n_audio_frames=len(audio),n_visual_frames=len(vision),face_detection_rate=float(detected.mean()) if len(detected) else 0.)

def pool_text(hidden,offsets,alignment,edges):
    out=np.zeros((50,768),np.float32);weight=np.zeros(50,np.float32)
    for word in alignment['words']:
        indexes=[i for i,(a,b) in enumerate(offsets) if b>a and a<word['char_end'] and b>word['char_start']]
        if not indexes:continue
        embedding=hidden[indexes].mean(0)
        for j,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
            overlap=max(0,min(right,word['end'])-max(left,word['start']))
            if overlap:out[j]+=embedding*overlap;weight[j]+=overlap
    mask=(weight>0).astype(np.float32);out/=np.maximum(weight[:,None],1e-8)
    return out,mask

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    import torch
    torch.set_num_threads(2)
    from text_encoder import load_bert
    tokenizer,bert=load_bert()
    labels=pd.read_excel(C.A1_LABEL,dtype={'video_id':str,'clip_id':str});texts={}
    for row in labels.itertuples(index=False):
        sid=f'{row.video_id}__{row.clip_id}'
        alignment=json.loads((C.DATA_DIR/'alignment/a1'/f'{sid}.json').read_text(encoding='utf-8'))
        enc=tokenizer(str(row.text),return_tensors='pt',return_offsets_mapping=True,truncation=True,max_length=512)
        offsets=enc.pop('offset_mapping')[0].numpy()
        with torch.no_grad():hidden=bert(**{k:v.to(next(bert.parameters()).device) for k,v in enc.items()}).last_hidden_state[0].cpu().numpy()
        edges=np.linspace(0,alignment['duration'],51)
        tf,tm=pool_text(hidden,offsets,alignment,edges)
        texts[sid]=(tf,tm,alignment)
    jobs=[(f'{r.video_id}__{r.clip_id}',str(C.A1_DIR/r.video_id/f'{r.clip_id}.mp4')) for r in labels.itertuples(index=False)]
    media={};fail=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(media_features,j):j[0] for j in jobs}
        for f in as_completed(futures):
            sid=futures[f]
            try:
                _,result=f.result();media[sid]=result
            except Exception as e:
                fail.append({'sample_id':sid,'error':str(e)})
                duration=texts[sid][2]['duration']
                media[sid]=dict(audio=np.zeros((50,74),np.float32),vision=np.zeros((50,35),np.float32),mask_audio=np.zeros(50,np.float32),mask_vision=np.zeros(50,np.float32),time_edges=np.linspace(0,duration,51).astype(np.float32),duration=duration,n_audio_frames=0,n_visual_frames=0,face_detection_rate=0.)
            print(f'MEDIA {len(media)}/100 {sid}',flush=True)
    rows=[]
    for r in labels.itertuples(index=False):
        sid=f'{r.video_id}__{r.clip_id}';d=media[sid];text,tm,al=texts[sid]
        np.savez_compressed(C.P1_DIR/f'{sid}.npz',text=text,audio=d['audio'],vision=d['vision'],mask_text=tm,mask_audio=d['mask_audio'],mask_vision=d['mask_vision'],time_edges=d['time_edges'],meta=np.array([d['duration'],d['n_audio_frames'],d['n_visual_frames'],len(al['words']),float(r.label),d['face_detection_rate']]),schema_version=np.array('physical_time_bins_v2'))
        rows.append(dict(sample_id=sid.replace('__',C.SEP),video_id=r.video_id,clip_id=r.clip_id,label=float(r.label),annotation=r.annotation,text=str(r.text),duration_s=d['duration'],time_bin_s=d['duration']/50,text_valid=int(tm.sum()),audio_valid=int(d['mask_audio'].sum()),vision_valid=int(d['mask_vision'].sum()),face_detection_rate=d['face_detection_rate'],alignment_confidence=al['mean_confidence'],transcript_similarity=al['transcript_similarity'],n_words=len(al['words']),processing_status='failed_media_preserved' if any(f['sample_id']==sid for f in fail) else 'ok'))
    pd.DataFrame(rows).to_csv(C.DATA_DIR/'p1_summary_v2.csv',index=False,encoding='utf-8-sig')
    (C.RESULT_DIR/'p1_processing_failures.json').write_text(json.dumps(fail,ensure_ascii=False,indent=2),encoding='utf-8')
    print('COMPLETE',len(rows),'failures',len(fail),flush=True)

if __name__=='__main__':main()
