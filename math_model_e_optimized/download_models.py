"""Download frozen feature extractors; these are not emotion training datasets."""
import argparse,os
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',default='external_models');ap.add_argument('--endpoint',default='https://huggingface.co');args=ap.parse_args()
    os.environ['HF_ENDPOINT']=args.endpoint
    from huggingface_hub import snapshot_download
    root=Path(args.directory);root.mkdir(parents=True,exist_ok=True)
    for repo,name in [('google-bert/bert-base-uncased','bert-base-uncased'),('facebook/wav2vec2-base-960h','wav2vec2-base-960h')]:
        snapshot_download(repo,local_dir=root/name,allow_patterns=['*.json','*.txt','model.safetensors'],endpoint=args.endpoint)
if __name__=='__main__':main()
