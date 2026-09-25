"""Download only required training assets from the project's private GitHub release.
Uses GITHUB_TOKEN/TOKEN or an already-configured Git credential helper.
No secrets are printed or stored. ZIPs are checked before any destination write.
"""
from __future__ import annotations
import argparse,hashlib,os,shutil,subprocess,tempfile,urllib.request,urllib.parse,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSETS={
 'tiny':(586384682,'ccb8a51416d0cef15636628ad169a7c21a7a0fdabe6bbfda75b96b576b6e839f','models'),
 'cache':(586384865,'7fb55ccd4e1c3698baf008b00606cdaddf12d261d108a08b9e5423ad8f22ae24','.'),
 'aligned':(587311383,'e34ed0182123665f562400e68ca26a800616b38b2749086402cb7397a838500d','E题'),
}
class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        follow=super().redirect_request(req,fp,code,msg,headers,newurl)
        if follow is not None and urllib.parse.urlsplit(req.full_url).netloc!=urllib.parse.urlsplit(newurl).netloc:
            follow.remove_header('Authorization')
        return follow

def token():
    value=os.environ.get('GITHUB_TOKEN') or os.environ.get('TOKEN')
    if value:return value
    env={**os.environ,'GIT_TERMINAL_PROMPT':'0'}
    r=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',text=True,capture_output=True,env=env)
    credentials=dict(line.split('=',1) for line in r.stdout.splitlines() if '=' in line)
    if not credentials.get('password'):raise RuntimeError('GitHub authentication unavailable; configure Git or set GITHUB_TOKEN securely')
    return credentials['password']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset',choices=ASSETS,nargs='+',default=['tiny','cache']);ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--replace',action='store_true');args=ap.parse_args()
    auth=token();opener=urllib.request.build_opener(SafeRedirect())
    for name in args.asset:
        asset_id,digest,destination=ASSETS[name]
        with tempfile.TemporaryDirectory(prefix='math-e-download-') as tmp:
            archive=Path(tmp)/'asset.zip'
            request=urllib.request.Request(f'https://api.github.com/repos/DaiShuHeng/math-model-e/releases/assets/{asset_id}',headers={'Authorization':f'Bearer {auth}','Accept':'application/octet-stream','User-Agent':'math-model-e-training'})
            print(f'Downloading {name} ...',flush=True)
            with opener.open(request,timeout=120) as src,archive.open('wb') as dst:shutil.copyfileobj(src,dst)
            with archive.open('rb') as f:
                h=hashlib.sha256()
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            if h.hexdigest()!=digest:raise ValueError(f'{name}: SHA256 mismatch')
            target=(args.root/destination).resolve()
            with zipfile.ZipFile(archive) as z:
                if z.testzip() is not None:raise ValueError(f'{name}: ZIP CRC failure')
                targets=[]
                for info in z.infolist():
                    if info.is_dir():continue
                    path=(target/info.filename).resolve()
                    if not path.is_relative_to(target):raise ValueError('Unsafe ZIP path')
                    mode=(info.external_attr>>16)&0o170000
                    if mode==0o120000:raise ValueError('ZIP symbolic links are not accepted')
                    if path.exists() and not args.replace:raise FileExistsError(f'{path}; verify and use --replace deliberately')
                    targets.append((info,path))
                for info,path in targets:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as dst:
                        staged=Path(dst.name)
                        try:
                            with z.open(info) as src:shutil.copyfileobj(src,dst)
                        except BaseException:
                            staged.unlink(missing_ok=True);raise
                    staged.replace(path)
            print(f'{name}: verified and installed {len(targets)} files',flush=True)

if __name__=='__main__':main()
