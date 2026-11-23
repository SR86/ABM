"""Create bounded ZIP assets plus a hash manifest for the complete local archive."""
from pathlib import Path
import hashlib
import json
import zipfile
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'abm-github-export'
SKIP = {'.git'}

def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(x in SKIP for x in p.relative_to(ROOT).parts))
    secrets = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    rows=[]
    for p in files:
        if p.name in ('gotify.local.json','.env'):
            raise RuntimeError('Credential file requires exclusion: '+str(p.relative_to(ROOT)))
        if p.suffix in ('.py','.json','.md','.txt','.log','.cfg','.yaml','.yml') and p.stat().st_size<5_000_000:
            if secrets.search(p.read_bytes()):
                raise RuntimeError('Potential credential requires review: '+str(p.relative_to(ROOT)))
    part=0
    size=0
    archive=None
    try:
        for p in files:
            n=p.stat().st_size
            if archive is None or size+n>700*1024*1024:
                if archive:
                    archive.close()
                part+=1
                archive=zipfile.ZipFile(OUT/f'project-part-{part:02d}.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1)
                size=0
            digest=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
            relative=p.relative_to(ROOT).as_posix()
            archive.write(p,relative)
            rows.append({'path':relative,'bytes':n,'sha256':digest,'asset':f'project-part-{part:02d}.zip'})
            size+=n
    finally:
        if archive:
            archive.close()
    manifest={'description':'Complete project snapshot including data, checkpoints, temporary references and local Python environments; nested Git administrative directories excluded. Extract every ZIP into the same directory.', 'files':rows}
    (OUT/'FILE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    assets=[{'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sorted(OUT.glob('project-part-*.zip'))]
    (OUT/'ASSET_CHECKSUMS.json').write_text(json.dumps(assets,indent=2))
    print(json.dumps({'files':len(rows),'parts':part,'assets':assets}),flush=True)

if __name__=='__main__':
    main()
