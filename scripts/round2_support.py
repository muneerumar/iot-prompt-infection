"""Round-two-only reports/ledger; round-one reports are immutable."""
from __future__ import annotations
import csv,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs/reports/round2'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def report(name,body,sources):
    DOCS.mkdir(parents=True,exist_ok=True);ledger=DOCS/'ledger.csv'
    rows=list(csv.DictReader(ledger.open())) if ledger.exists() else []
    rows=[r for r in rows if r['report']!=name]
    def visit(value,key,source,digest):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,key+'['+json.dumps(str(k))+']',source,digest)
        elif isinstance(value,list):
            for i,v in enumerate(value):visit(v,key+f'[{i}]',source,digest)
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            rows.append({'report':name,'statement':key,'value':repr(value),'source_file':str(source.relative_to(ROOT)),
                         'json_key_or_query':key,'sha256_of_source':digest})
    for path in dict.fromkeys(Path(p).resolve() for p in sources):
        if path.suffix=='.json':visit(json.loads(path.read_text()),'$',path,sha(path))
    fields=['report','statement','value','source_file','json_key_or_query','sha256_of_source']
    with ledger.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (DOCS/name).write_text(body+'\n\nSources: '+', '.join('`'+str(Path(p).resolve().relative_to(ROOT))+'`' for p in sources)+'.\n')

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)]+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])

def commit(message,paths):
    subprocess.run(['git','add','--',*map(str,paths)],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)

def verify_round1(snapshot):
    changed=[]
    for name in ('round1_reports.json','round1_results.json'):
        for relative,digest in json.loads((Path(snapshot)/name).read_text()).items():
            path=ROOT/relative
            if not path.exists() or sha(path)!=digest:changed.append(relative)
    if changed:raise RuntimeError('round-one artifacts changed: '+str(changed))
    return True
