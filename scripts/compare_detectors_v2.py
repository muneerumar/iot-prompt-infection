"""Compare the frozen candidates sequentially on calibration/train messages only."""
from __future__ import annotations
import argparse,gc,json,sys,hashlib
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iotpi.attacks import load_library
from iotpi.backends import make_backend
from iotpi.detectors import make_detector
from iotpi.detector_selection import build_corpus,summarise_scores,select_detector,digest
from iotpi.messages import Message
from iotpi.runlog import Run,read_jsonl

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--access',required=True)
    a=p.parse_args(argv)
    rule_path=ROOT/'configs/detector_selection_v2.json';rule=json.loads(rule_path.read_text())
    source=Path(a.source).resolve()
    if source != (ROOT/rule['corpus_source']).resolve():raise ValueError('corpus differs from preregistered source')
    if not (source/'metrics.json').exists():raise ValueError('calibration has not finished')
    cal=json.loads((source/'calibration.json').read_text())
    if cal['backend']!='mlx' or cal['model']!='llama-4bit':raise ValueError('real Llama calibration required')
    cfg=yaml.safe_load((source/'config.yaml').read_text())
    lib=load_library(ROOT,cfg['attacks'])
    corpus=build_corpus(read_jsonl(source/'generations.jsonl'),cfg,set(lib['split']['train']))
    run=Run(ROOT,'detector_selection_v2',{'rule':rule,'seed':cfg['seed'],'source':str(source),
            'source_sha256':hashlib.sha256((source/'generations.jsonl').read_bytes()).hexdigest(),
            'rule_sha256':hashlib.sha256(rule_path.read_bytes()).hexdigest(), 'corpus_hash':digest(corpus)})
    for row in corpus:run.record('corpus',row)
    run.save_json('corpus_manifest.json',{'hash':digest(corpus),'events':len(corpus),
        'buckets':{b:sum(r['bucket']==b for r in corpus) for b in ('raw','rewritten','benign')},
        'training_families':lib['split']['train'], 'contexts':sorted({r['context'] for r in corpus})})
    results={};backend=None;detector=None
    try:
        for spec in rule['candidates']:
            run.log('candidate '+spec)
            if 'Prompt-Guard' in spec:
                access=json.loads(Path(a.access).read_text())
                if access['status']!='available':
                    results[spec]={**access,'status':'unavailable'};continue
            backend=None;detector=None
            try:
                if spec=='judge':backend=make_backend('mlx',str((ROOT/cfg['models']['llama-4bit']).resolve()))
                detector=make_detector(spec,rule['threshold'],backend,ROOT,cfg['detectors']['device'])
                warm=corpus[0]
                detector.flags(Message('warmup','source','receiver',warm['kind'],warm['text'],0))
                rows=[]
                for i,event in enumerate(corpus):
                    # Only ordinary message fields reach the candidate; evaluator labels stay here.
                    flag,score=detector.flags(Message(event['event_id'],'source','receiver',event['kind'],event['text'],0))
                    row={k:v for k,v in event.items() if k!='text'}
                    row.update(detector=spec,flagged=flag,score=score,seconds=detector.observations[-1]['seconds'],
                               scoring=detector.observations[-1])
                    run.record('scores',row);rows.append(row)
                    if i%50==0:run.log(f'{spec} {i+1}/{len(corpus)}')
                model_revision=(getattr(getattr(detector,'pipe',None),'model',None))
                model_revision=getattr(getattr(model_revision,'config',None),'_commit_hash',None)
                results[spec]={'status':'available','summary':summarise_scores(rows),'revision':model_revision,
                               'shared_agents_model':spec=='judge',
                               'judge_prompt_sha256':hashlib.sha256((ROOT/'configs/prompts/judge.txt').read_bytes()).hexdigest() if spec=='judge' else None}
            except (MemoryError,KeyboardInterrupt):raise
            except Exception as error:
                # A partial scoring failure is ineligible, never silently a successful comparison.
                results[spec]={'status':'unavailable','error_type':type(error).__name__}
                run.record('failures',{'detector':spec,'error_type':type(error).__name__})
            finally:
                detector=None
                if backend:backend.close()
                backend=None;gc.collect()
        mandatory=[name for name in rule['candidates'] if 'Prompt-Guard' not in name]
        comparison_complete=all(results[name]['status']=='available' for name in mandatory)
        chosen=select_detector(results,rule) if comparison_complete else None
        out={'rule':rule,'candidates':results,'selected':chosen,'threshold':.5,'corpus_hash':digest(corpus),
             'comparison_complete':comparison_complete,
             'judge_caveat':'The judge shares the agents Llama model; detector and agent failures may be correlated.'}
        run.save_json('selection.json',out);run.save_json('metrics.json',out)
        run.log('selected '+str(chosen))
        return 0 if chosen else 2
    finally:
        if backend:backend.close()
        run.close()
if __name__=='__main__':sys.exit(main())
