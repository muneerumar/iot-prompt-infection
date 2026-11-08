"""Resumable round-two benign development/timing; confirmation is explicitly gated.

Only health metrics are logged during a confirmatory run. It writes no comparison
summary; the pre-registered analysis is run only after both model runs finish.
"""
from __future__ import annotations
import argparse,json,sys,hashlib,shutil,time,subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.attacks import load_library,adaptive_rewrite
from iotpi.backends import make_backend,CachingBackend
from iotpi.conditions import conditions
from iotpi.detectors import make_detector
from iotpi.experiment import load_prompts,llm_agents,risk_weights,run_condition
from iotpi.round2 import CONDITIONS,benign_plan,confirmatory_plan,build_round2_scenario
from iotpi.scenario import set_attack_text
from iotpi.attacks import AttackSpec
from iotpi.runlog import Run,read_jsonl
from iotpi.utility_audit import audit_work

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/round2.yaml'))
    p.add_argument('--model',choices=['llama-4bit','qwen-4bit']);p.add_argument('--backend',choices=['mlx','mock'],default='mlx')
    p.add_argument('--mode',choices=['benign','timing','confirm'],default='benign');p.add_argument('--iteration',type=int,default=0)
    p.add_argument('--resume');p.add_argument('--name',default='round2');p.add_argument('--confirmatory-authorized',action='store_true')
    a=p.parse_args(argv)
    if a.resume:
        run_dir=Path(a.resume);cfg=yaml.safe_load((run_dir/'config.yaml').read_text());res=cfg['resolved']
        if res['mode']=='confirm' and not a.confirmatory_authorized:raise PermissionError('explicit user go round 2 is required')
        run=Run.reopen(run_dir)
    else:
        if not a.model:raise ValueError('--model required')
        if a.mode=='confirm' and not a.confirmatory_authorized:raise PermissionError('STOP after preregistration; explicit user go round 2 required')
        cfg=yaml.safe_load(Path(a.config).read_text());res={'mode':a.mode,'model_key':a.model,'backend':a.backend,
            'iteration':a.iteration,'model_path':cfg['models'][a.model]}
        if a.mode=='confirm':
            if subprocess.run(['git','rev-parse','--verify','prereg-round2'],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
                raise PermissionError('prereg-round2 is missing')
        res.update(risk_rates_source=cfg['round2']['risk_rates'],risk_rates_sha256=sha(ROOT/cfg['round2']['risk_rates']),risk_rates_usage='fixed round-one Llama design constant for both models')
        cfg['resolved']=res;run=Run(ROOT,f'{a.name}_{a.mode}_{a.model}',cfg)
        (run.dir/'prompts').mkdir()
        for source in (ROOT/'configs/prompts').glob('*.txt'):shutil.copyfile(source,run.dir/'prompts'/source.name)
        run.save_json('input_hashes.json',{'prompts':{p.name:sha(p) for p in (run.dir/'prompts').glob('*.txt')},
            'attacks':sha(ROOT/cfg['attacks']),'risk_constant':sha(ROOT/cfg['round2']['risk_rates']),'config_sha256':sha(run.dir/'config.yaml')})
    hashes=json.loads((run.dir/'input_hashes.json').read_text())
    if hashes.get('config_sha256')!=sha(run.dir/'config.yaml') or any(sha(run.dir/'prompts'/name)!=digest for name,digest in hashes['prompts'].items()):raise ValueError('saved config or prompt snapshot changed')
    if hashes['attacks']!=sha(ROOT/cfg['attacks']) or hashes['risk_constant']!=sha(ROOT/cfg['round2']['risk_rates']):
        raise ValueError('fixed round-two inputs changed')
    run.recover_episodes();rows=read_jsonl(run.dir/'episodes.jsonl');done={(r['plan_index'],r['condition']) for r in rows}
    if len(done)!=len(rows):raise ValueError('duplicate round-two rows')
    lib=load_library(ROOT,cfg['attacks']);mode=res['mode']
    plan=confirmatory_plan(cfg,lib) if mode=='confirm' else benign_plan(cfg,mode=='timing')
    if mode!='confirm' and any(not 0<=r['context']<10 or (mode=='benign' and r['family'] is not None) or (mode=='timing' and r['family'] is not None and r['family'] not in lib['split']['train']) for r in plan):
        raise ValueError('development is benign; timing uses calibration buildings and training families only')
    names=CONDITIONS if mode in ('timing','confirm') else ['policy_only']
    expected={(i,name) for i in range(len(plan)) for name in names}
    if not done<=expected:raise ValueError('unexpected saved row')
    if done==expected and (run.dir/'metrics.json').exists():run.close();return run.dir
    prompts={p.stem:p.read_text() for p in (run.dir/'prompts').glob('*.txt')}
    rates=json.loads((ROOT/cfg['round2']['risk_rates']).read_text())
    backend=None;started=time.monotonic()
    try:
        backend=make_backend(res['backend'],str((ROOT/res['model_path']).resolve()) if res['backend']=='mlx' else None)
        backend=CachingBackend(backend,run.dir/'llm_cache.jsonl')
        det=cfg['detectors'];conds=conditions(cfg['pilot']['budget_per_node'],cfg['pilot']['cache_distance'])
        screen=(make_detector(det['screen'],det['screen_threshold'],backend,ROOT,det['device'])
                if any(conds[n].screen!='none' for n in names) else None)
        cheap=make_detector(det['cheap'],.5)
        saved={r['plan_index']:r for r in read_jsonl(run.dir/'adaptive_attacks.jsonl')}
        for i,item in enumerate(plan):
            if all((i,n) in done for n in names):continue
            sc=build_round2_scenario(cfg,lib if item['family'] else None,item)
            if item.get('adaptive'):
                if i in saved:
                    from dataclasses import replace
                    spec=replace(sc.attack,text=saved[i]['text'],adaptive_queries=saved[i]['queries'],adaptive_score=saved[i]['score'],adaptive_cost=saved[i]['cost'])
                else:
                    spec=adaptive_rewrite(sc.attack,lib,backend,screen,cfg['pilot']['adaptive']['queries'],det['screen_threshold'],cfg['seed']+i)
                    run.record('adaptive_attacks',{'plan_index':i,'text':spec.text,'queries':spec.adaptive_queries,'score':spec.adaptive_score,'cost':spec.adaptive_cost})
                set_attack_text(sc,spec)
            graph=__import__('iotpi.topologies',fromlist=['pilot']).pilot(sc.zones)
            weights=risk_weights(graph,rates['training_rates'],cfg['risk']['horizon_hops'])
            for name in names:
                if (i,name) in done:continue
                result=run_condition(sc,graph,llm_agents(graph,backend,prompts,cfg['generation']),conds[name],screen,cheap,weights,
                    seed=cfg['seed']*7919+i,reset_every=cfg['world']['reset_every'],keep_prompts=True)
                trace=result.pop('trace')
                episode={'plan_index':i,'condition':name,'split':item['stratum'],**item,**result}
                run.record_episode(episode,{'plan_index':i,'condition':name,'trace':trace,'context_id':sc.context_id,
                    'scenario_seed':sc.seed,'condition_seed':cfg['seed']*7919+i,'attack_text':sc.attack.text if sc.attack else None})
                if mode=='benign':run.record('utility_audit',{'plan_index':i,'condition':name,**audit_work(sc,episode,trace)})
                done.add((i,name));rows.append(episode)
                run.log(f'health rows={len(done)}/{len(expected)} parse_fail={result["utility"]["parse_fail"]} budget_screens={result["cost"].get("screens",0)}')
        calls=[r for row in read_jsonl(run.dir/'traces.jsonl') for r in row['trace'] if 'parse_ok' in r]
        health={'complete':done==expected,'rows':len(done),'expected_rows':len(expected),'model':res['model_key'],
            'mode':mode,'parse_ok':sum(r['parse_ok'] for r in calls),'parse_n':len(calls),
            'parse_rate':sum(r['parse_ok'] for r in calls)/len(calls), 'elapsed_this_process_seconds':time.monotonic()-started,
            'cache_hits':sum(r['cost'].get('cache_hits',0) for r in rows),'logical_agent_calls':len(calls),
            'minimum_available_bytes':min((r.get('generation_meta',{}).get('minimum_available_bytes',float('inf')) for r in calls),default=None)}
        if mode=='benign':
            nonempty=[r['utility']['done_rate'] for r in rows if r['utility']['expected']]
            health.update(mean_benign_completion=sum(nonempty)/len(nonempty) if nonempty else None,
                nonempty_episodes=len(nonempty),expected_commands=sum(r['utility']['expected'] for r in rows),completed_commands=sum(r['utility']['done'] for r in rows))
        run.save_json('metrics.json',health);run.log('complete: '+str(run.dir));return run.dir
    finally:
        if backend:backend.close()
        run.close()
if __name__=='__main__':main()
