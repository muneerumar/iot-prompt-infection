"""Supplemental paired benign probe for the false-quarantine decision gate.

The main pilot is unchanged. Both arms use the frozen TrustGate configuration; one
receives an injected false-positive quarantine before the first step.
"""
from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iotpi.backends import CachingBackend,make_backend
from iotpi.conditions import conditions
from iotpi.detectors import make_detector
from iotpi.engine import run_episode
from iotpi.experiment import build_scenario,llm_agents,load_prompts,risk_weights
from iotpi.guard import GuardNetwork
from iotpi.report import load_rows,markdown,summarise
from iotpi.runlog import Run
from iotpi.topologies import pilot

def main(argv=None):
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pilot')
    source.add_argument('--resume',help='append missing (plan_index, condition) rows using the saved config and llm_cache.jsonl')
    parser.add_argument('--name',default='quarantine_probe_llama')
    a=parser.parse_args(argv)
    if a.resume:
        saved=yaml.safe_load((Path(a.resume)/'config.yaml').read_text())
        pilot_dir=Path(saved['pilot']);cfg=saved['pilot_config']
    else:
        pilot_dir=Path(a.pilot)
        cfg=yaml.safe_load((pilot_dir/'config.yaml').read_text())
    res=cfg['resolved']
    items={r['plan_index']:r for r in load_rows(pilot_dir) if r['attack'] is None}
    if not items: raise ValueError('no frozen benign pilot episodes for the probe')
    run=(Run.reopen(Path(a.resume)) if a.resume else Run(ROOT,a.name,{'pilot':str(pilot_dir),'pilot_config':cfg,
                        'intervention':'false-positive quarantine of first gateway for configured quarantine_steps'}))
    run.recover_episodes()
    rows=load_rows(run.dir)
    keys=[(row['plan_index'],row['condition']) for row in rows]
    done=set(keys)
    expected={(index,name) for index in items for name in ('trustgate_control','trustgate_false_quarantine')}
    if len(keys)!=len(done) or not done<=expected:
        run.close();raise ValueError('duplicate or unexpected probe row')
    def finish():
        summary=summarise(rows,reference='trustgate_control')
        for filename in ('summary.json','metrics.json'):
            path=run.dir/filename
            if path.exists():
                if json.dumps(json.loads(path.read_text()),sort_keys=True)!=json.dumps(summary,sort_keys=True):
                    raise ValueError('existing probe summary differs from raw rows')
            else:run.save_json(filename,summary)
        rendered=markdown(summary);path=run.dir/'summary.md'
        if path.exists():
            if path.read_text()!=rendered:raise ValueError('existing probe Markdown differs')
        else:path.write_text(rendered)
    if done==expected:
        try:finish()
        finally:run.close()
        return run.dir
    if a.resume:run.log(f'resuming probe: {len(done)}/{len(expected)} rows complete; reusing llm_cache.jsonl')
    backend=make_backend(res['backend'],str((ROOT/res['model_path']).resolve()) if res['backend']=='mlx' else None)
    backend=CachingBackend(backend,run.dir/'llm_cache.jsonl',enabled=not res.get('disable_cache',False))
    det=cfg['detectors']
    screen=make_detector(det['screen'],det['screen_threshold'],backend,ROOT,det['device'])
    cheap=make_detector(det['cheap'],.5)
    guard_config=conditions(cfg['pilot']['budget_per_node'],cfg['pilot'].get('cache_distance'))['trustgate']
    prompts=load_prompts(ROOT)
    try:
        for index,item in sorted(items.items()):
            scenario=build_scenario(cfg,None,item)
            graph=pilot(scenario.zones)
            weights=risk_weights(graph,res['risk_rates'],cfg['risk']['horizon_hops'])
            for forced in [False,True]:
                name='trustgate_false_quarantine' if forced else 'trustgate_control'
                if (index,name) in done:continue
                screen.calls,screen.seconds,screen.observations=0,0.0,[]
                cheap.calls,cheap.seconds,cheap.observations=0,0.0,[]
                guard=GuardNetwork(copy.deepcopy(guard_config),graph,screen,cheap,weights,seed=cfg['seed']*7919+index)
                if forced:
                    node=graph.gateway(scenario.zones[0])
                    guard.quarantined[node]=guard_config.quarantine_steps
                    guard.events.append(('quarantine',node,0,'injected_false_positive'))
                agents=llm_agents(graph,backend,prompts,cfg['generation'],guard_config.prompt_defence)
                result=run_episode(scenario,graph,agents,guard,seed=cfg['seed']*7919+index,
                                   reset_every=cfg['world'].get('reset_every'),keep_prompts=True)
                trace=result.pop('trace')
                row={'plan_index':index,'condition':name,'split':'benign_probe',**item,**result}
                row['condition']=name
                row['split']='benign_probe'
                rows.append(row)
                run.record_episode(row,{'plan_index':index,'condition':name,'trace':trace,'attack_text':None})
                done.add((index,name))
                run.log(f'benign plan={index} condition={name} done={result["utility"]["done"]}/{result["utility"]["expected"]}')
        if done!=expected:raise RuntimeError('probe did not complete every frozen pair')
        finish()
    finally:
        backend.close();run.close()
    return run.dir

if __name__=='__main__': print(main(),flush=True)
