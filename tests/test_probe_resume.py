import json,sys
from pathlib import Path

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def test_probe_resume_skips_completed_pairs_reuses_cache_and_preserves_prefixes(tmp_path,monkeypatch):
    import run_pilot,run_quarantine_probe,recompute_run
    from iotpi.runlog import Run,read_jsonl
    from iotpi.backends import MockBackend
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    cfg['pilot'].update(episodes_per_cell=0,benign_episodes=2)
    cfg['pilot']['adaptive']['enabled']=False
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    pilot=run_pilot.main(['--config',str(config),'--backend','mock','--screen','heuristic','--conditions','trustgate'])
    record_episode=Run.record_episode;written=[]
    def interrupt(self,episode,trace):
        if len(written)==2:raise RuntimeError('probe interrupted after caching next row')
        written.append((episode['plan_index'],episode['condition']))
        return record_episode(self,episode,trace)
    monkeypatch.setattr(Run,'record_episode',interrupt)
    with pytest.raises(RuntimeError,match='probe interrupted'):
        run_quarantine_probe.main(['--pilot',str(pilot)])
    probe=next(p for p in (tmp_path/'runs').iterdir() if 'quarantine_probe' in p.name)
    prefixes={name:(probe/name).read_bytes() for name in ('episodes.jsonl','traces.jsonl','transactions.jsonl','llm_cache.jsonl','config.yaml','meta.json','pip_freeze.txt')}
    monkeypatch.setattr(Run,'record_episode',record_episode)
    generate=MockBackend.generate;new_calls=[]
    def counted(self,*args,**kwargs):new_calls.append(kwargs['seed']);return generate(self,*args,**kwargs)
    monkeypatch.setattr(MockBackend,'generate',counted)
    run_episode=run_quarantine_probe.run_episode;resumed=[]
    def measured(*args,**kwargs):
        before=len(new_calls);result=run_episode(*args,**kwargs)
        resumed.append({'model_misses':len(new_calls)-before,'cache_hits':result['cost']['cache_hits']})
        return result
    monkeypatch.setattr(run_quarantine_probe,'run_episode',measured)
    assert run_quarantine_probe.main(['--resume',str(probe)])==probe
    assert len(resumed)==2 and resumed[0]['model_misses']==0 and resumed[0]['cache_hits']>0
    rows=read_jsonl(probe/'episodes.jsonl');keys=[(r['plan_index'],r['condition']) for r in rows]
    assert len(rows)==len(set(keys))==4
    assert set(keys)=={(index,name) for index in {r['plan_index'] for r in rows}
                       for name in ('trustgate_control','trustgate_false_quarantine')}
    for name,prefix in prefixes.items():assert (probe/name).read_bytes().startswith(prefix)
    for name in ('config.yaml','meta.json','pip_freeze.txt'):assert (probe/name).read_bytes()==prefixes[name]
    assert recompute_run.main(['--run',str(probe)])==0
    completed_prefix=(probe/'episodes.jsonl').read_bytes();misses=len(new_calls)
    assert run_quarantine_probe.main(['--resume',str(probe)])==probe
    assert len(resumed)==2 and len(new_calls)==misses
    assert (probe/'episodes.jsonl').read_bytes()==completed_prefix


def test_probe_resume_repairs_committed_episode_before_projection(tmp_path,monkeypatch):
    import run_pilot,run_quarantine_probe
    from iotpi.runlog import Run,read_jsonl
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    cfg['pilot'].update(episodes_per_cell=0,benign_episodes=1)
    cfg['pilot']['adaptive']['enabled']=False
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    pilot=run_pilot.main(['--config',str(config),'--backend','mock','--screen','heuristic','--conditions','trustgate'])
    record=Run.record
    def interrupted(self,stream,row):
        if stream=='episodes':raise RuntimeError('projection interruption')
        return record(self,stream,row)
    monkeypatch.setattr(Run,'record',interrupted)
    with pytest.raises(RuntimeError,match='projection interruption'):run_quarantine_probe.main(['--pilot',str(pilot)])
    monkeypatch.setattr(Run,'record',record)
    probe=next(p for p in (tmp_path/'runs').iterdir() if 'quarantine_probe' in p.name)
    assert len(read_jsonl(probe/'transactions.jsonl'))==1
    assert run_quarantine_probe.main(['--resume',str(probe)])==probe
    assert len(read_jsonl(probe/'episodes.jsonl'))==len(read_jsonl(probe/'traces.jsonl'))==2
