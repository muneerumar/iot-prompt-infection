"""Infrastructure-only tests. Synthetic records never execute evaluation scenarios."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import round2_confirm_support as support
import confirm_round2 as controller
import analyze_round2 as analysis
from iotpi.round2 import CONDITIONS


def test_serial_order_and_no_qwen_after_llama_failure():
    events=[]
    def execute(model):
        events.append(('start',model));events.append(('exit',model));return model
    assert list(controller.serial_models(execute))==list(support.MODELS)
    assert events==[('start','llama-4bit'),('exit','llama-4bit'),('start','qwen-4bit'),('exit','qwen-4bit')]
    events=[]
    def fail(model):events.append(model);raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError):controller.serial_models(fail)
    assert events==['llama-4bit']


def test_owned_directory_recovered_after_controller_crash(tmp_path):
    p=tmp_path/'partial';p.mkdir();(p/'meta.json').write_text(json.dumps({'name':'owned'}))
    assert controller.owned_directory(tmp_path,'owned')==p
    assert controller.owned_directory(tmp_path,'other') is None
    duplicate=tmp_path/'duplicate';duplicate.mkdir();(duplicate/'meta.json').write_text(json.dumps({'name':'owned'}))
    with pytest.raises(RuntimeError):controller.owned_directory(tmp_path,'owned')


def health_fixture():
    item={'context':200,'episode':2000000,'family':'synthetic','goal':'c3_safety',
          'heldout_family':False,'stratum':'c3_main','c3_variant':0,'adaptive':False}
    rows=[{**item,'plan_index':0,'condition':c,'context_id':'ctx200','seed':2000000,
           'utility':{'parse_fail':int(c=='trustgate')},'cost':{'screen_audit':[
               {'screens':1,'credit_before':1.,'credit_after':0.}],
               'screen_call_measurements':[{'budget_units':1,'tiers':{'first':{},'second':{}}}]}}
          for c in CONDITIONS]
    return [item],rows


def test_health_retains_parse_failures_and_checks_pairs_budget_and_seeds():
    plan,rows=health_fixture();health=support.validate_rows(rows,plan,complete=True)
    assert health['rows']==6 and health['parse_failures']==1 and health['budget_violations']==0
    with pytest.raises(ValueError):support.validate_rows(rows[:-1],plan,complete=True)
    with pytest.raises(ValueError):support.validate_rows(rows+[rows[0]],plan)
    rows[0]['seed']+=1
    with pytest.raises(ValueError):support.validate_rows(rows,plan)
    rows[0]['seed']-=1;rows[0]['cost']['screen_audit'][0]['screens']=2
    assert support.validate_rows(rows,plan)['budget_violations']==1
    rows[0]['cost']['screen_call_measurements'][0]['budget_units']=2
    with pytest.raises(ValueError):support.validate_rows(rows,plan)


def test_append_only_reader_keeps_completed_records_around_crash_fragment(tmp_path):
    p=tmp_path/'units.jsonl';p.write_text('{"x":1}\n{"unfinished":\n{"x":2}\n')
    before=p.read_bytes();assert list(support.records(p))==[{'x':1},{'x':2}]
    assert before==p.read_bytes()


def test_no_analysis_before_both_completed_and_primary_persisted_before_secondary(tmp_path,monkeypatch):
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'analysis'))
    monkeypatch.setattr(analysis,'experiment_processes',lambda:[])
    monkeypatch.setattr(analysis,'verify_frozen',lambda:({'git_tag':'prereg-round2','git_commit':'fixture'},{}))
    paths=[]
    for model in support.MODELS:
        p=tmp_path/model;p.mkdir();paths.append(p)
        for name in ('episodes.jsonl','traces.jsonl','config.yaml','llm_cache.jsonl'):(p/name).write_text('')
    reached=[]
    def incomplete(*args,**kwargs):raise ValueError('incomplete')
    monkeypatch.setattr(analysis,'inspect_run',incomplete)
    monkeypatch.setattr(analysis,'both_models_primary',lambda rows:reached.append('primary') or {'fixture':True})
    with pytest.raises(ValueError):analysis.analyze(*paths)
    assert not reached and not (tmp_path/'analysis').exists()
    monkeypatch.setattr(analysis,'inspect_run',lambda *a,**k:{'rows':1920})
    def secondary(rows):
        primary=list((tmp_path/'analysis').glob('*/primary.json'))
        assert len(primary)==1 and json.loads(primary[0].read_text())=={'fixture':True}
        reached.append('secondary');return {'fixture':'descriptive'}
    monkeypatch.setattr(analysis,'secondary_tables',secondary)
    monkeypatch.setattr(analysis,'write_report',lambda *a:reached.append('report'))
    directory=analysis.analyze(*paths)
    assert reached==['primary','secondary','report']
    prefix=(directory/'primary.json').read_bytes()
    analysis.analyze(*paths,resume=directory)
    assert (directory/'primary.json').read_bytes()==prefix
    with pytest.raises(ValueError):analysis.save_once(directory/'primary.json',{'different':True})


def test_analysis_refuses_running_experiment(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'experiment_processes',lambda:['42 model process'])
    with pytest.raises(RuntimeError):analysis.analyze(tmp_path,tmp_path)


def test_descriptive_tables_include_sanity_splits_and_tier_costs():
    row={'condition':'trustgate','stratum':'sanity','family':'fixture','heldout_family':True,'goal':'exfil',
         'attack':{'executed':False,'exfil_attempted':True,'exfiltrated':False},
         'utility':{'expected':2,'done':1,'done_rate':.5,'parse_fail':1},
         'cost':{'calls':1,'screen_tier_calls':{'judge':1},'screen_tier_seconds':{'judge':2.}}}
    result=analysis.secondary_tables({'llama-4bit':[row]})['llama-4bit']
    assert result['descriptive_only']
    assert result['groups']['sanity/goal/exfil/trustgate']['exfil_attempted']==1
    assert result['groups']['sanity/split/heldout/trustgate']['mean_legitimate_completion']==.5
    assert result['cost_totals']['trustgate']['screen_tier_seconds']['judge']==2.


def test_completed_health_accepts_quarantine_steps_and_rejects_config_drift(tmp_path,monkeypatch):
    import shutil,yaml
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text());plan,rows=health_fixture()
    monkeypatch.setattr(support,'confirmatory_plan',lambda *a:plan)
    (tmp_path/'prompts').mkdir()
    for path in (ROOT/'configs/prompts').glob('*.txt'):shutil.copyfile(path,tmp_path/'prompts'/path.name)
    saved={**cfg,'resolved':{'mode':'confirm','backend':'mlx','model_key':'llama-4bit','model_path':cfg['models']['llama-4bit']}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(saved))
    hashes={'config_sha256':support.sha(tmp_path/'config.yaml'),
            'prompts':{p.name:support.sha(p) for p in (tmp_path/'prompts').glob('*.txt')},
            'attacks':support.sha(ROOT/cfg['attacks']),'risk_constant':support.sha(ROOT/cfg['round2']['risk_rates'])}
    (tmp_path/'input_hashes.json').write_text(json.dumps(hashes))
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    traces=[]
    for row in rows:
        calls=[]
        for i in range(12):
            cell={'node':['a','b','c'][i%3],'step':i//3}
            if i<2:cell.update(parse_ok=not(i==0 and row['condition']=='trustgate'),generation_meta={'minimum_available_bytes':2_000_000_000})
            else:cell['quarantined']=True
            calls.append(cell)
        traces.append({'plan_index':0,'condition':row['condition'],'trace':calls})
    (tmp_path/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in traces))
    (tmp_path/'metrics.json').write_text(json.dumps({'complete':True,'rows':6,'model':'llama-4bit','parse_n':12}))
    health=support.inspect_run(tmp_path,'llama-4bit',cfg,complete=True)
    assert health['parse_n']==12 and health['parse_failures']==1 and health['parse_ok']==11
    saved['generation']['temperature']=.5
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(saved))
    with pytest.raises(ValueError):support.inspect_run(tmp_path,'llama-4bit',cfg,complete=True)
