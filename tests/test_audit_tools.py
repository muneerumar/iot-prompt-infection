import json,sys
from pathlib import Path
import yaml
from iotpi.runlog import read_jsonl
from iotpi.experiment import load_prompts

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))

def test_raw_calibration_and_prediction_recompute(tmp_path,monkeypatch):
    import run_calibration,run_pilot,predict_multihop,recompute_run
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    cal=run_calibration.main(['--backend','mock','--screen','heuristic','--units','4'])
    pilot=run_pilot.main(['--backend','mock','--screen','heuristic','--limit','3','--no-adaptive','--conditions','policy_only','trustgate'])
    pred=predict_multihop.main(['--calibration',str(cal/'calibration.json'),'--pilot',str(pilot),'--draws','2','--reps','2'])
    for directory in [cal,pilot,pred]:assert recompute_run.main(['--run',str(directory)])==0

def test_supplemental_quarantine_probe_is_paired_and_benign(tmp_path,monkeypatch):
    import run_pilot,run_quarantine_probe,recompute_run
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    cfg['pilot'].update(episodes_per_cell=0,benign_episodes=2)
    cfg['pilot']['adaptive']['enabled']=False
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(cfg))
    pilot=run_pilot.main(['--config',str(config),'--backend','mock','--screen','heuristic','--conditions','trustgate'])
    probe=run_quarantine_probe.main(['--pilot',str(pilot)])
    rows=read_jsonl(probe/'episodes.jsonl')
    assert len(rows)==4 and all(r['attack'] is None for r in rows)
    for index in {r['plan_index'] for r in rows}:
        pair=[r for r in rows if r['plan_index']==index]
        assert pair[0]['utility']['expected']==pair[1]['utility']['expected']
    forced=[r for r in rows if r['condition']=='trustgate_false_quarantine']
    assert all(r['utility']['false_quarantines']>=1 and r['utility']['fallback_digests']>0 for r in forced)
    assert recompute_run.main(['--run',str(probe)])==0

def test_pilot_validation_detects_whole_missing_episodes(tmp_path,monkeypatch):
    import run_pilot
    from workflow_support import analyse_pilot
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    directory=run_pilot.main(['--backend','mock','--screen','heuristic','--limit','2','--no-adaptive','--conditions','policy_only'])
    assert analyse_pilot(directory)['expected_rows']==2
    rows=read_jsonl(directory/'episodes.jsonl')
    (directory/'episodes.jsonl').write_text(json.dumps(rows[0])+'\n')
    result=analyse_pilot(directory)
    assert result['rows']==1 and result['expected_rows']==2 and result['missing_plan_indices']==[1]

def test_gate_helpers_are_conservative():
    from workflow_support import gate_one,gate_two
    summary={'conditions':{name:{'harm':[v,None,None],'legit_done_rate':[.8,None,None]} for name,v in [('trustgate',.1),('classifier_cache',.2),('classifier_trust',.3)]},
             'paired':{name:{'p_holm':.01} for name in ['classifier_cache','classifier_trust']}}
    assert gate_one(summary)['verdict']=='pass'
    summary['conditions']['trustgate']['legit_done_rate'][0]=.79
    assert gate_one(summary)['verdict']=='fail'
    assert gate_two({})['verdict']=='not reached'


def test_quarantine_gate_cannot_pass_when_no_work_is_completed(tmp_path):
    from workflow_support import gate_three
    rows=[]
    for name in ['trustgate_control','trustgate_false_quarantine']:
        rows.append({'plan_index':0,'condition':name,'context_id':'ctx100',
                     'utility':{'expected':1,'done':0,'done_rate':0.,'fallback_digests':2}})
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert gate_three(tmp_path)['verdict']=='fail'
