import copy,sys
from pathlib import Path
from collections import Counter
import pytest,yaml
from iotpi.round2 import confirmatory_plan,benign_plan,build_round2_scenario,CONDITIONS
from iotpi.round2_analysis import model_primary,both_models_primary,paired_utility_bound
from iotpi.attacks import load_library
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))

def test_round2_plan_counts_fresh_buildings_and_target_balance_without_execution():
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text());lib=load_library(ROOT,cfg['attacks'])
    for n,total in ((40,320),(30,260)):
        cfg['round2']['c3_main_per_family']=n;plan=confirmatory_plan(cfg,lib)
        assert len(plan)==total and set(r['context'] for r in plan)==set(range(200,240))
        assert Counter(r['stratum'] for r in plan)=={'c3_main':6*n,'sanity':30,'adaptive_c3':20,'benign':30}
        for family in lib['families']:
            rows=[r for r in plan if r['family']==family and r['stratum']=='c3_main']
            assert [r['c3_variant'] for r in rows]==[i%2 for i in range(n)]
    assert all(0<=r['context']<10 and r['family'] is None for r in benign_plan(cfg))

def test_round2_alternation_renders_matching_targets_on_calibration_building_only():
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text());lib=load_library(ROOT,cfg['attacks'])
    for i,(kind,command) in enumerate([('smoke','maintenance_mode'),('ventilation','off')]):
        sc=build_round2_scenario(cfg,lib,{'context':0,'episode':987,'family':'authority','goal':'c3_safety','c3_variant':i})
        assert sc.attack.target[0].startswith(kind+'_') and sc.attack.target[1]==command
        assert sc.attack.target[0] in sc.attack.text
        assert all(e.device!=sc.attack.target[0] for e in sc.expected)

def test_confirmation_cannot_start_without_explicit_authorization(tmp_path,monkeypatch):
    import run_round2
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path))
    with pytest.raises(PermissionError):run_round2.main(['--mode','confirm','--model','llama-4bit','--backend','mock'])
    assert not list(tmp_path.iterdir())

def test_bootstrap_lower_bound_and_strict_margin():
    b=paired_utility_bound([-.02]*8,['a']*4+['b']*4)
    assert b['lower_one_sided_95']==pytest.approx(-.02) and b['bootstrap_reps']==10000
    b=paired_utility_bound([-.1,.1],['a','b'])
    assert b['lower_one_sided_95']==pytest.approx(-.1)

def primary_fixture(utility_loss=.02):
    rows=[]
    for i in range(16):
        for name in ('trustgate','classifier_cache','classifier_trust'):
            rows.append({'plan_index':i,'condition':name,'stratum':'c3_main','context_id':'building'+str(i%4),
                'attack':{'executed':name!='trustgate' and i<8},
                'utility':{'expected':1,'done_rate':1-utility_loss if name=='trustgate' else 1}})
    return rows

def test_primary_holm_family_only_two_and_both_models_required():
    rows=primary_fixture();r=model_primary(rows)
    assert r['wins'] and set(r['H1'])=={'H1a','H1b'}
    assert r['H1']['H1a']['p_holm']==pytest.approx(.015625)
    assert both_models_primary({'llama-4bit':rows,'qwen-4bit':rows})['cross_model_win']
    with pytest.raises(ValueError):both_models_primary({'llama-4bit':rows})
    assert not model_primary(primary_fixture(.03))['wins']

def test_development_runner_is_benign_resumable_and_uses_fixed_llama_rates_for_qwen(tmp_path,monkeypatch):
    import run_round2
    from iotpi.runlog import read_jsonl
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text());cfg['round2']['development']['benign_episodes']=2
    cfg['detectors']['screen']='heuristic'
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(cfg));monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path/'runs'))
    run=run_round2.main(['--config',str(path),'--model','qwen-4bit','--backend','mock'])
    rows=read_jsonl(run/'episodes.jsonl');assert len(rows)==2 and all(r['attack'] is None and r['context_id'] in ('ctx000','ctx001') for r in rows)
    prefixes=(run/'episodes.jsonl').read_bytes();assert run_round2.main(['--resume',str(run)])==run
    assert (run/'episodes.jsonl').read_bytes()==prefixes
    assert (run/'prompts/role_gateway.txt').exists()


def test_prompt_candidates_render_and_timing_protocol_are_fixed_before_runs():
    from develop_round2 import prompt_variants,preregistration
    from iotpi.agents import LLMAgent
    from iotpi.backends import MockBackend
    from iotpi.scenario import make_scenario
    from iotpi.topologies import pilot
    from iotpi.calibrate import _gateway_inbox
    from iotpi.experiment import load_prompts
    original={name:(ROOT/'configs/prompts'/f'{name}.txt').read_text() for name in ['role_gateway','task_gateway','role_coordinator','task_coordinator']}
    versions=prompt_variants(original);assert len(versions)==3
    sc=make_scenario(0,1000000,2,4);graph=pilot(sc.zones)
    for version in versions:
        prompts={**load_prompts(ROOT),**version}
        for node in graph.agents:
            role=graph.role[node];agent=LLMAgent(node,role,MockBackend(),prompts,{'temperature':0.,'max_tokens':300})
            inbox=_gateway_inbox(sc,graph.zone_of[node],0) if role=='gateway' else []
            assert len(agent.build_messages(inbox,'',0,sc,graph))==2
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text())
    timing=benign_plan(cfg,True)
    assert len(timing)==2 and [r['context'] for r in timing]==[0,1]
    assert timing[0]['family'] is None and timing[1]['family']=='authority'
    text=preregistration(cfg)
    assert 'Holm across {H1a, H1b} only' in text and '−0.03' in text and '10,000' in text


def test_timing_rule_has_only_the_prescribed_main_sample_reduction():
    from develop_round2 import timing_estimate
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text())
    profile={'unique_fraction':.2,'mean_agent_seconds':15.,'screen_seconds_two_episodes':20.,'mean_ticket_screen_seconds':1.}
    profiles={'llama-4bit':profile,'qwen-4bit':profile}
    first=timing_estimate(profiles,cfg)
    assert first['total_hours']>40
    reduced=copy.deepcopy(cfg);reduced['round2']['c3_main_per_family']=30
    second=timing_estimate(profiles,reduced)
    assert second['total_hours']<40 and second['models']['llama-4bit']['episodes']==260
    assert reduced['pilot']==cfg['pilot'] and reduced['generation']==cfg['generation']
