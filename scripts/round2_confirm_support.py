"""Phase-C infrastructure: frozen-input verification and outcome-blind health checks."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iotpi.attacks import load_library
from iotpi.round2 import CONDITIONS,confirmatory_plan
from round2_support import sha,verify_round1
MODELS=('llama-4bit','qwen-4bit')
MANIFEST=ROOT/'results/20261106-065753_round2_development/preregistration_manifest.json'


def verify_frozen():
    manifest=json.loads(MANIFEST.read_text())
    commit=subprocess.check_output(['git','rev-parse','prereg-round2'],cwd=ROOT,text=True).strip()
    if commit!=manifest['git_commit'] or sha(manifest['preregistration'])!=manifest['sha256']:
        raise RuntimeError('pre-registration tag/document changed')
    for path,digest in manifest['config_prompt_hashes'].items():
        if sha(ROOT/path)!=digest:raise RuntimeError('frozen input changed: '+path)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only','prereg-round2',
        'src/iotpi','scripts','requirements.lock','pyproject.toml'],cwd=ROOT,text=True).splitlines()
    for path in paths:
        original=subprocess.check_output(['git','show','prereg-round2:'+path],cwd=ROOT)
        if hashlib.sha256(original).hexdigest()!=sha(ROOT/path):
            raise RuntimeError('registered code changed: '+path)
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text())
    for model in MODELS:
        folder=Path(manifest['timing']['models'][model]['profile']['directory'])
        inputs=json.loads((folder/'input_hashes.json').read_text())
        if inputs['attacks']!=sha(ROOT/cfg['attacks']) or inputs['risk_constant']!=sha(ROOT/cfg['round2']['risk_rates']):
            raise RuntimeError('registered private attack library or risk constant changed')
    verify_round1(ROOT/cfg['round2']['round1_snapshot'])
    return manifest,cfg


def records(path):
    """Stream complete append-only records, preserving a possible crash fragment."""
    path=Path(path)
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                try:yield json.loads(line)
                except json.JSONDecodeError:continue


def validate_rows(rows,plan,complete=False):
    expected={(i,c) for i in range(len(plan)) for c in CONDITIONS};seen=set()
    budget_violations=0;parse_fail=0
    for row in rows:
        key=(row['plan_index'],row['condition'])
        if key not in expected or key in seen:raise ValueError('duplicate or unexpected confirmatory row')
        seen.add(key);item=plan[key[0]]
        if any(row.get(k)!=v for k,v in item.items()):raise ValueError('saved episode differs from frozen plan')
        if row['context_id']!=f"ctx{item['context']:03d}" or row['seed']!=item['episode']:
            raise ValueError('saved building/seed differs from frozen plan')
        parse_fail+=row['utility']['parse_fail']
        if row['condition']!='classifier_all':
            for event in row['cost']['screen_audit']:
                budget_violations+=int(event['screens']>int(event['credit_before']+1e-9) or event['credit_after']< -1e-9)
        for obs in row['cost'].get('screen_call_measurements',[]):
            if obs.get('budget_units',1)!=1:raise ValueError('screen message has incorrect budget charge')
    if complete and seen!=expected:raise ValueError('both full confirmatory datasets are required')
    return {'rows':len(seen),'expected_rows':len(expected),'parse_failures':parse_fail,
            'budget_violations':budget_violations}


def inspect_run(directory,model,cfg,complete=False):
    directory=Path(directory);saved=yaml.safe_load((directory/'config.yaml').read_text());resolved=saved.pop('resolved')
    if saved!=cfg or resolved['mode']!='confirm' or resolved['backend']!='mlx' or resolved['model_key']!=model:
        raise ValueError('confirmatory model/config mismatch')
    if resolved['model_path']!=cfg['models'][model]:raise ValueError('model path mismatch')
    h=json.loads((directory/'input_hashes.json').read_text())
    if h['config_sha256']!=sha(directory/'config.yaml'):raise ValueError('run config changed')
    current={p.name:sha(p) for p in (ROOT/'configs/prompts').glob('*.txt')}
    if h['prompts']!=current or any(sha(directory/'prompts'/name)!=digest for name,digest in current.items()):
        raise ValueError('run prompt snapshot mismatch')
    if h['attacks']!=sha(ROOT/cfg['attacks']) or h['risk_constant']!=sha(ROOT/cfg['round2']['risk_rates']):
        raise ValueError('run attack library/risk constant mismatch')
    plan=confirmatory_plan(cfg,load_library(ROOT,cfg['attacks']))
    health=validate_rows(records(directory/'episodes.jsonl'),plan,complete)
    if complete:
        metrics=json.loads((directory/'metrics.json').read_text())
        if not metrics.get('complete') or metrics['rows']!=health['expected_rows'] or metrics['model']!=model:
            raise ValueError('missing completed-run health marker')
        calls=ok=0;minimum=None;keys=set()
        for trace in records(directory/'traces.jsonl'):
            key=(trace['plan_index'],trace['condition'])
            if key in keys:raise ValueError('duplicate trace row')
            keys.add(key)
            cells={(call['node'],call['step']) for call in trace['trace']}
            if len(cells)!=len(trace['trace']) or len(cells)!=cfg['world']['horizon']*(cfg['world']['n_zones']+1):
                raise ValueError('incomplete node-step trace coverage')
            if any('parse_ok' not in call and not call.get('quarantined') for call in trace['trace']):
                raise ValueError('missing generation/quarantine trace marker')
            for call in trace['trace']:
                if 'parse_ok' not in call:continue
                calls+=1;ok+=bool(call['parse_ok'])
                available=call.get('generation_meta',{}).get('minimum_available_bytes')
                if available is not None:minimum=available if minimum is None else min(minimum,available)
        expected={(i,c) for i in range(len(plan)) for c in CONDITIONS}
        if keys!=expected or calls!=metrics['parse_n']:
            raise ValueError('incomplete trace coverage')
        if calls-ok!=health['parse_failures']:raise ValueError('parse count does not match traces')
        health.update(parse_ok=ok,parse_n=calls,parse_rate=ok/calls,minimum_available_bytes=minimum)
        if health['budget_violations']:raise ValueError('screen credit violation; retain data and investigate')
    return health


def experiment_processes():
    """Read process names only; no model or outcome computation."""
    import re
    lines=subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines()
    return [line.strip() for line in lines if re.search(r'(?:python\S*|Python)\s+.*(?:scripts/)?(?:run_(?:round2|pilot|calibration|quarantine_probe|preflight)|develop_round2)\.py(?:\s|$)',line)]
