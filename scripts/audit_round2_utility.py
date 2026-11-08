"""A2: read-only audit of round-one benign rows. New model development is separate."""
from __future__ import annotations
import json,sys
from collections import Counter,defaultdict
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.experiment import build_scenario
from iotpi.utility_audit import audit_work
from iotpi.runlog import Run,read_jsonl
from round2_support import report,table,DOCS,sha

def main():
    source=ROOT/'results/20261105-023254_pilot_llama_v3_llama-4bit'
    cfg=yaml.safe_load((source/'config.yaml').read_text())
    episodes={(r['plan_index'],r['condition']):r for r in read_jsonl(source/'episodes.jsonl') if r['attack'] is None}
    run=Run(ROOT,'round2_A2_round1_benign_audit',{'seed':13,'source':str(source),'read_only':True})
    counts=defaultdict(Counter);blocks=defaultdict(Counter);mismatches=[];seen=set()
    for line in (source/'traces.jsonl').open():
        trace=json.loads(line);key=(trace['plan_index'],trace['condition'])
        if key not in episodes:continue
        episode=episodes[key];seen.add(key);sc=build_scenario(cfg,None,episode)
        result=audit_work(sc,episode,trace)
        if not result['labels_match']:mismatches.append({'key':key,**result})
        condition=episode['condition'];counts[condition]['episodes']+=1
        for row in result['expected_work']:
            counts[condition][row['cause']]+=1
            if not row['completed']:
                run.record('missed_expected',{'plan_index':key[0],'condition':condition,'context':sc.context_id,**row})
                blocks[condition].update(row['blocked_reasons'])
        run.record('episode_audit',{'plan_index':key[0],'condition':condition,**{k:v for k,v in result.items() if k!='expected_work'}})
    if seen!=set(episodes):raise RuntimeError('benign trace rows missing')
    data={'source':str(source),'source_episode_sha256':sha(source/'episodes.jsonl'),'source_trace_sha256':sha(source/'traces.jsonl'),
          'counts':dict(counts),'blocked_reasons':dict(blocks),'evaluator_mismatches':mismatches,
          'method':'Completed = any executed exact device/command in the inclusive due/deadline window. For misses: timely blocked issue, else any out-of-window issue, else no timely gateway request, else no coordinator issue. Quarantined or dropped input can therefore enter gateway-never-requested; categories are operational, not counterfactual causal proof.',
          'historical_scope':'Only saved round-one benign rows; no attack-derived outcomes used. Reconstruction of their historical scenarios is read-only. All new model development uses calibration contexts 0-9.'}
    run.save_json('metrics.json',data)
    headers=['Condition','Completed','Gateway never requested','Coordinator never issued','Actuator blocked','Outside window']
    rows=[[name,*[c.get(k,0) for k in ('completed','gateway_never_requested','coordinator_never_issued','actuator_blocked','outside_window')]] for name,c in sorted(counts.items())]
    body='# A2: legitimate-work evaluator audit before prompt edits\n\n'+table(headers,rows)
    body+='\n\n'+data['method']+'\n\n'+data['historical_scope']
    body+='\n\nIndependent expected-count and completion rechecks found '+str(len(mismatches))+' evaluator mismatches. Inclusive windows and exact identifiers are tested independently. Duplicate same-device/same-command expected events retain the existing service-state semantics: one execution can satisfy overlapping windows; this is a metric limitation, not silently changed here. Terminal deadlines are bounded by the fixed episode horizon. Calibration-building benign baseline evidence will be added before any prompt change.'
    report('A2_utility_audit.md',body,[run.dir/'metrics.json'])
    run.close();print(run.dir)
    if mismatches:raise RuntimeError('evaluator mismatch must be resolved before prompt development')
    return run.dir
if __name__=='__main__':main()
