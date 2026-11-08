"""Recompute saved summaries from append-only raw rows. No model calls or writes."""
from __future__ import annotations
import argparse,ast,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iotpi.calibrate import Counter,detector_rates,rates_from_counts
from iotpi.predict import brier
from iotpi.report import load_rows,summarise
from iotpi.runlog import read_jsonl
from iotpi.stats import cluster_bootstrap

def recompute_calibration(directory):
    original=json.loads((directory/'calibration.json').read_text())
    agent=Counter(); detectors=defaultdict(Counter);training=defaultdict(Counter)
    for row in read_jsonl(directory/'agent_measurements.jsonl'):
        agent.add(tuple(row['key']),row['hit'],row)
    for row in read_jsonl(directory/'detector_measurements.jsonl'):
        detectors[row['detector']].add((row['bucket'],row['family'] if row['is_attack'] else row['kind']),row['flagged'],row)
        if row['family'] is None or row['family'] in original['training_families']:
            training[row['detector']].add((row['bucket'],row['family'] if row['is_attack'] else row['kind']),row['flagged'],row)
    counts={'agent_counts':agent.table(),'detector_counts':{k:v.table() for k,v in detectors.items()},
            'training_detector_counts':{k:v.table() for k,v in training.items()}}
    if 'source_shortfalls' in original['counts']:
        counts['source_shortfalls']=read_jsonl(directory/'source_shortfalls.jsonl')
    families=sorted({ast.literal_eval(k)[2] for k in counts['agent_counts']} - {'_pooled'})
    out={**original,'counts':counts,'rates':rates_from_counts(counts,families),
         'training_rates':rates_from_counts(counts,original['training_families']),
         'detectors':{k:detector_rates(counts,k) for k in detectors},
         'training_detectors':{k:detector_rates({'detector_counts':counts['training_detector_counts']},k) for k in training}}
    return out

def recompute_prediction(directory):
    groups=defaultdict(list)
    for row in read_jsonl(directory/'predictions.jsonl'):
        groups[(row['condition'],'all')].append(row)
        groups[(row['condition'],row['split'])].append(row)
    out={}
    for (condition,split),rows in sorted(groups.items()):
        observed=cluster_bootstrap([float(r['observed']) for r in rows],[r['context_id'] for r in rows])
        means=np.mean(np.array([r['pred_draws'] for r in rows]),axis=0)
        predictive=(np.asarray([r['predictive_outcomes'] for r in rows],dtype=float).mean(axis=0).ravel()
                    if all('predictive_outcomes'in r for r in rows) else means)
        bound_rows=[r for r in rows if r['bound'] is not None]
        bounds=[r['bound'] for r in bound_rows]
        observed_bound=sum(r['observed'] for r in bound_rows)/len(bound_rows) if bound_rows else None
        out[f'{condition}|{split}']={'n':len(rows),'observed':observed,
            'predicted_mean':float(np.mean([r['pred_mean'] for r in rows])),
            'predicted_95':[float(np.percentile(predictive,2.5)),float(np.percentile(predictive,97.5))],
            'brier':brier([r['pred_mean'] for r in rows],[r['observed'] for r in rows]),
            'mean_bound':float(np.mean(bounds)) if bounds else None,
            'observed_on_bound_episodes':observed_bound,
            'observed_le_bound':observed_bound<=float(np.mean(bounds)) if bounds else None}
        if all('predictive_outcomes'in r for r in rows):
            out[f'{condition}|{split}']['parameter_mean_95']=[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))]
            out[f'{condition}|{split}']['predictive_replications']=len(predictive)
    return out

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--print',action='store_true')
    a=p.parse_args(argv);directory=Path(a.run)
    if (directory/'selection.json').exists():
        from iotpi.detector_selection import summarise_scores,select_detector,digest
        filename='selection.json';computed=json.loads((directory/filename).read_text())
        grouped=defaultdict(list)
        for row in read_jsonl(directory/'scores.jsonl'):grouped[row['detector']].append(row)
        for name,candidate in computed['candidates'].items():
            if candidate['status']=='available':candidate['summary']=summarise_scores(grouped[name])
        computed['selected']=select_detector(computed['candidates'],computed['rule']) if computed['comparison_complete'] else None
        computed['corpus_hash']=digest(read_jsonl(directory/'corpus.jsonl'))
    elif (directory/'calibration.json').exists():
        filename='calibration.json';computed=recompute_calibration(directory)
    elif (directory/'prediction_summary.json').exists():
        filename='prediction_summary.json';computed=recompute_prediction(directory)
    else:
        filename='summary.json';rows=load_rows(directory)
        reference='trustgate_control' if any(r['condition']=='trustgate_control' for r in rows) else 'trustgate'
        computed=summarise(rows,reference=reference)
        original=json.loads((directory/filename).read_text())
        if 'backend' in original: computed['backend']=yaml.safe_load((directory/'config.yaml').read_text())['resolved']['backend']
    original=json.loads((directory/filename).read_text())
    matches=json.loads(json.dumps(computed))==original
    print(json.dumps(computed if a.print else {'file':str(directory/filename),'matches':matches},indent=2))
    return 0 if matches else 1
if __name__=='__main__':sys.exit(main())
