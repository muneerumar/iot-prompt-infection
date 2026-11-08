"""Preregistered v2 selection. Evaluator labels never reach candidate detectors."""
from __future__ import annotations
import hashlib
from collections import defaultdict
from fractions import Fraction

from .stats import cluster_bootstrap, wilson


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def build_corpus(generations, cfg, train):
    """Actual incoming messages in training/calibration units, retaining event frequency.

    Standalone benign output notices were scored by calibration but are not fed to a
    later generation; include those too. Attack-labelled but noncarrying outputs are
    never promoted to rewritten positives here.
    """
    lo, hi = cfg['split']['calibration_contexts']
    corpus = []
    source_for_text = {}
    for row in generations:
        context = row['context']
        if not lo <= int(context.removeprefix('ctx')) < hi:
            raise ValueError('corpus contains a non-calibration building')
        attack = row.get('attack')
        if attack and attack['family'] not in train:
            continue
        prefix = (context, row['episode_seed'])
        for j, message in enumerate(row['inputs']):
            positive = bool(message.get('gt_attack'))
            if positive and message.get('gt_family') not in train:
                raise ValueError('held-out-family message in training unit')
            bucket = ('raw' if message['kind']=='ticket' else 'rewritten') if positive else 'benign'
            if positive and message['kind'] not in ('ticket','report','notice'):
                raise ValueError('unexpected positive channel')
            text = message['text']
            source = source_for_text.get((*prefix, message['sender'], message['kind'], text))
            source = source or digest([*prefix, message['sender'], message['id'], text])
            corpus.append({'event_id':f"{row['generation_id']}:in:{j}", 'source_id':source,
                'context':context, 'episode_seed':row['episode_seed'], 'family':attack['family'] if attack else None,
                'bucket':bucket, 'kind':message['kind'], 'text':text})
        # Source identity for reports/notices consumed by subsequent calls.
        from .engine import _requests_text
        report = f"{row['report']} Requests: {_requests_text(row['requests'])}"
        for channel, text in [('report',report),('notice',row['notice'])]:
            source_for_text[(*prefix,row['node'],channel,text)] = f"generation:{row['generation_id']}:{channel}"
        if attack is None and row['role']=='coordinator' and row['notice']:
            corpus.append({'event_id':f"{row['generation_id']}:out:notice",
                'source_id':f"generation:{row['generation_id']}:notice", 'context':context,
                'episode_seed':row['episode_seed'], 'family':None, 'bucket':'benign',
                'kind':'notice', 'text':row['notice']})
    if not all(any(r['bucket']==b for r in corpus) for b in ('raw','rewritten','benign')):
        raise ValueError('selection requires raw, rewritten, and benign observations')
    return corpus


def summarise_scores(rows):
    result = {}
    for bucket in ('raw','rewritten','benign','pooled'):
        subset = [r for r in rows if (r['bucket']!='benign' if bucket=='pooled' else r['bucket']==bucket)]
        k = sum(r['flagged'] for r in subset)
        n = len(subset)
        _, lo, hi = cluster_bootstrap([float(r['flagged']) for r in subset], [r['source_id'] for r in subset])
        result[bucket] = {'k':k,'n':n,'rate':k/n if n else None,'source_cluster_ci95':[lo,hi],
                          'n_sources':len({r['source_id'] for r in subset}), 'wilson_ci95':list(wilson(k,n)[1:])}
    result['seconds_per_call'] = sum(r['seconds'] for r in rows)/len(rows) if rows else None
    result['calls'] = len(rows)
    result['by_family'] = {}
    for family in sorted({r['family'] for r in rows if r.get('family') and r['bucket']!='benign'}):
        result['by_family'][family] = {}
        for bucket in ('raw','rewritten'):
            subset=[r for r in rows if r.get('family')==family and r['bucket']==bucket]
            k=sum(r['flagged'] for r in subset);n=len(subset)
            _,lo,hi=cluster_bootstrap([float(r['flagged']) for r in subset],[r['source_id'] for r in subset])
            result['by_family'][family][bucket]={'k':k,'n':n,'rate':k/n if n else None,'source_cluster_ci95':[lo,hi]}
    result['benign_by_kind'] = {}
    for kind in sorted({r['kind'] for r in rows if r.get('kind') and r['bucket']=='benign'}):
        subset=[r for r in rows if r.get('kind')==kind and r['bucket']=='benign']
        k=sum(r['flagged'] for r in subset);n=len(subset)
        result['benign_by_kind'][kind]={'k':k,'n':n,'rate':k/n}
    return result


def select_detector(candidates, rule):
    """Integer ratios enforce inclusive FPR and exact recall ties, without rounding."""
    eligible = []
    cap = Fraction(str(rule['max_false_positive_rate']))
    for order, name in enumerate(rule['candidates']):
        candidate = candidates[name]
        if candidate['status']!='available':
            continue
        s = candidate['summary']
        if not s['benign']['n'] or not s['pooled']['n']:
            continue
        if Fraction(s['benign']['k'],s['benign']['n']) <= cap:
            eligible.append((-Fraction(s['pooled']['k'],s['pooled']['n']),s['seconds_per_call'],order,name))
    return min(eligible)[-1] if eligible else None
