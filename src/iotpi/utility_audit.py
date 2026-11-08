"""Independent audit of legitimate-work labels and operational missed-work causes."""
from __future__ import annotations
from dataclasses import asdict


def audit_work(scenario,episode,trace):
    records=trace.get('trace',trace) if isinstance(trace,dict) else trace
    commands=episode['commands'];out=[]
    expected_count=len(scenario.expected);done=0
    for index,e in enumerate(scenario.expected):
        issued=[c for c in commands if (c['device'],c['command'])==(e.device,e.command)]
        timely=[c for c in issued if e.due<=c['step']<=e.deadline]
        completed=any(c['executed'] for c in timely);done+=completed
        requests=[r for r in records if r.get('role')=='gateway' and e.due<=r['step']<=e.deadline
                  and any(tuple(q[:2])==(e.device,e.command) for q in r.get('requests',[]))]
        if completed:cause='completed'
        elif timely:cause='actuator_blocked'
        elif issued:cause='outside_window'
        elif not requests:cause='gateway_never_requested'
        else:cause='coordinator_never_issued'
        out.append({'expected_index':index,**asdict(e),'completed':completed,'cause':cause,
                    'blocked_reasons':sorted({c['reason'] for c in timely if not c['executed']}),
                    'issued_steps':[c['step'] for c in issued],'timely_gateway_requests':len(requests)})
    return {'expected':expected_count,'done':done,'recorded_expected':episode['utility']['expected'],
            'recorded_done':episode['utility']['done'],
            'labels_match':(expected_count,done)==(episode['utility']['expected'],episode['utility']['done']),
            'expected_work':out}
