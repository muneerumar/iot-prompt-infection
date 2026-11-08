"""Round-two plan and fixed inputs. Construction never executes an evaluation case."""
from __future__ import annotations
import copy
from dataclasses import replace

CONDITIONS=['policy_only','classifier_all','classifier_cache','classifier_trust','trustgate','provenance_block']


def benign_plan(cfg,timing=False):
    lo,hi=cfg['split']['calibration_contexts']
    n=2 if timing else cfg['round2']['development']['benign_episodes']
    base=1_100_000 if timing else 1_000_000
    plan=[{'context':lo+i%(hi-lo),'episode':base+i,'family':None,'goal':None,
             'heldout_family':False,'stratum':'timing' if timing else 'benign_development'} for i in range(n)]
    if timing:plan[1].update(family='authority',goal='c3_safety',c3_variant=0)
    return plan


def confirmatory_plan(cfg,lib):
    lo,hi=cfg['split']['evaluation_contexts']
    if (lo,hi)!=(200,240):raise ValueError('round two requires fresh buildings 200-239')
    plan=[]
    def add(family,goal,stratum,variant=None,adaptive=False):
        i=len(plan)
        plan.append({'context':lo+i%(hi-lo),'episode':2_000_000+i,'family':family,'goal':goal,
            'heldout_family':family in lib['split']['heldout'] if family else False,
            'stratum':stratum,'c3_variant':variant,'adaptive':adaptive})
    for family in sorted(lib['families']):
        for i in range(cfg['round2']['c3_main_per_family']):add(family,'c3_safety','c3_main',i%2)
    for family in sorted(lib['families']):
        for goal in ['exfil']*3+['c1_out_of_policy','c2_unlock']:add(family,goal,'sanity')
    for family in ['authority','dormant']:
        for i in range(10):add(family,'c3_safety','adaptive_c3',i%2,True)
    for _ in range(30):add(None,None,'benign')
    return plan


def build_round2_scenario(cfg,lib,item):
    from .scenario import make_scenario
    return make_scenario(item['context'],item['episode'],cfg['world']['n_zones'],cfg['world']['horizon'],
        lib,item['family'],item['goal'],c3_variant=item.get('c3_variant'))
