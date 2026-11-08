import copy
import pytest
from iotpi.detectors import PROTECTAI_V2
from iotpi.v3_revision import replay_tiers,project_timing


def fixture():
    corpus=[];scores=[]
    for i,(kind,bucket,p,j) in enumerate([('ticket','raw',False,True),('report','rewritten',False,True),
                                         ('ticket','benign',False,False),('report','benign',False,True)]):
        event={'event_id':str(i),'source_id':str(i),'context':'ctx000','family':'authority',
               'episode_seed':i,'kind':kind,'bucket':bucket,'text':'Normal. Requests: smoke_north maintenance_mode (service)'}
        corpus.append(event)
        for detector,flag,seconds in [(PROTECTAI_V2,p,1),('judge',j,2)]:
            scores.append({k:v for k,v in event.items() if k!='text'} | {
                'detector':detector,'flagged':flag,'score':float(flag),'seconds':seconds})
    return scores,corpus


def test_replay_joins_exact_events_and_routes_by_kind():
    scores,corpus=fixture();rows,evidence=replay_tiers(scores,corpus,{'authority'},{'ctx000'})
    assert [r['flagged'] for r in rows]==[True,False,False,False]
    assert [r['seconds'] for r in rows]==[3,1,3,1]
    assert evidence['raw_tickets']=={'n':1,'protectai':0,'judge':1,'either':1}
    assert evidence['benign_reports']['judge']==evidence['judge_benign_reports_with_device_requests']==1


def test_replay_rejects_missing_duplicate_or_heldout_events():
    scores,corpus=fixture()
    with pytest.raises(ValueError):replay_tiers(scores[:-1],corpus,{'authority'},{'ctx000'})
    with pytest.raises(ValueError):replay_tiers(scores+[scores[0]],corpus,{'authority'},{'ctx000'})
    with pytest.raises(ValueError):replay_tiers(scores,corpus,{'imperative'},{'ctx000'})
    with pytest.raises(ValueError):replay_tiers(scores,corpus,{'authority'},{'ctx100'})


def test_projection_includes_judge_cost_and_refreshes_completed_calibration():
    cfg={'world':{'n_zones':2,'horizon':4},'pilot':{'conditions':['a','b'],'benign_episodes':2,
         'adaptive':{'enabled':True,'families':['authority'],'episodes_per_family':1,'queries':8}}}
    base={'roles':{'gateway':{'mean_seconds':10},'coordinator':{'mean_seconds':12}},'cache_hit_rate':.5,
          'calibration_call_upper_bound':40,'v3_calibration_screen_upper_bound':70,'estimated_pilot_episodes':5}
    observations=[{'kind':'ticket','seconds':3,'tiers':{'protectai':{'seconds':1},'judge':{'seconds':2}}},
                  {'kind':'report','seconds':1,'tiers':{'protectai':{'seconds':1}}}]
    first=project_timing(base,cfg,observations)
    assert first['v3_tier_timing']['judge']['calls']==1
    assert first['v3_total_slow_screen_mix_hours']>first['estimated_total_with_probe_hours']
    assert first['estimated_calibration_plus_pilot_uncached_hours']>first['estimated_calibration_plus_pilot_hours']
    done=project_timing(base,cfg,observations,[{'role':'gateway','seconds':10},{'role':'coordinator','seconds':12}])
    assert done['v3_calibration_complete']
    assert done['v3_remaining_after_calibration_hours']==pytest.approx(done['estimated_total_with_probe_hours']-26/3600)
    with pytest.raises(ValueError):project_timing(base,cfg,observations[1:])
