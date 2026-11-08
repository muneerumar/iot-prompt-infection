import copy
from pathlib import Path

import pytest

from iotpi.detectors import Detector, KindTieredDetector, TWO_TIER_SPEC, make_detector
from iotpi.guard import GuardConfig, GuardNetwork
from iotpi.messages import Message
from iotpi.topologies import pilot


class Tier(Detector):
    def __init__(self,value,seconds,llm=False):
        super().__init__(.5);self.value=value;self.duration=seconds;self.llm=llm
    def score(self,msg):
        self.calls+=1;self.seconds+=self.duration
        self.observations.append({'seconds':self.duration,'physical_seconds':self.duration,
            'kind':msg.kind,'score':self.value,'llm_calls':int(self.llm),
            'prompt_tokens':12 if self.llm else 0,'generation_tokens':1 if self.llm else 0})
        return self.value


def message(kind='ticket',**kwargs):
    return Message('m','external','gw_north',kind,'fixture',0,**kwargs)


@pytest.mark.parametrize('protect,judge,expected,p_calls,j_calls',[
    (.1,.1,False,1,1),(.1,.5,True,1,1),(.5,.1,True,1,0),(.9,.9,True,1,0)])
def test_ticket_or_inclusive_threshold_and_short_circuit(protect,judge,expected,p_calls,j_calls):
    p,j=Tier(protect,1),Tier(judge,2,True);d=KindTieredDetector(p,j)
    assert d.flags(message())[0] is expected
    assert (d.calls,p.calls,j.calls)==(1,p_calls,j_calls)
    obs=d.observations[-1]
    assert obs['budget_units']==1 and obs['tiers']['protectai']['seconds']==1
    if j_calls:
        assert obs['tiers']['judge']['seconds']==2 and obs['seconds']==3
        assert obs['llm_calls']==1 and obs['prompt_tokens']==12
    else:assert 'judge' not in obs['tiers']


@pytest.mark.parametrize('kind',['report','notice','sensor','digest'])
def test_non_ticket_never_runs_judge_even_when_protectai_negative(kind):
    p,j=Tier(.1,1),Tier(1,2,True);d=KindTieredDetector(p,j)
    assert not d.flags(message(kind))[0] and j.calls==0
    assert d.observations[-1]['tiers_run']==['protectai']


def test_composite_ignores_evaluator_labels_and_resets_without_tier_leakage():
    d=KindTieredDetector(Tier(.1,1),Tier(1,2,True))
    first=d.flags(message(gt_attack=False,gt_family=None))
    d.calls,d.seconds,d.observations=0,0.,[]
    second=d.flags(message(gt_attack=True,gt_family='authority',gt_hop=7))
    assert first==second and d.calls==1 and d.seconds==3
    assert len(d.observations)==1 and d.protectai.observations==d.judge.observations==[]


def test_one_message_consumes_one_guard_credit_with_both_tiers():
    d=KindTieredDetector(Tier(.1,1),Tier(.1,2,True))
    graph=pilot(['north','south'])
    guard=GuardNetwork(GuardConfig('one',screen='taint',budget_per_node=1),graph,d)
    guard.begin_step(0)
    inbox=[message(tainted=True),Message('m2','external','gw_north','ticket','fixture2',0,tainted=True)]
    assert len(guard.filter_inbox('gw_north',inbox,0))==2
    assert guard.nodes['gw_north'].credit==0
    assert d.calls==d.protectai.calls==d.judge.calls==1
    cost=guard.cost_summary()
    assert cost['screens']==1 and cost['screen_tier_calls']=={'protectai':1,'judge':1}
    assert cost['screen_tier_seconds']=={'protectai':1.,'judge':2.}


def test_factory_uses_same_backend_and_fixed_prompt(monkeypatch,tmp_path):
    import iotpi.detectors as module
    captured={}
    def hf(model,threshold,device):
        captured['hf']=(model,threshold,device);return Tier(.1,1)
    def judge(backend,path,threshold):
        captured['judge']=(backend,path,threshold);return Tier(.1,2,True)
    monkeypatch.setattr(module,'HFDetector',hf);monkeypatch.setattr(module,'LLMJudgeDetector',judge)
    backend=object();d=make_detector(TWO_TIER_SPEC,.5,backend,tmp_path,'cpu')
    assert isinstance(d,KindTieredDetector)
    assert captured['hf']==('protectai/deberta-v3-base-prompt-injection-v2',.5,'cpu')
    assert captured['judge']==(backend,tmp_path/'configs/prompts/judge.txt',.5)
    with pytest.raises(ValueError):make_detector(TWO_TIER_SPEC,.5,None,tmp_path)
    with pytest.raises(ValueError):KindTieredDetector(Tier(.1,1),Tier(.1,2),.4)


def test_adaptive_query_uses_ticket_routing_and_records_both_tiers():
    from iotpi.attacks import AttackSpec,adaptive_rewrite
    spec=AttackSpec('authority','spread','north',0,'Sabc123',None,None,'fixture')
    d=KindTieredDetector(Tier(.1,1),Tier(.1,2,True))
    out=adaptive_rewrite(spec,{},None,d,8,.5,0)
    assert out.adaptive_queries==1 and d.calls==1
    assert out.adaptive_cost['screen_tier_seconds']=={'protectai':1.,'judge':2.}
    assert out.adaptive_cost['screen_llm_calls']==1
