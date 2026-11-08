"""Real adaptive-rewrite engineering check, training family and calibration context only."""
import json,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.attacks import adaptive_rewrite,load_library,strip_rewrite_preamble
from iotpi.backends import MLXBackend
from iotpi.detectors import HFDetector
from iotpi.runlog import Run
from iotpi.scenario import make_scenario

def main():
    cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text());run=Run(ROOT,'audit_h_real',cfg);backend=None
    try:
        detector=HFDetector(cfg['detectors']['screen'][3:],cfg['detectors']['screen_threshold'])
        backend=MLXBackend(str((ROOT/cfg['models']['llama-4bit']).resolve()))
        lib=load_library(ROOT,cfg['attacks']);scenario=make_scenario(0,42,2,4,lib,'imperative','c3_safety')
        class Rewriter:
            def generate(self,messages,**kwargs):
                out=backend.generate(messages,**kwargs)
                run.record('rewrites',{'messages':messages,'output':out.text,'seconds':out.seconds,'meta':out.meta})
                return out
        class Queries:
            calls=0
            def score_text(self,text):
                score=detector.score_text(text);self.calls+=1
                run.record('queries',{'query':self.calls,'text':text,'score':score});return score
        queries=Queries();budget=cfg['pilot']['adaptive']['queries']
        result=adaptive_rewrite(scenario.attack,lib,Rewriter(),queries,budget,detector.threshold,cfg['seed'])
        data={'family':result.family,'context':scenario.context_id,'canary_survives':result.canary in result.text,
              'target_device_survives':result.target[0] in result.text,'reported_queries':result.adaptive_queries,
              'actual_queries':queries.calls,'query_cap':budget,'queries_match':result.adaptive_queries==queries.calls<=budget,
              'normalized':strip_rewrite_preamble(result.text)==result.text,'rewrite_exercised':result.adaptive_cost['llm_calls']>0,'cost':result.adaptive_cost,'score':result.adaptive_score}
        run.save_json('adaptive_check.json',data);run.save_json('metrics.json',data)
        print(json.dumps({'run':str(run.dir),**data}),flush=True)
    except BaseException as error:
        run.save_json('failure.json',{'error':repr(error)});raise
    finally:
        if backend is not None:backend.close()
        run.close()
    return run.dir
if __name__=='__main__':main()
