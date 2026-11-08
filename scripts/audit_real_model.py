"""Small real-model audit on calibration contexts only; no pilot results or tuning."""
from __future__ import annotations

import hashlib
import json
import resource
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from iotpi.agents import LLMAgent
from iotpi.attacks import load_library
from iotpi.backends import MLXBackend, _ensure_available
from iotpi.calibrate import _gateway_inbox
from iotpi.detectors import HFDetector
from iotpi.engine import _requests_text
from iotpi.experiment import load_prompts
from iotpi.messages import Message
from iotpi.runlog import Run
from iotpi.scenario import make_scenario
from iotpi.topologies import pilot


def main():
    cfg = yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    run = Run(ROOT, 'audit_real_llama', cfg)
    backend = None
    outcomes = {'status':'in_progress', 'model':'llama-4bit', 'generations':[], 'determinism':{}}
    try:
        outcomes['available_before_classifier'] = _ensure_available()
        detector = HFDetector(cfg['detectors']['screen'][3:], cfg['detectors']['screen_threshold'])
        outcomes['available_with_classifier'] = _ensure_available()
        lib = load_library(ROOT, cfg['attacks'])
        prompts = load_prompts(ROOT)
        model_path = (ROOT/cfg['models']['llama-4bit']).resolve()
        seed = cfg['seed']
        sc = make_scenario(0, 123, cfg['world']['n_zones'], cfg['world']['horizon'])
        graph = pilot(sc.zones)
        node = graph.gateway(sc.zones[0])
        def log_generation(check, role, output, **extra):
            row={'check':check, 'role':role, **output, **extra}
            run.record('generations', row)
            outcomes['generations'].append({k: row[k] for k in ['check','role','parse_ok','seconds','generation_meta']})
            run.log(f"{check}: role={role} parse_ok={row['parse_ok']} seconds={row['seconds']:.3f}")
        for load_index in range(2):
            backend=MLXBackend(str(model_path))
            agent=LLMAgent(node,'gateway',backend,prompts,cfg['generation'])
            messages=agent.build_messages(_gateway_inbox(sc,sc.zones[0],0), '',0,sc,graph)
            texts=[]
            for repetition in range(3):
                g=backend.generate(messages,seed=seed,max_tokens=cfg['generation']['max_tokens'],temperature=0)
                texts.append(g.text)
                run.record('determinism', {'load_index':load_index,'repetition':repetition,
                    'text':g.text,'sha256':hashlib.sha256(g.text.encode()).hexdigest(),
                    'seconds':g.seconds,'prompt_tokens':g.prompt_tokens,'generation_tokens':g.generation_tokens,'meta':g.meta})
                run.log(f'determinism load={load_index} repetition={repetition}')
            outcomes['determinism'][str(load_index)]=[hashlib.sha256(text.encode()).hexdigest() for text in texts]
            if load_index == 0:
                backend.close(); backend=None
        hashes=[h for hs in outcomes['determinism'].values() for h in hs]
        outcomes['deterministic']=len(set(hashes))==1
        cases=[]
        train=lib['split']['train']
        for i in range(10):
            family=train[i % len(train)] if i % 2 == 0 else None
            goal=cfg['pilot']['goals'][i % len(cfg['pilot']['goals'])] if family else None
            sc=make_scenario(i, 700+i, cfg['world']['n_zones'],cfg['world']['horizon'],lib,family,goal)
            graph=pilot(sc.zones)
            zone=sc.attack.zone if sc.attack else sc.zones[0]
            step=sc.attack.step if sc.attack else 0
            agent=LLMAgent(graph.gateway(zone),'gateway',backend,prompts,cfg['generation'])
            output=agent.respond(_gateway_inbox(sc,zone,step), '',step,sc,graph,seed=seed+i)
            data={'text':output.text,'report':output.report,'requests':output.requests,'replies':output.replies,
                  'parse_ok':output.parse_ok,'prompt':output.prompt,'seconds':output.seconds,
                  'prompt_tokens':output.prompt_tokens,'generation_tokens':output.generation_tokens,
                  'generation_meta':output.generation_meta}
            log_generation(f'gateway_{i}','gateway',data,context=sc.context_id, family=family)
            cases.append((sc,graph,zone,step,output))
        for i,(sc,graph,zone,step,gateway) in enumerate(cases):
            report=Message(f'audit{i}',graph.gateway(zone),'coord','report',
                f'{gateway.report} Requests: {_requests_text(gateway.requests)}',step,
                frozenset({sc.attack.ticket_id}) if sc.attack else frozenset(),bool(sc.attack),
                {'requests':gateway.requests})
            agent=LLMAgent('coord','coordinator',backend,prompts,cfg['generation'])
            output=agent.respond([report], '',step,sc,graph,seed=seed+100+i)
            data={'text':output.text,'report':output.report,'commands':output.commands,'notice':output.notice,
                  'parse_ok':output.parse_ok,'prompt':output.prompt,'seconds':output.seconds,
                  'prompt_tokens':output.prompt_tokens,'generation_tokens':output.generation_tokens,
                  'generation_meta':output.generation_meta}
            log_generation(f'coordinator_{i}','coordinator',data,context=sc.context_id)
        outcomes['parse_ok']=sum(row['parse_ok'] for row in outcomes['generations'])
        outcomes['parse_n']=len(outcomes['generations'])
        outcomes['parse_rate']=outcomes['parse_ok']/outcomes['parse_n']
        outcomes['minimum_available_bytes']=backend.minimum_available_bytes
        outcomes['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        outcomes['peak_metal_bytes']=max(row['generation_meta'].get('peak_memory_bytes',0) for row in outcomes['generations'])
        outcomes['status']='complete'
        run.save_json('audit_real.json',outcomes)
        run.save_json('metrics.json',outcomes)
        run.log(json.dumps({k:v for k,v in outcomes.items() if k not in ['generations','determinism']}))
    except BaseException as error:
        outcomes['status']='failed'
        outcomes['error']=repr(error)
        run.save_json('audit_real_failure.json',outcomes)
        raise
    finally:
        if backend is not None: backend.close()
        run.close()
    return run.dir


if __name__=='__main__':
    print(main(),flush=True)
