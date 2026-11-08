"""Selected-prompt deterministic reload/parse/memory check on benign calibration inputs."""
import argparse,json,sys,hashlib,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.backends import make_backend
from iotpi.agents import LLMAgent
from iotpi.experiment import load_prompts
from iotpi.calibrate import _gateway_inbox
from iotpi.messages import Message
from iotpi.scenario import make_scenario
from iotpi.topologies import pilot
from iotpi.runlog import Run

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/round2.yaml').read_text());prompts=load_prompts(ROOT)
    sc=make_scenario(0,1_000_000,2,4);graph=pilot(sc.zones);zone=sc.zones[0]
    inputs={'gateway':_gateway_inbox(sc,zone,0),'coordinator':[
        Message('r','gw_'+zone,'coord','report',f'Zone {zone} normal. Requests: none',0,payload={'requests':[]})]}
    cfg['resolved']={'mode':'selected-prompt preflight','model_key':a.model,'model_path':cfg['models'][a.model],'calibration_context':0,'episode_seed':1000000}
    run=Run(ROOT,'round2_preflight_'+a.model,cfg)
    (run.dir/'prompts').mkdir()
    for source in (ROOT/'configs/prompts').glob('*.txt'):shutil.copyfile(source,run.dir/'prompts'/source.name)
    run.save_json('prompt_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run.dir/'prompts').glob('*.txt')})
    baseline={};responses=[];backend=None
    try:
        for reload in range(2):
            backend=make_backend('mlx',str((ROOT/cfg['models'][a.model]).resolve()))
            for role in ('gateway','coordinator'):
                node=graph.gateway(zone) if role=='gateway' else 'coord';agent=LLMAgent(node,role,backend,prompts,cfg['generation'])
                for repeat in range(3 if reload==0 else 1):
                    out=agent.respond(inputs[role],'',0,sc,graph,seed=13)
                    row={'role':role,'reload':reload,'repeat':repeat,'output':out.text,'parse_ok':out.parse_ok,
                         'seconds':out.seconds,'prompt':out.prompt,'generation_meta':out.generation_meta}
                    run.record('generations',row);responses.append(row)
                    if reload==0 and repeat==0:baseline[role]=out.text
            backend.close();backend=None
        provenance=json.loads((ROOT/cfg['models'][a.model]/'cgcr_provenance.json').read_text())
        data={'model':a.model,'deterministic':all(r['output']==baseline[r['role']] for r in responses),
              'parse_ok':sum(r['parse_ok'] for r in responses),'parse_n':len(responses),
              'parse_rate':sum(r['parse_ok'] for r in responses)/len(responses),
              'minimum_available_bytes':min(r['generation_meta']['minimum_available_bytes'] for r in responses),
              'peak_metal_bytes':max(r['generation_meta']['peak_memory_bytes'] for r in responses),
              'model_source_revision':provenance['source_revision']}
        run.save_json('metrics.json',data);run.log(str(data))
        if not data['deterministic'] or data['parse_rate']<.95:raise RuntimeError('selected-prompt preflight failed')
    finally:
        if backend:backend.close()
        run.close()
if __name__=='__main__':main()
