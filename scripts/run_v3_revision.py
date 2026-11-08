"""Document user-fixed v3 routing, probe timing, freeze v3, then run sequentially."""
from __future__ import annotations
import argparse,json,os,re,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.attacks import load_library
from iotpi.detectors import TWO_TIER_SPEC
from iotpi.runlog import Run,read_jsonl
from iotpi.v3_revision import replay_tiers,project_timing
from workflow_support import DOCS,commit,evidence,report,sha,timing_analysis
from workflow_after_freeze import select_owned_run
from phase_reports import freeze,table


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--test-run',required=True);parser.add_argument('--handoff-copy',required=True)
    a=parser.parse_args(argv);cfg=yaml.safe_load((ROOT/'configs/default.yaml').read_text())
    if cfg['detectors']['screen']!=TWO_TIER_SPEC or cfg['detectors']['screen_threshold']!=.5:
        raise ValueError('v3 spec and threshold required')
    test_run=Path(a.test_run);test_output=(test_run/'pytest.txt').read_text()
    if json.loads((test_run/'metrics.json').read_text())['returncode']!=0:raise ValueError('regression suite failed')
    run=Run(ROOT,'v3_revision_pipeline',{'seed':cfg['seed'],'design':'configs/detector_design_v3.json','test_run':str(test_run)})
    child=None;stage='preparing v3 evidence';worker_started=False
    def status(text):
        record=evidence('v3_revision_status',{'status':text,'stage':stage,'pid':os.getpid(),
            'child_pid':child.pid if child else None,'log':str(run.dir/'run.log'),
            'partial_v2':'results/20261104-175118_cal_llama_v2_llama-4bit','gates':{'gate1':'not reached','gate2':'not reached','gate3':'not reached'}})
        report('HANDOFF.md','# Handoff — v3 in progress\n\n'+text+'\n\nV2 calibration and both controllers were stopped at the user request; its partial directory is preserved and will not supply fitted rates.\n\nStage: '+stage+'. Pipeline PID: `'+str(os.getpid())+'`; log: `'+str(run.dir/'run.log')+'`. Child PID: `'+str(child.pid if child else None)+'`.\n\nOne message costs one budget unit. Tickets use ProtectAI OR the existing judge, with ProtectAI positives short-circuited; non-ticket messages use ProtectAI alone. Both thresholds stay 0.5. Each executed tier logs time. The judge shares the agents model.\n\nAfter the calibration-building/training-family timing probe: refresh the runtime projection, freeze frozen-pilot-v3, run a new calibration, refresh timing from its journal, select cache distance, then run the pilot, quarantine probe and prediction. No automatic calibration doubling.\n\nNo second model job or tests may start while this pipeline is alive. If calibration stops, preserve its partial directory and use a new directory; pilot resumes in its existing directory.\n\nGates 1–3: not reached.\n',[record])
        shutil.copyfile(DOCS/'HANDOFF.md',Path(a.handoff_copy))
    def execute(label,command):
        nonlocal child,stage
        stage=label;before=set((ROOT/'results').iterdir())
        child=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        run.record('processes',{'stage':label,'pid':child.pid,'command':command})
        if not worker_started:status('In progress')
        run.log(f'start {label} pid={child.pid}')
        code=child.wait();child=None
        run.record('processes',{'stage':label,'returncode':code})
        if code:raise RuntimeError(f'{label} failed with exit {code}')
        return set((ROOT/'results').iterdir())-before
    def write_report(design_path,timing_path=None):
        d=json.loads(design_path.read_text())
        rows=[['Raw tickets',d['raw_tickets']['protectai'],d['raw_tickets']['judge'],d['raw_tickets']['either'],d['raw_tickets']['n']],
              ['Benign tickets',d['benign_tickets']['protectai'],d['benign_tickets']['judge'],d['benign_tickets']['either'],d['benign_tickets']['n']],
              ['Benign reports',d['benign_reports']['protectai'],d['benign_reports']['judge'],'ProtectAI only',d['benign_reports']['n']]]
        body='# User-fixed pre-pilot v3 revision\n\n'+table(['Surface','ProtectAI flags','Judge flags','OR flags','n'],rows)
        body+='\n\nJudge benign-report flags containing device requests: '+str(d['judge_benign_reports_with_device_requests'])+'. This is paired replay of existing training/calibration score journals, not an independent model evaluation.\n\nSpec: `'+TWO_TIER_SPEC+'`. Tickets use OR with ProtectAI first and positives short-circuited. Non-ticket messages use ProtectAI alone. Threshold 0.5 on both tiers. One screened message is one budget unit; each executed tier logs logical/physical seconds and judge token counts.\n\n'+d['judge_caveat']
        body+='\n\nAgent-reworded attacks are largely undetectable by either screened setup on the pre-pilot calibration evidence: both screen reports/notices using ProtectAI. Attack-derived C3 requests look like legitimate requests. Downstream containment therefore relies on lineage holds, whose effectiveness is tested by the pilot. The judge is excluded from reports because of its benign-request false positives.\n\nV2 partial calibration is preserved and excluded from fitted rates. V1/v2 tags and all existing results are preserved. Source reuse limits, clustered intervals, and no coverage doubling carry forward.\n\nRegression command: `python -m pytest -q`.\n\n```text\n'+test_output.strip()+'\n```'
        sources=[design_path,test_run/'metrics.json']
        if timing_path:
            t=json.loads(timing_path.read_text())
            body+='\n\nV3 timing projection:\n\n'+table(['Metric','Hours'],[(k,t[k]) for k in ['estimated_calibration_plus_pilot_hours','estimated_probe_hours','estimated_total_with_probe_hours','estimated_calibration_plus_pilot_uncached_hours','v3_total_slow_screen_mix_hours']])
            body+='\n\n'+t['interpretation'];sources.append(timing_path)
        else:body+='\n\nFresh real v3 timing probe and projection: pending.'
        report('07_v3_revision.md',body,sources)
    try:
        status('In progress: v2 stopped; preparing v3 design evidence')
        source=ROOT/'results/20261104-145902_detector_selection_v2'
        lib=load_library(ROOT,cfg['attacks']);contexts={f'ctx{i:03d}' for i in range(*cfg['split']['calibration_contexts'])}
        rows,design=replay_tiers(read_jsonl(source/'scores.jsonl'),read_jsonl(source/'corpus.jsonl'),set(lib['split']['train']),contexts)
        design.update(source_scores=str(source/'scores.jsonl'),source_sha256=sha(source/'scores.jsonl'),corpus_sha256=sha(source/'corpus.jsonl'),
                      regression={'passed':int(re.search(r'(\d+) passed',test_output).group(1)), 'output':str(test_run/'pytest.txt'),'sha256':sha(test_run/'pytest.txt')})
        for row in rows:run.record('replayed_scores',row)
        run.save_json('design_evidence.json',design);design_path=run.dir/'design_evidence.json'
        write_report(design_path)
        commit('phase: record paired training-only evidence for v3 routing',[DOCS/'07_v3_revision.md',DOCS/'ledger.csv'])
        dirs=execute('v3 calibration-only timing probe',['caffeinate','-dimsu',sys.executable,'-u','scripts/run_pilot.py',
            '--backend','mlx','--model','llama-4bit','--split','calibration','--limit','2','--no-adaptive','--name','timing_v3','--keep-prompts'])
        probe=select_owned_run(dirs,'timing_v3_llama-4bit')
        base=timing_analysis(probe)
        observed=[obs for row in read_jsonl(probe/'episodes.jsonl') for obs in row['cost'].get('screen_call_measurements',[])]
        if base['validation']['budget_violations'] or base['validation']['credit_violations']:
            raise RuntimeError('v3 timing probe violated message budget')
        if any(row['split']!='train' or row['context'] not in range(*cfg['split']['calibration_contexts']) for row in read_jsonl(probe/'episodes.jsonl')):
            raise RuntimeError('timing probe left the calibration/training split')
        def projection():
            cells=len(lib['families']);units=cfg['calibration']['units_per_cell'];p=cfg['pilot'];ad=p['adaptive']
            adaptive=len(ad['families'])*ad['episodes_per_family'] if ad['enabled'] else 0
            base.update(calibration_call_upper_bound=(6*cells+4)*units,
                v3_calibration_screen_upper_bound=(9*cells+16)*units,
                estimated_pilot_episodes=cells*len(p['goals'])*p['episodes_per_cell']+p['benign_episodes']+adaptive,
                pilot=p,source_probe=str(probe))
            return project_timing(base,cfg,observed)
        projected=projection()
        reduced=projected['estimated_total_with_probe_hours']>72
        if reduced:
            cfg['pilot'].update(episodes_per_cell=4,benign_episodes=20);projected=projection()
        projected['predefined_size_reduction_applied']=reduced
        timing_path=evidence('v3_timing',projected);write_report(design_path,timing_path)
        if projected['estimated_total_with_probe_hours']>72:raise RuntimeError('v3 projection exceeds 72 h after predefined size reduction')
        if reduced:
            p=ROOT/'configs/default.yaml';text=p.read_text()
            text=re.sub(r'(?m)^  episodes_per_cell:.*$','  episodes_per_cell: 4',text,count=1)
            text=re.sub(r'(?m)^  benign_episodes:.*$','  benign_episodes: 20',text,count=1);p.write_text(text)
        with (ROOT/'docs/design_changes.md').open('a') as f:
            f.write('\n\nV3 evidence and timing are recorded before freezing: `'+str(design_path)+'` (SHA256 `'+sha(design_path)+'`); `'+str(timing_path)+'`. The user-fixed kind rule supersedes the v2 pooled selection. No held-out families or evaluation buildings were used to choose or time the detector.\n')
        commit('phase: record measured v3 timing and freeze inputs',[ROOT/'configs/default.yaml',ROOT/'docs/design_changes.md',DOCS/'07_v3_revision.md',DOCS/'ledger.csv'])
        stage='freezing v3';manifest=freeze(timing_path,'frozen-pilot-v3',design_path)
        run.log('v3 freeze '+str(manifest));worker_started=True
        execute('frozen v3 calibration, pilot and prediction',['caffeinate','-dimsu',sys.executable,'-u','scripts/workflow_after_freeze.py',
            '--freeze',str(manifest),'--timing',str(timing_path),'--handoff-copy',a.handoff_copy])
        run.save_json('metrics.json',{'status':'complete','design':str(design_path),'timing':str(timing_path),'freeze':str(manifest)})
        return 0
    except BaseException as error:
        run.record('errors',{'error':repr(error),'traceback':traceback.format_exc()});run.log('stopped: '+repr(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if not worker_started:status('Stopped: '+repr(error))
        run.save_json('metrics.json',{'status':'stopped','stage':stage,'error':repr(error)})
        return 1
    finally:run.close()
if __name__=='__main__':sys.exit(main())
