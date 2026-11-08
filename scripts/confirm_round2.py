"""Detached, serial, resumable Phase C. Only health metrics until BOTH models finish."""
from __future__ import annotations
import argparse,fcntl,json,os,re,shlex,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iotpi.runlog import Run
from round2_support import DOCS,report,table,sha
from round2_confirm_support import MODELS,MANIFEST,verify_frozen,inspect_run,records,experiment_processes


def owned_directory(results,name):
    matches=[]
    for path in Path(results).iterdir():
        meta=path/'meta.json'
        if meta.exists():
            try:
                if json.loads(meta.read_text()).get('name')==name:matches.append(path)
            except json.JSONDecodeError:continue
    if len(matches)>1:raise RuntimeError('ambiguous owned model directories')
    return matches[0] if matches else None


def serial_models(execute):
    """execute must return only after its child exits and completion is validated."""
    result={}
    for model in MODELS:result[model]=execute(model)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--resume');parser.add_argument('--tests')
    parser.add_argument('--confirmatory-authorized',action='store_true')
    parser.add_argument('--handoff-copy',default='docs/reports/round2/HANDOFF-copy.md')
    a=parser.parse_args(argv)
    if not a.confirmatory_authorized:raise PermissionError('explicit user go round 2 is required')
    lock=open('/tmp/iotpi-round2-confirmatory.lock','a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('another Phase-C controller holds the lock')
    if experiment_processes():raise RuntimeError('an experiment process is already alive; do not launch another')
    manifest,cfg=verify_frozen()
    if a.resume:
        folder=Path(a.resume);settings=__import__('yaml').safe_load((folder/'config.yaml').read_text())
        run=Run.reopen(folder)
    else:
        if not a.tests:raise ValueError('--tests results directory is required for a fresh controller')
        tests=Path(a.tests).resolve();testdata=json.loads((tests/'metrics.json').read_text())
        if testdata['returncode']:raise RuntimeError('regression tests did not pass')
        settings={'seed':13,'prereg_tag':'prereg-round2','prereg_commit':manifest['git_commit'],
            'authorization':'User: go round 2 — run Phase C exactly as pre-registered',
            'tests':str(tests),'handoff_copy':a.handoff_copy,
            'infrastructure_hashes':{p:sha(ROOT/p) for p in ('scripts/confirm_round2.py','scripts/analyze_round2.py','scripts/round2_confirm_support.py','tests/test_round2_phasec.py')}}
        run=Run(ROOT,'round2_confirmatory_workflow',settings)
        shutil.copyfile(DOCS/'HANDOFF.md',run.dir/'HANDOFF_before_phaseC.md')
        run.save_json('frozen_manifest.json',manifest)
    def verify():
        verify_frozen()
        for path,digest in settings['infrastructure_hashes'].items():
            if sha(ROOT/path)!=digest:raise RuntimeError('Phase-C infrastructure changed during execution: '+path)
    verify()
    paths={row['model']:Path(row['directory']) for row in records(run.dir/'steps.jsonl')}
    active=None;stage='start';analysis_dir=None
    historical=(run.dir/'HANDOFF_before_phaseC.md').read_text()
    historical=re.sub(r'^Sources:.*\n?','',historical,flags=re.M)
    historical=historical.replace('# Round 2 handoff','# Phase A/B historical record',1)
    def handoff(status):
        health={m:inspect_run(p,m,cfg) for m,p in paths.items()}
        evidence={'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'status':status,'stage':stage,
            'controller_pid':os.getpid(),'active_pid':active.pid if active else None,
            'paths':{m:str(p) for m,p in paths.items()},'health':health,
            'analysis_directory':str(analysis_dir) if analysis_dir else None}
        name=f'status_{len(list(run.dir.glob("status_*.json")))}.json';run.save_json(name,evidence)
        body='# Round 2 handoff\n\n'+status+'\n\nPhase C is authorized by the user. Registered tag: `prereg-round2`; commit: `'+manifest['git_commit']+'`. Frozen experiment code, prompts and configuration are unchanged.\n\n'
        body+='Stage: '+stage+'. Controller PID: `'+str(os.getpid())+'`; active caffeinate PID: `'+str(active.pid if active else None)+'`. Controller directory: `'+str(run.dir)+'`. Logs: `run.log`, `controller.nohup.log`, and per-model process logs in that directory.\n\n'
        body+='Llama runs first, Qwen second, never concurrently. Completed rows, seeds, prompt snapshots and the same model cache are reused after interruption. Only health metrics are inspected while either model is running. Primary analysis waits for both completion markers and full-data validation; secondary tables follow saved primary results. Parse failures remain in the data.\n\n'
        body+=table(['Model','Rows','Expected','Parse failures','Budget violations'],[(m,h['rows'],h['expected_rows'],h['parse_failures'],h['budget_violations']) for m,h in health.items()])
        body+='\n\nModel directories:\n\n'+'\n'.join('- '+m+': `'+str(p)+'`' for m,p in paths.items())
        if analysis_dir:body+='\n\nFinal analysis directory: `'+str(analysis_dir)+'`. Report: `docs/reports/round2/09_confirmatory.md`. Sources and hashes are in the analysis directory.\n'
        command=[str(ROOT/'.venv/bin/python'),'-u','scripts/confirm_round2.py','--resume',str(run.dir),'--confirmatory-authorized']
        body+='\n\nIf the controller has exited, resume after confirming its old child is also stopped: `caffeinate -dimsu '+shlex.join(command)+'`. Never launch a second controller beside a surviving model child.\n\n'
        body+='The pre-run two-model timing projection is '+str(manifest['timing']['total_hours'])+' hours; actual duration depends on cache reuse and generated output. It is a projection, not a deadline.\n\n'
        if analysis_dir:
            p=json.loads((analysis_dir/'primary.json').read_text())
            body+='Confirmatory decisions: '+', '.join(m+' win='+str(p['models'][m]['wins']) for m in MODELS)+'; cross-model win='+str(p['cross_model_win'])+'.\n\n'
        body+='The following record describes the completed A/B stage before the user authorized Phase C. Its STOP statements are historical.\n\n'+historical
        sources=[run.dir/name,MANIFEST,Path(settings['tests'])/'metrics.json']
        if analysis_dir:sources.extend([analysis_dir/'primary.json',analysis_dir/'secondary.json',analysis_dir/'sources.json'])
        report('HANDOFF.md',body,sources)
        target=Path(settings['handoff_copy']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(DOCS/'HANDOFF.md',target)
    def stop(signum,frame):raise KeyboardInterrupt('controller signal '+str(signum))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def execute(model):
        nonlocal active,stage
        stage=model;verify()
        if experiment_processes():raise RuntimeError('another model process is alive')
        name='round2c_'+run.dir.name.split('_',1)[0]
        expected_name=name+'_confirm_'+model
        directory=paths.get(model) or owned_directory(ROOT/'results',expected_name)
        if directory:
            paths[model]=directory
            if (directory/'metrics.json').exists():inspect_run(directory,model,cfg,complete=True);return directory
            inspect_run(directory,model,cfg)
        for attempt in range(2):
            args=['--resume',str(directory)] if directory else ['--mode','confirm','--model',model,'--name',name]
            command=['caffeinate','-dimsu',sys.executable,'-u',str(ROOT/'scripts/run_round2.py'),*args,'--confirmatory-authorized']
            log=run.dir/f'{model}_{len(list(run.dir.glob(model+"_*.log")))}.log'
            with log.open('x') as output:
                active=subprocess.Popen(command,cwd=ROOT,start_new_session=True,stdout=output,stderr=subprocess.STDOUT)
                run.record('processes',{'model':model,'pid':active.pid,'command':command,'log':str(log)})
                run.log(f'{model} launched caffeinate pid={active.pid}')
                deadline=time.monotonic()+45
                while directory is None and time.monotonic()<deadline:
                    directory=owned_directory(ROOT/'results',expected_name)
                    if directory is not None or active.poll() is not None:break
                    time.sleep(.2)
                if directory:
                    paths[model]=directory;run.record('steps',{'model':model,'directory':str(directory)})
                handoff('Running; only health metrics available')
                while active.poll() is None:
                    try:active.wait(timeout=60)
                    except subprocess.TimeoutExpired:handoff('Running; only health metrics available')
                code=active.wait();active=None
            run.record('exits',{'model':model,'returncode':code,'log':str(log)})
            if code==0 and directory is not None:
                inspect_run(directory,model,cfg,complete=True);handoff('Model complete; next stage pending');return directory
            error=log.read_text()
            if 'MemoryError' in error or 'GB available' in error:
                raise RuntimeError(model+' stopped at the unchanged memory guard; free memory and resume the same directory. Log: '+str(log))
            if directory is None or attempt:raise RuntimeError(model+' failed; retained partial results. Log: '+str(log))
            run.log(model+' interrupted; resuming saved rows/cache once, without design changes')
        raise RuntimeError('unreachable')
    try:
        handoff('Phase C controller started')
        paths.update(serial_models(execute))
        stage='primary_then_secondary';verify();handoff('Both models complete; post-completion analysis starts')
        from analyze_round2 import analyze
        previous=list(records(run.dir/'analysis.jsonl'))
        analysis_dir=analyze(paths['llama-4bit'],paths['qwen-4bit'],previous[-1]['directory'] if previous else None,
            on_created=lambda directory:run.record('analysis',{'directory':str(directory)}))
        run.record('analysis',{'directory':str(analysis_dir)})
        stage='complete';verify();handoff('Phase C complete: registered primary, then descriptive secondary analysis and report written')
        report_target=Path(settings['handoff_copy']).parent/'09_confirmatory.md'
        shutil.copyfile(DOCS/'09_confirmatory.md',report_target)
        if not (run.dir/'metrics.json').exists():run.save_json('metrics.json',{'complete':True,'analysis_directory':str(analysis_dir),'models':{m:str(p) for m,p in paths.items()}})
        run.log('Phase C complete')
    except BaseException as exc:
        if active and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=45)
            except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
            active=None
        stage='interrupted';run.log(traceback.format_exc());handoff('Stopped; preserve partial artifacts and resume: '+str(exc));raise
    finally:run.close();lock.close()
if __name__=='__main__':main()
