"""Repair run ownership after the current controller/model exit, then continue.

This maintenance job never loads a model. It waits for the existing model process
group before tests or source changes, preserves the original freeze, and records a
controller-only erratum without changing experimental settings.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def alive(pid):
    result=subprocess.run(['ps','-p',str(pid),'-o','stat='],capture_output=True,text=True)
    return result.returncode==0 and result.stdout.strip() and not result.stdout.lstrip().startswith('Z')

def group_alive(pgid):
    try:os.killpg(pgid,0);return True
    except ProcessLookupError:return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--controller',type=int,required=True)
    parser.add_argument('--model-group',type=int,required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--freeze',required=True)
    parser.add_argument('--timing',required=True)
    parser.add_argument('--record-dir',required=True)
    args=parser.parse_args();record_dir=Path(args.record_dir)
    print('Waiting for current controller and model group; no tests or model jobs are started.',flush=True)
    while alive(args.controller) or group_alive(args.model_group):
        if (ROOT/'data/STOP_WORKFLOW').exists():print('Cancelled by STOP_WORKFLOW marker.',flush=True);return 2
        time.sleep(5)
    calibration=Path(args.calibration)
    if not (calibration/'calibration.json').exists():
        with (record_dir/'repair_stopped.json').open('x') as f:
            json.dump({'reason':'Current calibration did not complete; preserving its partial directory. No recovery model job started.','calibration':str(calibration)},f,indent=2)
        print('Calibration incomplete; preserving evidence and stopping.',flush=True);return 1
    from iotpi.runlog import Run
    repair=Run(ROOT,'controller_discovery_repair',{'original_freeze':args.freeze,'completed_calibration':str(calibration)})
    path=ROOT/'scripts/workflow_after_freeze.py';source=path.read_text()
    helper='''def select_owned_run(candidates, expected_name, known=None):
    """Identify this command's artifact; unrelated result directories are allowed."""
    candidates = ([Path(known)] if known is not None else []) + list(candidates)
    matches = set()
    for directory in candidates:
        meta = directory / 'meta.json'
        if not directory.is_dir() or not meta.exists():
            continue
        try:
            name = json.loads(meta.read_text()).get('name')
        except json.JSONDecodeError:
            continue
        if name == expected_name:
            matches.add(directory)
    if known is not None and Path(known) in matches:
        return Path(known)
    if len(matches) != 1:
        raise RuntimeError(f'cannot identify owned run {expected_name}: {sorted(map(str,matches))}')
    return next(iter(matches))


'''
    assert 'def select_owned_run(' not in source
    source=source.replace('def main(argv=None):',helper+'def main(argv=None):',1)
    needle="        before=set((ROOT/'results').iterdir())"
    assert needle in source
    source=source.replace(needle,needle+'''
        run_name = arguments[arguments.index('--name')+1] if '--name' in arguments else 'quarantine_probe_llama'
        expected_name = run_name + ('_' + arguments[arguments.index('--model')+1]
                                    if script in ['run_calibration.py','run_pilot.py'] else '')''',1)
    before='''            early=[p for p in set((ROOT/'results').iterdir())-before if p.is_dir() and (p/'meta.json').exists()]
            if len(early)==1:
                paths[stage]=early[0];active['directory']=str(early[0]);break'''
    after='''            try:
                owned=select_owned_run(set((ROOT/'results').iterdir())-before, expected_name)
            except RuntimeError:
                owned=None
            if owned is not None:
                paths[stage]=owned;active['directory']=str(owned);break'''
    assert before in source;source=source.replace(before,after,1)
    start=source.index('        added=[p for p in set((ROOT/\'results\').iterdir())-before')
    end=source.index('        if return_code and stage==\'pilot\'',start)
    source=source[:start]+'''        directory=select_owned_run(set((ROOT/'results').iterdir())-before, expected_name, paths.get(stage))
        paths[stage]=directory
'''+source[end:]
    source=source.replace("'- Original baseline mock artifacts predate complete calibration journals and are preserved as wiring evidence, excluded from research results.']",
        "'- Original baseline mock artifacts predate complete calibration journals and are preserved as wiring evidence, excluded from research results.',\n                '- A controller-only artifact ownership repair was tested after the initial calibration; the completed calibration and original frozen experimental settings were retained.']")
    path.write_text(source)
    test=ROOT/'tests/test_controller_discovery.py'
    assert not test.exists()
    test.write_text('''import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from workflow_after_freeze import select_owned_run

def test_controller_ignores_unrelated_analysis_directories(tmp_path):
    def artifact(folder,name):
        p=tmp_path/folder;p.mkdir();(p/'meta.json').write_text(json.dumps({'name':name}));return p
    calibration=artifact('calibration','cal_llama_llama-4bit')
    unrelated=artifact('analysis','handoff_restart_settings')
    snapshot=artifact('snapshot','handoff_snapshot')
    candidates={calibration,unrelated,snapshot}
    assert select_owned_run(candidates,'cal_llama_llama-4bit')==calibration
    assert select_owned_run(candidates,'cal_llama_llama-4bit',unrelated)==calibration
    assert select_owned_run(set(),'cal_llama_llama-4bit',calibration)==calibration
    with pytest.raises(RuntimeError):select_owned_run({unrelated,snapshot},'missing')
''')
    note='\n\nController-only erratum: artifact discovery now selects the launched experiment by its exact metadata name and recorded directory. An independently created bookkeeping result directory exposed the old directory-count assumption. The repair is applied after calibration and its controller exit, with a regression test. Configuration, task prompts, attack templates, conditions, budgets, sample plan, and statistical calculations are unchanged. The original freeze manifest is preserved; a supplemental manifest records the repaired controller hash.\n'
    with (ROOT/'docs/design_changes.md').open('a') as f:f.write(note)
    result=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,capture_output=True,text=True)
    (repair.dir/'pytest.txt').write_text(result.stdout+result.stderr)
    print(result.stdout+result.stderr,flush=True)
    if result.returncode:
        repair.save_json('metrics.json',{'tests_exit_code':result.returncode,'status':'repair not committed; no model job started'})
        repair.close();return result.returncode
    subprocess.run(['git','add','--','scripts/workflow_after_freeze.py','scripts/repair_controller.py','tests/test_controller_discovery.py','docs/design_changes.md'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','fix(workflow): identify owned runs amid unrelated artifacts [audit n-directory]'],cwd=ROOT,check=True)
    fix_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest=json.loads(Path(args.freeze).read_text())
    original=manifest['protected_files']
    for name,digest in original.items():
        if name!='scripts/workflow_after_freeze.py' and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen input changed unexpectedly: '+name)
    manifest['controller_erratum']={'original_manifest':args.freeze,'fix_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                                  'reason':'Select experiment-owned run metadata rather than count all new result directories.'}
    manifest['protected_files']['scripts/workflow_after_freeze.py']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['protected_files']['scripts/repair_controller.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    repair.save_json('manifest.json',manifest)
    repair.save_json('metrics.json',{'tests_exit_code':0,'status':'controller repaired; experimental settings unchanged','completed_calibration':str(calibration)})
    repair.close()
    with (ROOT/'docs/reports/FROZEN.md').open('a') as f:
        f.write(note+'Controller supplement: `'+str(repair.dir/'manifest.json')+'`. Regression output: `'+str(repair.dir/'pytest.txt')+'`.\n')
    with (ROOT/'docs/reports/01_audit.md').open('a') as f:
        f.write('\n\nController-only maintenance after initial calibration: run discovery now ignores unrelated result directories. Fix: `'+fix_commit+'`; regression: `test_controller_ignores_unrelated_analysis_directories`. Original experiment settings and calibration are retained.\n\n```text\n'+(result.stdout+result.stderr).strip()+'\n```\n\nSource: `'+str(repair.dir/'pytest.txt')+'`.\n')
    pytest_file=repair.dir/'pytest.txt'
    with (ROOT/'docs/reports/ledger.csv').open('a',newline='') as f:
        writer=csv.writer(f)
        for statement,pattern in [('Controller repair passed tests',r'(\d+) passed'),('Controller repair test seconds',r'in ([\d.]+)s')]:
            match=re.search(pattern,result.stdout+result.stderr)
            if match:
                query='python -c '+repr('import re;from pathlib import Path;print(re.search('+repr(pattern)+',Path('+repr(str(pytest_file))+').read_text()).group(1))')
                writer.writerow(['01_audit.md',statement,match.group(1),str(pytest_file.relative_to(ROOT)),query,hashlib.sha256(pytest_file.read_bytes()).hexdigest()])
    subprocess.run(['git','add','--','docs/reports/FROZEN.md','docs/reports/01_audit.md','docs/reports/ledger.csv'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','docs: record controller-only freeze erratum'],cwd=ROOT,check=True)
    command=['nohup','caffeinate','-dimsu',sys.executable,'-u',str(path),'--freeze',str(repair.dir/'manifest.json'),
             '--timing',args.timing,'--calibration',str(calibration)]
    with (record_dir/'continued_workflow.log').open('x') as output:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    with (record_dir/'controller_continued.json').open('x') as f:
        json.dump({'pid':child.pid,'command':command,'log':str(record_dir/'continued_workflow.log'),'manifest':str(repair.dir/'manifest.json')},f,indent=2)
    print('Continued repaired controller pid='+str(child.pid),flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
