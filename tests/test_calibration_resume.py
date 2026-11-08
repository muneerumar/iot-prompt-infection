import copy,json,sys,hashlib
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def bytes_by_file(directory):
    return {p.name:p.read_bytes() for p in directory.iterdir() if p.is_file()}


def test_resume_skips_completed_units_and_preserves_seeds_counts_and_bytes(tmp_path,monkeypatch):
    import run_calibration,recompute_run
    from iotpi.backends import MockBackend
    from iotpi.runlog import read_jsonl
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path))
    full=run_calibration.main(['--backend','mock','--screen','heuristic','--units','4','--name','complete'])
    original=MockBackend.generate;calls=[]
    def crash(self,messages,**kw):
        if len(calls)==9:raise RuntimeError('interruption fixture')
        calls.append(kw['seed']);return original(self,messages,**kw)
    monkeypatch.setattr(MockBackend,'generate',crash)
    with pytest.raises(RuntimeError,match='interruption fixture'):
        run_calibration.main(['--backend','mock','--screen','heuristic','--units','4','--name','interrupted'])
    partial=next(p for p in tmp_path.iterdir() if 'interrupted' in p.name)
    old=bytes_by_file(partial)
    units=read_jsonl(partial/'units.jsonl')
    completed_gens=[r for u in units for stream,r in u['records'] if stream=='generations']
    completed_seeds=[r['generation_seed'] for r in completed_gens]
    assert completed_seeds==calls[:len(completed_seeds)]
    resumed_calls=[]
    def count(self,messages,**kw):
        resumed_calls.append(kw['seed']);return original(self,messages,**kw)
    monkeypatch.setattr(MockBackend,'generate',count)
    assert run_calibration.main(['--resume',str(partial)])==partial
    for name,data in old.items():assert (partial/name).read_bytes().startswith(data)
    full_gens=read_jsonl(full/'generations.jsonl');actual=read_jsonl(partial/'generations.jsonl')
    signature=lambda rows:[(r['generation_id'],r['generation_seed'],r['context'],r['episode_seed'],r['output']) for r in rows]
    assert signature(actual)==signature(full_gens)
    assert resumed_calls==[r['generation_seed'] for r in full_gens[len(completed_gens):]]
    assert json.loads((partial/'calibration.json').read_text())==json.loads((full/'calibration.json').read_text())
    keys=[tuple(r['unit_key']) for r in read_jsonl(partial/'units.jsonl')]
    assert len(keys)==len(set(keys))
    assert recompute_run.main(['--run',str(partial)])==0
    final=bytes_by_file(partial)
    assert run_calibration.main(['--resume',str(partial)])==partial
    assert bytes_by_file(partial)==final


def test_projection_recovery_appends_after_torn_write_without_duplicate_ids(tmp_path):
    from iotpi.runlog import Run,read_jsonl
    from iotpi.calibration_resume import CalibrationJournal
    raw=Run(tmp_path,'projection',{})
    j=CalibrationJournal(raw);j.begin_unit('gateway_raw','authority',0)
    j.record('agent_measurements',{'key':['gateway','carry_raw','authority'],'hit':True})
    record=raw.record
    def fail(stream,row):
        if stream=='agent_measurements':raise RuntimeError('after commit')
        return record(stream,row)
    raw.record=fail
    with pytest.raises(RuntimeError,match='after commit'):j.record('units',{'unit':'gateway_raw','family':'authority'})
    directory=j.dir;j.close()
    with (directory/'agent_measurements.jsonl').open('a') as f:f.write('{"torn":')
    prefix=(directory/'agent_measurements.jsonl').read_bytes()
    with pytest.warns(RuntimeWarning):j=CalibrationJournal(Run.reopen(directory))
    assert (directory/'agent_measurements.jsonl').read_bytes().startswith(prefix)
    with pytest.warns(RuntimeWarning):rows=read_jsonl(directory/'agent_measurements.jsonl')
    assert len(rows)==1 and rows[0]['hit'];j.close()
    with pytest.warns(RuntimeWarning):j=CalibrationJournal(Run.reopen(directory))
    with pytest.warns(RuntimeWarning):assert len(read_jsonl(directory/'agent_measurements.jsonl'))==1
    j.close()


def test_legacy_import_leaves_source_untouched_and_reuses_outputs(tmp_path,monkeypatch):
    import run_calibration
    from iotpi.runlog import read_jsonl
    from iotpi.backends import MockBackend
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path))
    full=run_calibration.main(['--backend','mock','--screen','heuristic','--units','3','--name','original'])
    legacy=tmp_path/'legacy';legacy.mkdir()
    for file in ['config.yaml','meta.json','pip_freeze.txt']:(legacy/file).write_bytes((full/file).read_bytes())
    units=read_jsonl(full/'units.jsonl')
    keep=[u for u in units if u['unit']!='benign'][:8]
    records={}
    for u in keep:
        for stream,row in u['records']:records.setdefault(stream,[]).append(row)
    (legacy/'units.jsonl').write_text(''.join(json.dumps(u['summary'])+'\n' for u in keep))
    for stream,rows in records.items():
        (legacy/f'{stream}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    before=bytes_by_file(legacy)
    original=MockBackend.generate;new_calls=[]
    def count(self,messages,**kw):new_calls.append(kw['seed']);return original(self,messages,**kw)
    monkeypatch.setattr(MockBackend,'generate',count)
    recovered=run_calibration.main(['--resume-from',str(legacy),'--name','recovered'])
    assert bytes_by_file(legacy)==before
    assert json.loads((recovered/'calibration.json').read_text())==json.loads((full/'calibration.json').read_text())
    assert len(new_calls)==len(read_jsonl(full/'generations.jsonl'))-len(records['generations'])
    assert json.loads((recovered/'resume_import.json').read_text())['completed_units_imported']==8


def test_resume_rejects_concurrent_ownership(tmp_path,monkeypatch):
    from iotpi.runlog import Run
    from iotpi.calibration_resume import CalibrationJournal
    raw=Run(tmp_path,'locked',{})
    j=CalibrationJournal(raw)
    with pytest.raises(RuntimeError,match='owned'):
        CalibrationJournal(Run.reopen(j.dir))
    j.close()


def test_resume_rejects_changed_inputs_before_any_model_call(tmp_path,monkeypatch):
    import run_calibration
    from iotpi.backends import MockBackend
    monkeypatch.setenv('IOTPI_RESULTS_DIR',str(tmp_path))
    def crash(*args,**kwargs):raise RuntimeError('stop before completion')
    monkeypatch.setattr(MockBackend,'generate',crash)
    with pytest.raises(RuntimeError):run_calibration.main(['--backend','mock','--screen','heuristic','--units','2'])
    directory=next(tmp_path.iterdir());before=bytes_by_file(directory)
    class Changed:
        def hexdigest(self):return 'changed-input-hash'
    monkeypatch.setattr(run_calibration.hashlib,'sha256',lambda data:Changed())
    with pytest.raises(ValueError,match='inputs changed'):run_calibration.main(['--resume',str(directory)])
    assert bytes_by_file(directory)==before
