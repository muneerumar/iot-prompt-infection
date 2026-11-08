import json,sys
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


def test_resume_owns_existing_directory_without_model_or_name_override(tmp_path):
    from workflow_after_freeze import owned_run_identity
    (tmp_path/'meta.json').write_text(json.dumps({'name':'cal_llama_v3_llama-4bit'}))
    name,known=owned_run_identity('run_calibration.py',['--resume',str(tmp_path)])
    assert name=='cal_llama_v3_llama-4bit' and known==tmp_path
    assert select_owned_run(set(),name,known)==tmp_path
    assert owned_run_identity('run_calibration.py',['--backend','mlx','--model','llama-4bit','--name','cal_llama_v3'])==(name,None)
