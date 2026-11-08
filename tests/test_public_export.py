import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import reproduce_results as public


def test_data_manifest_rejects_modified_input(tmp_path, monkeypatch):
    monkeypatch.setattr(public, 'ROOT', tmp_path)
    (tmp_path / 'data/provenance').mkdir(parents=True)
    data = tmp_path / 'data/example.json'
    data.write_text('{"count": 1}\n')
    entry = {'path': 'data/example.json', 'export_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}
    (tmp_path / 'data/provenance/exported_files.json').write_text(json.dumps([entry]))
    assert public.validate_exports() == 1
    data.write_text('{"count": 2}\n')
    with pytest.raises(ValueError, match='digest mismatch'):
        public.validate_exports()


def test_data_manifest_rejects_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(public, 'ROOT', tmp_path)
    (tmp_path / 'data/provenance').mkdir(parents=True)
    (tmp_path / 'data/provenance/exported_files.json').write_text(json.dumps([
        {'path': '../outside.json', 'export_sha256': 'irrelevant'}]))
    with pytest.raises(ValueError, match='Invalid or duplicate'):
        public.validate_exports()


def test_complete_archived_reanalysis():
    receipt = public.reproduce()
    assert receipt['primary_matches_archive'] and receipt['secondary_matches_archive']
    assert receipt['cross_model_win'] is False
    assert receipt['model_wins'] == {'llama-4bit': True, 'qwen-4bit': False}
    assert {v['rows'] for v in receipt['health'].values()} == {1920}
