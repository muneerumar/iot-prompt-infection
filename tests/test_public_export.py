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
    public.compare_json(receipt['model_wins'], {'llama-4bit': True, 'qwen-4bit': False})
    assert set(receipt['max_abs_float_deviation']) == {'primary', 'secondary'}
    assert all(value >= 0 for value in receipt['max_abs_float_deviation'].values())
    assert {v['rows'] for v in receipt['health'].values()} == {1920}


def test_recursive_comparison_reports_maximum_float_deviation():
    a = {'values': [0.0, {'bound': -0.07 + 1e-13}, float('nan')], 'wins': False}
    b = {'wins': False, 'values': [1e-13, {'bound': -0.07}, float('nan')]}
    assert public.compare_json(a, b) == pytest.approx(1e-13, rel=1e-3)
    assert public.compare_json([1, 2.0, None, 'x'], [1.0, 2, None, 'x']) == 0
    assert public.compare_json([float('inf'), -float('inf')],
                               [float('inf'), -float('inf')]) == 0


@pytest.mark.parametrize('actual,archived', [
    (1, 2), (True, 1), (False, 0.0), (1.0 + 1e-13, 1),
    ('1', 1), (None, False), (float('nan'), 0.0), (1.0, 1.0 + 2e-9),
    ({'a': 1}, {'b': 1}), ([1], [1, 2]),
])
def test_recursive_comparison_rejects_discrete_structure_and_large_float_changes(actual, archived):
    with pytest.raises(ValueError, match=r'\$\["nested"\]\[0\].*actual=.*archived='):
        public.compare_json({'nested': [actual]}, {'nested': [archived]})


@pytest.mark.parametrize('decision', ['holds', 'wins', 'cross_model_win'])
def test_decisions_are_exact_even_beside_tolerable_float_noise(decision):
    with pytest.raises(ValueError, match=decision):
        public.compare_json({'mean': 0.1 + 1e-13, decision: True},
                            {'mean': 0.1, decision: False})
