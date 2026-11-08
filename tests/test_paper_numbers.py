"""Public-data regeneration checks; no manuscript artifacts are committed."""
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip('matplotlib')
ROOT = Path(__file__).resolve().parents[1]
HASHES = {
    'generated/numbers.tex': '40631202de70045d206cb2fe1874e8e485ff7f2bcab93e735cc5a057e9f54d9e',
    'generated/tab_calibration.tex': 'aab6a306a4ee6774eece60dce0eae60a5d15f6012101adc0a25ad36bac3258ed',
    'generated/tab_conditions.tex': '46a4cf7bc5fe72b4f8ad2d74088771e952a564f380249f84fb346af69c615b7f',
    'generated/tab_cost.tex': '868f7fb614c795f4a011e97ddcac5c6dcf912c4cad1afed736763dfb51bd2d71',
    'generated/tab_family.tex': '1e165dace40f74cc709036f938da63be6e8eaa916d57f8318ca28899bcaf76a6',
    'generated/tab_mechanism.tex': 'bfd67d72bb7048a69023917258e4a0e6f6538c0df56e02888edf9906eb3e4574',
    'generated/tab_pilot.tex': '3e77f84ac48973581479356b1d348db3955fcfe1d8f9fe291db3fda291d789a5',
    'generated/tab_primary.tex': '4e468f37c86dd6278f6cfb41809a32cb3b973e487c4fbabefec6eb6225a48876',
    'generated/tab_safety.tex': '196d02611e86b7e4e9112e2b507efd21afcacacffa8531cf2267f28fb9fc1e96',
}


def run_generator(*args):
    return subprocess.run([sys.executable, str(ROOT / 'scripts/paper/make_numbers.py'),
                           '--root', str(ROOT), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)


def test_public_data_regenerates_nine_exact_tex_files_and_three_figures(tmp_path):
    output = tmp_path / 'numbers'
    output.mkdir()  # An explicitly empty directory is also a valid destination.
    result = run_generator('--out', output)
    assert result.returncode == 0, result.stdout + result.stderr
    for relative, expected in HASHES.items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected, relative
    for name in ('fig_harm_cost.pdf', 'fig_utility.pdf', 'fig_false_drops.pdf'):
        figure = output / 'figures' / name
        assert figure.is_file() and figure.stat().st_size > 0


def test_generator_requires_output_and_preserves_nonempty_destination(tmp_path):
    assert run_generator().returncode == 2
    marker = tmp_path / 'existing.txt'
    marker.write_text('preserve this output')
    result = run_generator('--out', tmp_path)
    assert result.returncode == 2 and '--out must be a new or empty directory' in result.stderr
    assert list(tmp_path.iterdir()) == [marker]
    assert marker.read_text() == 'preserve this output'


def test_scenario_diagnostics_requires_withheld_input_and_preserves_saved_aggregate():
    saved = ROOT / 'data/paper/scenario_diagnostics.json'
    before = saved.read_bytes()
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/paper/scenario_diagnostics.py'),
                             '--root', str(ROOT)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and 'requires the withheld' in result.stderr
    assert saved.read_bytes() == before
