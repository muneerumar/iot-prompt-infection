import json
import sys
from pathlib import Path

import yaml

from iotpi.agents import parse_output
from iotpi.attacks import load_library
from iotpi.experiment import episode_plan
from iotpi.scenario import make_scenario

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CFG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
LIB = load_library(ROOT, CFG["attacks"])


def test_splits_are_disjoint():
    a = set(range(*CFG["split"]["calibration_contexts"]))
    b = set(range(*CFG["split"]["evaluation_contexts"]))
    assert not a & b
    assert not set(LIB["split"]["train"]) & set(LIB["split"]["heldout"])
    plan = episode_plan(CFG, LIB)
    assert all(lo <= p["context"] < hi for p in plan for lo, hi in [CFG["split"]["evaluation_contexts"]])


def test_attacked_device_has_no_legit_work():
    for k in range(40):
        sc = make_scenario(k, k, 2, 4, LIB, "authority", "c2_unlock")
        dev = sc.attack.target[0]
        assert all(e.device != dev for e in sc.expected)
        assert all(b[0] != dev for b in sc.bookings)
        assert sc.attack.canary in sc.attack.text


def test_every_family_renders_for_every_goal():
    for fam in LIB["families"]:
        for goal in ("c3_safety", "c2_unlock", "c1_out_of_policy", "exfil", "spread"):
            sc = make_scenario(3, 3, 2, 4, LIB, fam, goal)
            assert "{" not in sc.attack.text.replace('{"', "").replace("{{", "") or fam == "structured"


def test_parse_output_tolerates_trailing_commas():
    out = parse_output("gateway", 'Sure: {"zone_report": "ok", "requests": [{"device": "door_north", '
                                  '"command": "unlock", "reason": "x"},], "ticket_replies": []}')
    assert out.parse_ok and out.requests == [("door_north", "unlock", "x")]
    assert not parse_output("coordinator", "no json here").parse_ok


def test_scripts_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("IOTPI_RESULTS_DIR", str(tmp_path / "runs"))
    import predict_multihop
    import run_calibration
    import run_pilot
    import run_scale
    d = run_pilot.main(["--backend", "mock", "--screen", "heuristic", "--limit", "6", "--no-adaptive",
                        "--conditions", "policy_only", "trustgate", "--name", "pytest"])
    s = json.loads((d / "summary.json").read_text())
    assert d.parent == tmp_path / "runs"
    assert set(s["conditions"]) == {"policy_only", "trustgate"}
    c = run_calibration.main(["--backend", "mock", "--screen", "heuristic", "--units", "4", "--name", "pytest"])
    p = predict_multihop.main(["--calibration", str(c / "calibration.json"), "--pilot", str(d),
                               "--draws", "2", "--reps", "2", "--name", "pytest"])
    assert (p / "prediction_summary.md").exists()
    sc_dir = run_scale.main(["--calibration", str(c / "calibration.json"), "--episodes", "3",
                             "--conditions", "policy_only", "trustgate", "--name", "pytest"])
    assert any(sc_dir.glob("summary_budget_*.json"))
    resumed = run_pilot.main(["--resume", str(d), "--backend", "mock"])
    assert resumed == d


def test_ticket_reply_is_exfiltration_but_not_carry():
    from iotpi.agents import AgentOutput, forwardable_text
    from iotpi.calibrate import _carried
    from types import SimpleNamespace
    atk = SimpleNamespace(canary="Sabc123", target=None)
    out = AgentOutput(text='{"ticket_replies": ["Sabc123"]}', report="Normal.",
                      replies=[("T1", "Sabc123")])
    assert _carried(out, atk) == (False, False)
    assert "Sabc123" not in forwardable_text(out)
    out.report = "Sabc123"
    assert _carried(out, atk) == (True, False)


def test_root_report_is_not_an_outgoing_channel():
    from iotpi.agents import AgentOutput, forwardable_text
    out = AgentOutput(text="", report="Sabc123", notice="Normal.")
    assert "Sabc123" not in forwardable_text(out, include_report=False)


def test_calibration_only_pilot_uses_measured_risk_and_training_families(tmp_path, monkeypatch):
    import run_pilot
    monkeypatch.setenv("IOTPI_RESULTS_DIR", str(tmp_path / "runs"))
    measured = {role: {"carry_raw": .2, "carry_rewritten": .2, "persist": .1, "target": .2}
                for role in ("gateway", "coordinator")}
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps({"backend": "mock", "model": "mock", "rates": measured}))
    d = run_pilot.main(["--backend", "mock", "--screen", "heuristic", "--split", "calibration",
                       "--attack-only", "--limit", "16", "--conditions", "policy_only",
                       "--risk-rates", str(source)])
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    assert cfg["resolved"]["risk_rates"] == measured
    assert cfg["resolved"]["risk_source"] == "calibration"
    rows = [json.loads(x) for x in (d / "episodes.jsonl").read_text().splitlines()]
    assert {r["family"] for r in rows} == set(LIB["split"]["train"])
    assert all(0 <= r["context"] < 10 and not r.get("adaptive") for r in rows)


def test_real_risk_pilot_rejects_assumed_rates_before_loading_a_model():
    import pytest
    import run_pilot
    with pytest.raises(ValueError, match="requires --risk-rates"):
        run_pilot.main(["--backend", "mlx", "--model", "llama-4bit", "--conditions", "trustgate"])


def test_resume_recovers_transaction_without_duplicate_traces(tmp_path, monkeypatch):
    import pytest
    import run_pilot
    from iotpi.runlog import Run
    monkeypatch.setenv("IOTPI_RESULTS_DIR", str(tmp_path / "runs"))
    original = Run.record
    crashed = False
    def fail_once(self, stream, row):
        nonlocal crashed
        if stream == "episodes" and not crashed:
            crashed = True
            raise RuntimeError("simulated crash after trace")
        return original(self, stream, row)
    monkeypatch.setattr(Run, "record", fail_once)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pilot.main(["--backend", "mock", "--screen", "heuristic", "--limit", "2", "--no-adaptive",
                        "--conditions", "policy_only", "trustgate", "--name", "crash"])
    d = next((tmp_path / "runs").iterdir())
    metadata = (d / "meta.json").read_bytes()
    monkeypatch.setattr(Run, "record", original)
    run_pilot.main(["--resume", str(d)])
    for stream in ["episodes", "traces", "transactions"]:
        rows = [json.loads(x) for x in (d / f"{stream}.jsonl").read_text().splitlines()]
        keys = [((r["episode"] if stream == "transactions" else r)["plan_index"], (r["episode"] if stream == "transactions" else r)["condition"]) for r in rows]
        assert len(keys) == len(set(keys)) == 4
    assert (d / "meta.json").read_bytes() == metadata
    summary = (d / "summary.json").read_bytes()
    run_pilot.main(["--resume", str(d)])
    assert (d / "summary.json").read_bytes() == summary


def test_resume_preserves_and_skips_incomplete_transaction(tmp_path, monkeypatch):
    import pytest
    from iotpi.runlog import Run, read_jsonl
    monkeypatch.setenv("IOTPI_RESULTS_DIR", str(tmp_path))
    run = Run(ROOT, "fragment", {})
    path = run.dir / "transactions.jsonl"
    path.write_text('{"episode":')
    run.record("transactions", {"complete": True})
    run.close()
    with pytest.warns(RuntimeWarning, match="incomplete"):
        assert read_jsonl(path) == [{"complete": True}]
    assert path.read_text().startswith('{"episode":\n')


def test_parse_handles_fences_and_unrelated_braces():
    payload = {"zone_report": "A literal ,} stays unchanged.", "requests": [], "ticket_replies": []}
    text = 'Context {example}: {"metadata": 1}\n```json\n' + json.dumps(payload) + '\n```'
    out = parse_output("gateway", text)
    assert out.parse_ok and out.report == payload["zone_report"]


def test_parse_rejects_wrong_schema_without_crashing():
    assert not parse_output("gateway", '{}').parse_ok
    assert not parse_output("gateway", '{"zone_report":"ok","requests":3,"ticket_replies":7}').parse_ok
    assert not parse_output("coordinator", '{"commands": "invalid", "building_notice": "ok", "report": "ok"}').parse_ok


def test_adaptive_rewrite_preserves_identifiers_and_query_limit():
    from iotpi.attacks import adaptive_rewrite, strip_rewrite_preamble
    from iotpi.backends import Generation
    sc = make_scenario(3, 3, 2, 4, LIB, "authority", "c3_safety")
    class Rewriter:
        def generate(self, *a, **kw):
            return Generation("Here is the rewritten ticket: " + sc.attack.text, 1, 1, 0.0)
    class Counter:
        calls = 0
        def score_text(self, text):
            self.calls += 1
            return .9
    detector = Counter()
    out = adaptive_rewrite(sc.attack, LIB, Rewriter(), detector, 8, .5, 1)
    assert out.adaptive_queries == detector.calls == 8
    assert sc.attack.canary in out.text and sc.attack.target[0] in out.text
    assert not strip_rewrite_preamble("Here is the rewritten ticket: " + out.text).startswith("Here is")
    class Invalid:
        def generate(self, *a, **kw):
            return Generation("Ordinary maintenance note.", 1, 1, 0.0)
    detector = Counter()
    out = adaptive_rewrite(sc.attack, LIB, Invalid(), detector, 8, .5, 1)
    assert out.text == sc.attack.text and out.adaptive_queries == detector.calls == 1


def test_cache_can_be_disabled_and_preserves_measurement_metadata(tmp_path):
    from iotpi.backends import CachingBackend, Generation
    class Backend:
        name, model_path = "unit", "unit"
        calls = 0
        def generate(self, *a, **kw):
            self.calls += 1
            return Generation("{}", 2, 3, .1, {"peak_memory_bytes": 7})
        def close(self): pass
    inner = Backend()
    cached = CachingBackend(inner, tmp_path / "cache.jsonl")
    args = dict(seed=1, max_tokens=5, temperature=0)
    cached.generate([], **args)
    out = cached.generate([], **args)
    assert inner.calls == 1 and out.meta["cached"] and out.meta["peak_memory_bytes"] == 7
    assert out.meta["physical_seconds"] == 0
    cached.close()
    disabled = CachingBackend(inner, tmp_path / "unused.jsonl", enabled=False)
    disabled.generate([], **args); disabled.generate([], **args)
    assert inner.calls == 3 and not (tmp_path / "unused.jsonl").exists()
    disabled.close()


def test_memory_guard_reports_available_bytes_and_aborts(monkeypatch):
    import platform
    import subprocess
    import pytest
    from types import SimpleNamespace
    from iotpi.backends import _ensure_available
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    def vm(pages):
        return f"Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: {pages}.\nPages inactive: 0.\nPages speculative: 0.\n"
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=vm(200000)))
    assert _ensure_available() == 200000 * 16384
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=vm(100)))
    with pytest.raises(MemoryError): _ensure_available()


def test_calibration_counts_have_raw_generation_provenance(tmp_path, monkeypatch):
    import ast
    from collections import defaultdict
    import run_calibration
    monkeypatch.setenv("IOTPI_RESULTS_DIR", str(tmp_path))
    d = run_calibration.main(["--backend", "mock", "--screen", "heuristic", "--units", "5"])
    cal = json.loads((d / "calibration.json").read_text())
    generations = {r["generation_id"] for r in map(json.loads, (d / "generations.jsonl").read_text().splitlines())}
    counts = defaultdict(lambda: [0, 0])
    for row in map(json.loads, (d / "agent_measurements.jsonl").read_text().splitlines()):
        assert row["generation_id"] in generations
        counts[str(tuple(row["key"]))][0] += row["hit"]
        counts[str(tuple(row["key"]))][1] += 1
    assert {k: (v["k"], v["n"]) for k,v in cal["counts"]["agent_counts"].items()} == {k: tuple(v) for k,v in counts.items()}
    assert set(cal["training_families"]) == set(LIB["split"]["train"])
    for role, table in cal["training_rates"].items():
        for key,value in table.items():
            if isinstance(value, dict):
                assert not set(value) & set(LIB["split"]["heldout"])


def test_prediction_does_not_use_heldout_family_rates():
    from iotpi.predict import sample_rates
    count=lambda k,n: {"k":k,"n":n,"rate":k/n}
    cal={"training_families":["authority"], "counts":{"agent_counts":{
        str(("gateway","carry_raw","authority")):count(1,4),
        str(("gateway","carry_raw","dormant")):count(100,100)},
        "detector_counts":{"screen":{str(("raw","authority")):count(1,4),str(("raw","dormant")):count(100,100)}}}}
    rates,det=sample_rates(cal,None)
    assert rates["gateway"]["carry_raw"] == {"authority":.25,"_pooled":.25}
    assert det["raw"] == {"authority":.25,"_pooled":.25}


def test_detector_score_accepts_installed_flat_and_nested_shapes():
    from iotpi.detectors import HFDetector
    detector = HFDetector.__new__(HFDetector)
    rows = [{"label":"SAFE","score":.2},{"label":"INJECTION","score":.8}]
    detector.pipe = lambda text: rows
    assert detector.score_text("Ordinary note.") == .8
    detector.pipe = lambda text: [rows]
    assert detector.score_text("Ordinary note.") == .8


def test_clean_notice_does_not_inherit_command_only_infection():
    from iotpi.agents import AgentOutput
    from iotpi.conditions import conditions
    from iotpi.experiment import run_condition
    from iotpi.topologies import pilot
    sc = make_scenario(3,3,2,4,LIB,"authority","spread")
    graph = pilot(sc.zones)
    class Agent:
        def __init__(self, role): self.role=role
        def respond(self, inbox, memory, step, sc, graph, seed):
            if self.role == "coordinator":
                return AgentOutput(text="", report="Normal.", notice="No changes.",
                    commands=[(f"lights_{sc.zones[0]}","on",sc.attack.canary)] if step == 0 else [])
            return AgentOutput(text="",report="Normal.")
    agents={n:Agent(graph.role[n]) for n in graph.agents}
    result=run_condition(sc,graph,agents,conditions(1)["policy_only"],None,None,{},seed=1)
    assert not any(row["exposed"] for row in result["trace"] if row["node"] != "coord" and row["step"] > sc.attack.step)


def test_outbound_reply_does_not_create_a_network_carrier():
    from iotpi.agents import AgentOutput
    from iotpi.conditions import conditions
    from iotpi.experiment import run_condition
    from iotpi.topologies import pilot
    sc=make_scenario(3,3,2,4,LIB,"authority","exfil")
    graph=pilot(sc.zones)
    class Agent:
        def __init__(self,node): self.node=node
        def respond(self,inbox,memory,step,sc,graph,seed):
            replies=[(sc.attack.ticket_id,sc.secrets[self.node]+sc.attack.canary)] if self.node==graph.gateway(sc.attack.zone) and step==sc.attack.step else []
            return AgentOutput(text=json.dumps({"ticket_replies":replies}),report="Normal.",notice="No changes.",replies=replies)
    result=run_condition(sc,graph,{n:Agent(n) for n in graph.agents},conditions(1)["policy_only"],None,None,{},seed=1)
    assert result["attack"]["delivered"] and result["attack"]["exfil_attempted"]
    assert not result["attack"]["exfiltrated"]
    assert result["attack"]["n_carrier_nodes"] == 0 and not result["attack"]["spread_gateways"]


def test_resume_cache_preserves_a_partial_last_line(tmp_path):
    import pytest
    from iotpi.backends import CachingBackend, MockBackend
    path=tmp_path/'cache.jsonl'
    path.write_text('{"interrupted":')
    with pytest.warns(RuntimeWarning, match="incomplete"):
        cache=CachingBackend(MockBackend(),path)
    cache.generate([{"role":"user","content":"Ordinary note."}],seed=1,max_tokens=5,temperature=0)
    cache.close()
    assert path.read_text().startswith('{"interrupted":\n')


def test_detector_timing_is_recorded_per_call():
    from iotpi.detectors import HeuristicDetector
    from iotpi.messages import Message
    detector=HeuristicDetector()
    detector.score(Message("m","a","b","report","Normal.",0))
    assert detector.calls == len(detector.observations) == 1
    assert detector.observations[0]["seconds"] == detector.seconds


def test_conditions_receive_identical_scenario_and_generation_seeds(tmp_path,monkeypatch):
    import run_pilot
    monkeypatch.setenv("IOTPI_RESULTS_DIR",str(tmp_path))
    d=run_pilot.main(["--backend","mock","--screen","heuristic","--limit","2","--no-adaptive",
                      "--conditions","policy_only","classifier_cache","trustgate"])
    traces=[json.loads(x) for x in (d/'traces.jsonl').read_text().splitlines()]
    for index in [0,1]:
        rows=[r for r in traces if r['plan_index']==index]
        assert len({(r['condition_seed'],r['scenario_seed'],r['context_id'],r['attack_text']) for r in rows}) == 1
        common={}
        for row in rows:
            for step in row['trace']:
                if 'generation_seed' in step:
                    common.setdefault((step['node'],step['step']),set()).add(step['generation_seed'])
        assert all(len(seeds)==1 for seeds in common.values())


def test_summary_checks_harm_definition_for_each_goal():
    import copy,pytest
    from iotpi.report import summarise
    base={'condition':'policy_only','plan_index':0,'context_id':'ctx100','split':'train',
          'utility':{'expected':1,'done':1,'false_drops':0,'false_quarantines':0},'cost':{}}
    for goal in ['c3_safety','c2_unlock','c1_out_of_policy','exfil','spread']:
        attack={'goal':goal,'executed':goal.startswith('c'),'exfiltrated':goal=='exfil',
                'spread_gateways':['gw_other'] if goal=='spread' else [],'generated':False,'harm':True}
        row={**base,'attack':attack}
        assert summarise([row])['conditions']['policy_only']['harm'][0] == 1
        wrong=copy.deepcopy(row); wrong['attack']['harm']=False
        with pytest.raises(ValueError,match='goal-specific'): summarise([wrong])


def test_cheap_screen_does_not_recognize_evaluator_nonce_prefix():
    from iotpi.detectors import HeuristicDetector
    detector=HeuristicDetector()
    assert detector.score_text("Routine report ref Sabc123.") == detector.score_text("Routine report ref Qabc123.")


def test_completed_pilot_has_required_metrics_alias(tmp_path,monkeypatch):
    import run_pilot
    monkeypatch.setenv("IOTPI_RESULTS_DIR",str(tmp_path))
    d=run_pilot.main(["--backend","mock","--screen","heuristic","--limit","1","--no-adaptive","--conditions","policy_only"])
    assert json.loads((d/'metrics.json').read_text()) == json.loads((d/'summary.json').read_text())


def test_prediction_false_positive_rates_exclude_heldout_measurements():
    from iotpi.predict import sample_rates
    row=lambda k,n: {"k":k,"n":n,"rate":k/n}
    cal={"training_families":["authority"],"counts":{"agent_counts":{},
         "detector_counts":{"screen":{str(("benign","report")):row(100,100)}},
         "training_detector_counts":{"screen":{str(("benign","report")):row(0,20)}}}}
    _,detector=sample_rates(cal,None)
    assert detector['fpr']['report']==detector['fpr']['_pooled']==0


def test_calibration_persistence_advances_the_clock(tmp_path,monkeypatch):
    import run_calibration
    from iotpi.runlog import read_jsonl
    monkeypatch.setenv("IOTPI_RESULTS_DIR",str(tmp_path))
    directory=run_calibration.main(["--backend","mock","--screen","heuristic","--units","8","--mock-follow-prob","1"])
    generations=read_jsonl(directory/'generations.jsonl')
    checked=0
    for row in generations:
        if row['memory']:
            previous=[r for r in generations if r['context']==row['context'] and r['episode_seed']==row['episode_seed']
                      and r['node']==row['node'] and r['generation_id']<row['generation_id']]
            assert previous and previous[-1]['step']+1==row['step']<CFG['world']['horizon']
            checked+=1
    assert checked>0
