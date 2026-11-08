import copy
import ast
from pathlib import Path
import yaml
from iotpi.attacks import load_library
import random

from iotpi.detectors import HeuristicDetector
from iotpi.guard import Alert, GuardConfig, GuardNetwork
from iotpi.messages import Command, Message
from iotpi.policy import classify
from iotpi.scenario import make_scenario
from iotpi.topologies import pilot

G = pilot(["north", "south"])
INJ = make_scenario(0, 0, 2, 4, load_library(Path(__file__).resolve().parents[1], "tests/fixtures/attacks.synthetic.yaml"), "imperative", "spread").attack.text


def msgs(gt_flip=False):
    rng = random.Random(0)
    out = []
    for i in range(6):
        text = INJ if i == 2 else f"zone north: temperature {20 + i} C"
        m = Message(f"m{i}", f"ticket_T{i}" if i % 2 else "dev_north", "gw_north", "ticket" if i % 2 else "sensor",
                    text, 0, frozenset({f"T{i}"}) if i % 2 else frozenset(), bool(i % 2))
        m.gt_attack = (i == 2) ^ gt_flip ^ (rng.random() < 0.5 and gt_flip)
        out.append(m)
    return out


def test_guard_ignores_ground_truth():
    cfg = GuardConfig("t", screen="risk", budget_per_node=1, cache=True, alerts=True, quarantine=True,
                      hold_alerted_c3=True, degrade_to_digest=True)
    res = []
    for flip in (False, True):
        g = GuardNetwork(cfg, G, HeuristicDetector(0.5), HeuristicDetector(0.5), {"gw_north": 1, "gw_south": 1, "coord": 1}, seed=1)
        g.begin_step(0)
        res.append([m.id for m in g.filter_inbox("gw_north", msgs(flip), 0)])
    assert res[0] == res[1]


def test_budget_is_respected():
    for strategy in ("random", "taint", "cheap", "trust_ewma", "risk"):
        g = GuardNetwork(GuardConfig("t", screen=strategy, budget_per_node=1), G, HeuristicDetector(0.5),
                         HeuristicDetector(0.5), {"gw_north": 3, "gw_south": 1, "coord": 1}, seed=2)
        total = 0
        for step in range(5):
            g.begin_step(step)
            for node in G.agents:
                before = g.nodes[node].cost["screens"]
                g.filter_inbox(node, copy.deepcopy(msgs()), step)
                total += g.nodes[node].cost["screens"] - before
        assert total <= 1 * len(G.agents) * 5 + len(G.agents), strategy   # + one step of banked credit


def test_forged_alert_rejected_and_expiry():
    g = GuardNetwork(GuardConfig("t", alerts=True, alert_expiry=1), G, seed=0)
    a = Alert("x#1", "gw_north", frozenset({"T9"}), 0, 0, 2, "0" * 32)
    assert not g.receive_alert("coord", a, 0)
    g._emit_alert("gw_north", frozenset({"T9"}), 123, 0)
    assert "T9" in g.nodes["coord"].flagged_roots(0) and "T9" in g.nodes["act"].flagged_roots(1)
    assert "T9" not in g.nodes["act"].flagged_roots(2)


def test_alert_rate_limit():
    g = GuardNetwork(GuardConfig("t", alerts=True, alert_rate_limit=2), G, seed=0)
    g.begin_step(0)
    for k in range(5):
        g._emit_alert("gw_north", frozenset({f"T{k}"}), k, 0)
    assert len(g.nodes["coord"].alerts) == 2


def test_actuator_policy_classes():
    sc = make_scenario(1, 1, 2, 4)
    z = sc.zones[0]
    g = GuardNetwork(GuardConfig("t", hold_alerted_c3=True, taint_block_c3=False), pilot(sc.zones), seed=0)
    ok = lambda d, c, lin=frozenset(), t=False: g.authorize(Command(d, c, "coord", 0, lin, t), 0, sc)  # noqa: E731
    assert ok("alarm_system", "disable_all") == (False, "policy")
    assert ok(f"door_{z}", "lock") == (True, "safe")
    assert ok(f"smoke_{z}", "maintenance_mode")[0] is True
    booked = (f"door_{z}", 0) in sc.bookings
    assert ok(f"door_{z}", "unlock")[0] is booked
    g._emit_alert(f"gw_{z}", frozenset({"Tbad"}), 1, 0)
    assert ok(f"smoke_{z}", "maintenance_mode", frozenset({"Tbad"})) == (False, "alert_hold")
    tb = GuardNetwork(GuardConfig("t", taint_block_c3=True), pilot(sc.zones), seed=0)
    assert tb.authorize(Command(f"lights_{z}", "on", "coord", 0, frozenset(), True), 0, sc) == (False, "taint")


def test_digest_keeps_only_checkable_requests():
    m = Message("r1", "gw_north", "coord", "report", INJ, 0, frozenset({"T2"}), True,
                {"requests": [("door_north", "unlock", "x"), ("smoke_north", "maintenance_mode", "y")]}, True)
    d = GuardNetwork._digest("coord", m, 0)
    assert d.payload["requests"] == [("door_north", "unlock", "guard-validated request")]
    assert not d.tainted and not d.lineage and "ignore" not in d.text
    assert classify("door_north", "unlock") == "C2"


def test_digest_strips_untrusted_request_reason():
    m = Message("r2", "gw_north", "coord", "report", "Withheld.", 0,
                payload={"requests": [("door_north", "unlock", INJ)]})
    d = GuardNetwork._digest("coord", m, 0)
    assert INJ not in d.text
    assert all(INJ not in reason for _, _, reason in d.payload["requests"])


def test_unknown_devices_are_out_of_policy_even_for_safe_commands():
    sc = make_scenario(1, 1, 2, 4)
    cmd = Command("door_nonexistent", "lock", "coord", 0, frozenset(), False)
    guard = GuardNetwork(GuardConfig("policy"), pilot(sc.zones))
    assert guard.authorize(cmd, 0, sc) == (False, "policy")
    unguarded = GuardNetwork(GuardConfig("none", policy=False), pilot(sc.zones))
    assert unguarded.authorize(cmd, 0, sc) == (True, "no_policy")


def test_guard_and_real_detectors_have_no_ground_truth_reads():
    root = Path(__file__).resolve().parents[1] / "src/iotpi"
    violations = []
    class Inspect(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            if node.name != "CalibratedDetector":
                self.generic_visit(node)
        def visit_Attribute(self, node):
            if isinstance(node.ctx, ast.Load) and node.attr.startswith("gt_"):
                violations.append((filename, node.lineno))
            self.generic_visit(node)
    for filename in ("guard.py", "detectors.py"):
        Inspect().visit(ast.parse((root / filename).read_text()))
    assert not violations


def test_cache_distance_is_calibration_selected_and_shared_by_conditions():
    import pytest
    from iotpi.signatures import choose_cache_distance
    from iotpi.conditions import conditions
    assert choose_cache_distance([0] + [15]*99) == 14
    assert {cfg.cache_distance for cfg in conditions(1,14).values()} == {14}
    with pytest.raises(ValueError, match="exact matching"):
        choose_cache_distance([0]*2 + [64]*98)


def test_digest_rejects_non_identifier_device_fields():
    m=Message("r3","gw_north","coord","report","Withheld.",0,
              payload={"requests":[("door_north\n"+INJ,"unlock","normal")]})
    digest=GuardNetwork._digest("coord",m,0)
    assert digest.payload["requests"] == [] and INJ not in digest.text


def test_falsely_quarantined_gateway_has_a_checkable_fallback():
    from iotpi.backends import MockBackend
    from iotpi.experiment import llm_agents,load_prompts
    from iotpi.engine import run_episode
    from iotpi.scenario import Expected
    sc=make_scenario(3,3,2,4)
    graph=pilot(sc.zones)
    zone=sc.zones[0]
    sc.sensors[(zone,0)]["co2_ppm"]=1400
    sc.expected.append(Expected(f"ventilation_{zone}","boost",0,1,"sensor","C2"))
    guard=GuardNetwork(GuardConfig("forced-quarantine"),graph)
    guard.quarantined[graph.gateway(zone)]=2
    backend=MockBackend(0)
    agents=llm_agents(graph,backend,load_prompts(Path(__file__).resolve().parents[1]),{"max_tokens":300,"temperature":0})
    result=run_episode(sc,graph,agents,guard,seed=1)
    assert result["utility"]["fallback_digests"] > 0
    assert any(c["device"]==f"ventilation_{zone}" and c["command"]=="boost" and c["executed"] and c["step"]==0
               for c in result["commands"])


def test_alert_cost_counts_actuator_subscription_delivery():
    guard=GuardNetwork(GuardConfig("alerts",alerts=True),G)
    guard.begin_step(0)
    guard._emit_alert("gw_north",frozenset({"T1"}),123,0)
    assert guard.cost_summary()["alert_msgs"] == 3
