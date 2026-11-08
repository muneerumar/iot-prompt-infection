"""The trusted guard runtime: one instance per agent node plus the actuator interface.

It runs OUTSIDE the LLM, so an infected agent cannot switch it off. It sees message text,
sender, kind, lineage and taint (computed by the runtime from what each agent actually read),
never ground truth.

Screening strategies, all held to the same per-step screening budget unless `screen="all"`:
  none        no classifier
  all         classify every message (unbudgeted reference)
  random      classify a random subset
  taint       classify messages derived from untrusted input first
  cheap       order by a near-free heuristic score
  trust_ewma  order by sender distrust, where trust is an EWMA of the classifier's own verdicts
              (the original TrustGate design, kept as an ablation of its circularity)
  risk        TrustGate: the site budget is split across nodes by downstream actuator risk
              (distance-vector risk weights x local suspicious load); within a node, messages are
              ordered by suspicion = taint + cheap score + alert match + fingerprint near-match
Containment add-ons: fingerprint cache, authenticated bounded alerts, sender quarantine with
expiry and a rule-based fallback, and an actuator hold on C3 commands whose lineage was alerted.
"""
from __future__ import annotations

import hashlib
import hmac
import random
import re
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field

from .messages import Command, Message
from .policy import classify, evidence_ok
from .signatures import hamming, simhash

STRATEGIES = ("none", "all", "random", "taint", "cheap", "trust_ewma", "risk")


@dataclass
class GuardConfig:
    name: str
    policy: bool = True
    egress_filter: bool = True
    screen: str = "none"
    budget_per_node: float = 1.0
    cache: bool = False
    alerts: bool = False
    quarantine: bool = False
    hold_alerted_c3: bool = False
    taint_block_c3: bool = False
    prompt_defence: str = "none"
    quarantine_k: int = 2
    quarantine_window: int = 3
    quarantine_steps: int = 2
    quarantine_roles: tuple = ("gateway",)   # coordinators are never isolated automatically
    alert_ttl: int = 2
    alert_expiry: int = 3
    alert_rate_limit: int = 4
    alert_store: int = 64
    cache_distance: int = 10
    cache_store: int = 256
    trust_alpha: float = 0.7
    min_share: float = 0.25
    risk_split: bool = True        # split the site budget across nodes by risk weight x load
    degrade_to_digest: bool = False  # a caught report is replaced by a trusted digest of its checkable requests
    weights: dict = field(default_factory=lambda: {"taint": 1.0, "cheap": 1.0, "alert": 2.0, "near": 1.0})

    def __post_init__(self):
        if self.screen not in STRATEGIES:
            raise ValueError(f"unknown screen strategy {self.screen}")


@dataclass
class Alert:
    id: str
    reporter: str
    roots: frozenset
    fp: int
    step: int
    ttl: int
    mac: str

    def body(self) -> bytes:
        return f"{self.id}|{self.reporter}|{sorted(self.roots)}|{self.fp}|{self.step}|{self.ttl}".encode()

    @property
    def nbytes(self) -> int:
        return len(self.body()) + len(self.mac)


class NodeState:
    def __init__(self, cfg: GuardConfig):
        self.cfg = cfg
        self.credit = 0.0
        self.budget = cfg.budget_per_node
        self.alerts: OrderedDict = OrderedDict()   # id -> (Alert, expiry)
        self.cache: OrderedDict = OrderedDict()    # fp -> step
        self.trust = defaultdict(lambda: 1.0)
        self.detections = defaultdict(deque)       # sender -> steps of detections
        self.load = 1.0                            # EWMA of tainted arrivals per step
        self.cost = defaultdict(float)

    def flagged_roots(self, step: int) -> set:
        return {r for a, exp in self.alerts.values() if exp >= step for r in a.roots}

    def flagged_fps(self, step: int) -> list[int]:
        return [a.fp for a, exp in self.alerts.values() if exp >= step]


class GuardNetwork:
    def __init__(self, cfg: GuardConfig, graph, screen_detector=None, cheap_detector=None,
                 risk_weights: dict | None = None, seed: int = 0, key: bytes = b"site-key"):
        self.cfg = cfg
        self.graph = graph
        self.screen = screen_detector
        self.cheap = cheap_detector
        self.risk = risk_weights or {n: 1.0 for n in graph.agents}
        self.rng = random.Random(seed)
        self.key = key
        self.nodes = {n: NodeState(cfg) for n in list(graph.agents) + ["act"]}
        self.quarantined: dict = {}          # sender -> release step
        self.events: list = []
        self.screen_audit = []
        self._alert_seq = 0
        self._sent_this_step = defaultdict(int)
        if cfg.screen != "none" and screen_detector is None:
            raise ValueError("screening strategy needs a screen detector")

    # ---------------------------------------------------------------- budget
    def begin_step(self, step: int) -> list[str]:
        """Refill budgets; release expired quarantines. Returns nodes released this step."""
        self._sent_this_step.clear()
        released = []
        agents = list(self.graph.agents)
        b = self.cfg.budget_per_node
        if self.cfg.screen == "risk" and self.cfg.risk_split:
            total = b * len(agents)
            w = {n: self.risk.get(n, 0.0) * self.nodes[n].load for n in agents}
            s = sum(w.values())
            floor = self.cfg.min_share * b
            spare = total - floor * len(agents)
            for n in agents:
                self.nodes[n].budget = floor + (spare * w[n] / s if s > 0 else spare / len(agents))
        else:
            for n in agents:
                self.nodes[n].budget = b
        for n in agents:
            st = self.nodes[n]
            st.credit = min(st.credit + st.budget, max(1.0, 2 * st.budget))
        for node, release in list(self.quarantined.items()):
            if release <= step:
                del self.quarantined[node]
                released.append(node)
                self.events.append(("release", node, step))
        return released

    # -------------------------------------------------------------- inbound
    def is_quarantined(self, node: str) -> bool:
        return node in self.quarantined

    def filter_inbox(self, node: str, msgs: list[Message], step: int) -> list[Message]:
        cfg, st = self.cfg, self.nodes[node]
        delivered, detected = [], []
        candidates = []
        tainted_n = 0
        for m in msgs:
            if m.sender in self.quarantined:
                self.events.append(("drop_quarantined", node, step, m.id))
                continue
            tainted_n += m.tainted
            candidates.append(m)
        st.load = 0.7 * st.load + 0.3 * tainted_n

        near = {}
        if cfg.cache or cfg.alerts:
            fps = list(st.cache) + st.flagged_fps(step)
            for m in candidates:
                if not m.tainted or not fps:
                    continue
                st.cost["hash_ops"] += 1
                d = min(hamming(simhash(m.text), f) for f in fps)
                if cfg.cache and d <= cfg.cache_distance and any(
                        hamming(simhash(m.text), f) <= cfg.cache_distance for f in st.cache):
                    detected.append((m, "cache"))
                elif d <= 2 * cfg.cache_distance:
                    near[m.id] = 1.0
        hit_ids = {m.id for m, _ in detected}
        candidates = [m for m in candidates if m.id not in hit_ids]

        credit_before = st.credit
        selected = self._select(node, candidates, step, near)
        for m in selected:
            st.credit -= 1
            flagged, score = self.screen.flags(m)
            st.cost["screens"] += 1
            if cfg.screen == "trust_ewma":
                st.trust[m.sender] = cfg.trust_alpha * st.trust[m.sender] + (1 - cfg.trust_alpha) * (0.0 if flagged else 1.0)
            if flagged:
                detected.append((m, "screen"))
        hit_ids = {m.id for m, _ in detected}
        delivered = [m for m in candidates if m.id not in hit_ids]
        for m, how in detected:
            self._on_detect(node, m, step, how)
            if cfg.degrade_to_digest and m.kind == "report":
                delivered.append(self._digest(node, m, step))
        self.screen_audit.append({"node": node, "step": step, "budget": st.budget,
                                  "credit_before": credit_before, "screens": len(selected), "credit_after": st.credit})
        return delivered

    @staticmethod
    def _digest(node, m: Message, step: int) -> Message:
        """Keep only requests the actuator can check against trusted state (C2) or that are safe.
        No free text from the caught message survives, so the digest carries no taint."""
        keep = [(r[0], r[1], "guard-validated request") for r in m.payload.get("requests", [])
                if re.fullmatch(r"[a-z]+_[a-z0-9]+", r[0]) and classify(r[0], r[1]) in ("C2", "SAFE")]
        listed = "; ".join(f"{d} {c} ({w})" for d, c, w in keep) if keep else "none"
        return Message(f"{m.id}d", f"guard_{node}", node, "digest",
                       f"Report from {m.sender} was withheld by the guard; only checkable requests are kept. Requests: {listed}", step,
                       payload={"requests": keep, "withheld_from": m.sender})

    def _select(self, node, cands, step, near) -> list[Message]:
        cfg, st = self.cfg, self.nodes[node]
        if cfg.screen == "none" or not cands:
            return []
        if cfg.screen == "all":
            st.credit += len(cands)
            return list(cands)
        n = int(st.credit)
        if n <= 0:
            return []
        order = list(cands)
        self.rng.shuffle(order)
        if cfg.screen == "taint":
            order.sort(key=lambda m: not m.tainted)
        elif cfg.screen == "cheap":
            order.sort(key=lambda m: -self._cheap(node, m))
        elif cfg.screen == "trust_ewma":
            order.sort(key=lambda m: st.trust[m.sender])
        elif cfg.screen == "risk":
            roots = st.flagged_roots(step)
            w = cfg.weights

            def suspicion(m):
                if not m.tainted:
                    return 0.0
                return (w["taint"] + w["cheap"] * self._cheap(node, m) + w["alert"] * bool(m.lineage & roots)
                        + w["near"] * near.get(m.id, 0.0))
            order = [m for m in order if m.tainted] + [m for m in order if not m.tainted]
            order.sort(key=lambda m: -suspicion(m))
        return order[:n]

    def _cheap(self, node, m) -> float:
        if self.cheap is None:
            return 0.0
        if "_cheap" not in m.payload:
            m.payload["_cheap"] = self.cheap.score(m)
            self.nodes[node].cost["cheap_scores"] += 1
        return m.payload["_cheap"]

    def _on_detect(self, node, m: Message, step: int, how: str) -> None:
        cfg, st = self.cfg, self.nodes[node]
        self.events.append(("detect", node, step, m.id, m.sender, how))
        fp = simhash(m.text)
        if cfg.cache:
            st.cache[fp] = step
            while len(st.cache) > cfg.cache_store:
                st.cache.popitem(last=False)
        if cfg.alerts:
            self._emit_alert(node, m.lineage, fp, step)
        if cfg.quarantine and self.graph.role.get(m.sender) in cfg.quarantine_roles:
            q = st.detections[m.sender]
            q.append(step)
            while q and q[0] <= step - cfg.quarantine_window:
                q.popleft()
            if len(q) >= cfg.quarantine_k and m.sender not in self.quarantined:
                self.quarantined[m.sender] = step + cfg.quarantine_steps
                self.events.append(("quarantine", m.sender, step, node))

    # --------------------------------------------------------------- alerts
    def _mac(self, a: Alert) -> str:
        return hmac.new(self.key, a.body(), hashlib.sha256).hexdigest()[:32]

    def _emit_alert(self, reporter: str, roots: frozenset, fp: int, step: int) -> None:
        if self._sent_this_step[reporter] >= self.cfg.alert_rate_limit:
            self.events.append(("alert_rate_limited", reporter, step))
            return
        self._sent_this_step[reporter] += 1
        self._alert_seq += 1
        a = Alert(f"{reporter}#{self._alert_seq}", reporter, frozenset(roots), fp, step, self.cfg.alert_ttl, "")
        a.mac = self._mac(a)
        self.receive_alert(reporter, a, step)
        frontier, seen = deque([(reporter, 0)]), {reporter}
        while frontier:
            u, d = frontier.popleft()
            if d == a.ttl:
                continue
            for v in self.graph.undirected_neighbours(u):
                if v not in seen:
                    seen.add(v)
                    self.nodes[reporter].cost["alert_msgs"] += 1
                    self.nodes[reporter].cost["alert_bytes"] += a.nbytes
                    self.receive_alert(v, a, step)
                    frontier.append((v, d + 1))
        self.nodes[reporter].cost["alert_msgs"] += 1
        self.nodes[reporter].cost["alert_bytes"] += a.nbytes
        self.receive_alert("act", a, step)   # count the actuator subscription delivery

    def receive_alert(self, node: str, a: Alert, step: int) -> bool:
        if not hmac.compare_digest(a.mac, self._mac(a)):
            self.events.append(("alert_rejected", node, step, a.id))
            return False
        st = self.nodes[node]
        if a.id in st.alerts:
            return False
        st.alerts[a.id] = (a, step + self.cfg.alert_expiry)
        while len(st.alerts) > self.cfg.alert_store:
            st.alerts.popitem(last=False)
        return True

    # -------------------------------------------------------------- actuator
    def authorize(self, cmd: Command, step: int, state) -> tuple[bool, str]:
        if not self.cfg.policy:
            return True, "no_policy"
        if cmd.device not in {d for devices in state.devices.values() for d in devices}:
            return False, "policy"
        cls = classify(cmd.device, cmd.command)
        if cls == "C1":
            return False, "policy"
        if cls == "SAFE":
            return True, "safe"
        if cls == "C2":
            return (True, "evidence_ok") if evidence_ok(cmd.device, cmd.command, step, state) else (False, "evidence")
        if self.cfg.taint_block_c3 and cmd.tainted:
            return False, "taint"
        if self.cfg.hold_alerted_c3 and cmd.lineage & self.nodes["act"].flagged_roots(step):
            return False, "alert_hold"
        return True, "c3_pass"

    def cost_summary(self) -> dict:
        tot = defaultdict(float)
        for st in self.nodes.values():
            for k, v in st.cost.items():
                tot[k] += v
        if self.screen is not None:
            tot["screen_seconds"] = self.screen.seconds
            tot["screen_call_measurements"] = self.screen.observations
            obs = self.screen.observations
            tot["screen_llm_calls"] = sum(r.get("llm_calls",0) for r in obs)
            tot["screen_llm_prompt_tokens"] = sum(r.get("prompt_tokens",0) for r in obs)
            tot["screen_llm_generation_tokens"] = sum(r.get("generation_tokens",0) for r in obs)
            tot["screen_physical_seconds"] = sum(r.get("physical_seconds",r["seconds"]) for r in obs)
            tier_calls, tier_seconds, tier_physical = defaultdict(int), defaultdict(float), defaultdict(float)
            for row in obs:
                for name, tier in row.get("tiers",{}).items():
                    tier_calls[name] += 1
                    tier_seconds[name] += tier["seconds"]
                    tier_physical[name] += tier.get("physical_seconds",tier["seconds"])
            if tier_calls:
                tot["screen_tier_calls"] = dict(tier_calls)
                tot["screen_tier_seconds"] = dict(tier_seconds)
                tot["screen_tier_physical_seconds"] = dict(tier_physical)
        if self.cheap is not None:
            tot["cheap_seconds"] = self.cheap.seconds
        return {**dict(tot), "screen_audit": self.screen_audit}
