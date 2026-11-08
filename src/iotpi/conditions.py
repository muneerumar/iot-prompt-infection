"""Experimental conditions. All screened conditions share one per-node screening budget (budget-matched),
except `classifier_all`, the unbudgeted reference. The actuator policy (C1 block, C2 evidence) is on
everywhere except `no_guard`, which only measures raw attack strength."""
from __future__ import annotations

from .guard import GuardConfig

TRUSTGATE = dict(screen="risk", cache=True, alerts=True, quarantine=True, hold_alerted_c3=True,
                 degrade_to_digest=True)


def conditions(budget: float, cache_distance: int | None = None) -> dict:
    b = budget
    c = {
        # references
        "no_guard": GuardConfig("no_guard", policy=False, egress_filter=False),
        "policy_only": GuardConfig("policy_only"),
        "spotlight": GuardConfig("spotlight", prompt_defence="spotlight"),
        "tagging": GuardConfig("tagging", prompt_defence="tagging"),
        "provenance_block": GuardConfig("provenance_block", taint_block_c3=True),
        "classifier_all": GuardConfig("classifier_all", screen="all"),
        # budget-matched baselines (audit item 7)
        "classifier_random": GuardConfig("classifier_random", screen="random", budget_per_node=b),
        "classifier_taint": GuardConfig("classifier_taint", screen="taint", budget_per_node=b),
        "classifier_cache": GuardConfig("classifier_cache", screen="taint", cache=True, budget_per_node=b),
        "classifier_trust": GuardConfig("classifier_trust", screen="trust_ewma", budget_per_node=b),
        # TrustGate and ablations
        "trustgate": GuardConfig("trustgate", budget_per_node=b, **TRUSTGATE),
        "tg_no_risk_split": GuardConfig("tg_no_risk_split", budget_per_node=b, **TRUSTGATE, risk_split=False),
        "tg_no_alerts": GuardConfig("tg_no_alerts", budget_per_node=b, **{**TRUSTGATE, "alerts": False,
                                                                         "hold_alerted_c3": False}),
        "tg_no_quarantine": GuardConfig("tg_no_quarantine", budget_per_node=b, **{**TRUSTGATE, "quarantine": False}),
        "tg_no_cache": GuardConfig("tg_no_cache", budget_per_node=b, **{**TRUSTGATE, "cache": False}),
        "tg_no_hold": GuardConfig("tg_no_hold", budget_per_node=b, **{**TRUSTGATE, "hold_alerted_c3": False}),
        "tg_no_digest": GuardConfig("tg_no_digest", budget_per_node=b, **{**TRUSTGATE, "degrade_to_digest": False}),
    }
    if cache_distance is not None:
        for config in c.values():
            config.cache_distance = int(cache_distance)
    return c


PILOT_CORE = ["no_guard", "policy_only", "classifier_all", "classifier_taint", "classifier_cache",
              "classifier_trust", "trustgate", "provenance_block"]
