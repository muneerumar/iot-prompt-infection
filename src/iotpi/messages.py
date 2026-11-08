"""Message and command records exchanged in an episode.

Fields prefixed `gt_` are ground truth for the evaluator. Guards must never read them, except the
`CalibratedDetector`, which is a simulator stand-in for a measured detector (tests enforce this).
"""
from __future__ import annotations

from dataclasses import dataclass, field

UNTRUSTED_KINDS = frozenset({"ticket"})   # text written outside the building system


@dataclass
class Message:
    id: str
    sender: str
    receiver: str
    kind: str                   # sensor | ticket | report | notice | digest
    text: str
    step: int
    lineage: frozenset = frozenset()   # ids of untrusted root inputs this message derives from
    tainted: bool = False              # derived (conservatively) from any untrusted input
    payload: dict = field(default_factory=dict)   # structured content (requests, readings)
    # ground truth, evaluator only
    gt_attack: bool = False
    gt_family: str | None = None
    gt_hop: int = 0             # 0 = attacker's own text; k = rewritten by k agents

    @property
    def nbytes(self) -> int:
        return len(self.text.encode())


@dataclass
class Command:
    device: str
    command: str
    issuer: str
    step: int
    lineage: frozenset
    tainted: bool
    reason: str = ""


@dataclass
class Ticket:
    id: str
    zone: str
    step: int
    text: str
    legit_command: tuple[str, str] | None = None   # (device, command) a legitimate ticket asks for
    is_attack: bool = False
