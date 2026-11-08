"""Fuzzy payload signatures for TrustGate's immunity gossip (SimHash over word 3-shingles)."""
from __future__ import annotations

import hashlib
import re

BITS = 64


def _shingles(text: str, k: int = 3):
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < k:
        return [" ".join(words)] if words else []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def simhash(text: str) -> int:
    v = [0] * BITS
    for sh in _shingles(text):
        h = int(hashlib.md5(sh.encode()).hexdigest()[:16], 16)
        for b in range(BITS):
            v[b] += 1 if (h >> b) & 1 else -1
    return sum(1 << b for b in range(BITS) if v[b] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def matches(sig: int, text: str, max_distance: int = 12) -> bool:
    return hamming(sig, simhash(text)) <= max_distance


def payload_part(text: str) -> str:
    """The bracketed notice block if present (what a defender would fingerprint), else the text."""
    m = re.search(r"\[(\w[\w-]*)\].*\[/\1\]", text, re.S)
    return m.group(0) if m else text


def choose_cache_distance(distances: list[int], max_fpr: float = 0.01) -> int:
    """Largest observed-safe Hamming threshold; calibration data only."""
    if not distances:
        raise ValueError("no benign report/notice distances to calibrate")
    accepted = [d for d in range(BITS + 1) if sum(x <= d for x in distances)/len(distances) <= max_fpr]
    if not accepted:
        raise ValueError("even exact matching exceeds the permitted benign false-match rate")
    return max(accepted)
