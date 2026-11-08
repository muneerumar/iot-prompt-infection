"""Application message-flow graphs (who actually passes text to whom), not radio connectivity.

Each edge is a real opportunity for prompt transfer:
  gateway -> parent coordinator   zone report (same step)
  coordinator -> parent           area report (same step)
  coordinator -> children         building notice (next step)
Area coordinators command the devices of their own zones. The pilot is one coordinator over
`n_zones` gateways; the campus graph adds area coordinators under a campus coordinator, which
creates a lateral path between areas (gateway -> area -> campus -> other area -> gateway).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlowGraph:
    zones: list[str]
    parent: dict            # node -> parent node (or None)
    children: dict          # node -> list of child nodes
    role: dict              # node -> "gateway" | "coordinator"
    zone_of: dict           # gateway node -> zone
    commands_for: dict      # coordinator -> list of zones whose devices it commands
    levels: list = field(default_factory=list)   # processing order, leaves first

    @property
    def agents(self) -> list[str]:
        return [n for lvl in self.levels for n in lvl]

    def gateway(self, zone: str) -> str:
        return f"gw_{zone}"

    def commander_of(self, zone: str) -> str:
        return next(c for c, zs in self.commands_for.items() if zone in zs)

    def out_edges(self, node: str) -> list[tuple[str, str]]:
        """(target, kind) for every message this node sends."""
        out = []
        if self.parent.get(node):
            out.append((self.parent[node], "report"))
        for c in self.children.get(node, []):
            out.append((c, "notice"))
        return out

    def undirected_neighbours(self, node: str) -> list[str]:
        nb = list(self.children.get(node, []))
        if self.parent.get(node):
            nb.append(self.parent[node])
        return nb


def pilot(zones: list[str]) -> FlowGraph:
    gws = [f"gw_{z}" for z in zones]
    parent = {g: "coord" for g in gws} | {"coord": None}
    return FlowGraph(zones, parent, {"coord": gws}, {**{g: "gateway" for g in gws}, "coord": "coordinator"},
                     {f"gw_{z}": z for z in zones}, {"coord": list(zones)}, [gws, ["coord"]])


def campus(zones: list[str], zones_per_area: int) -> FlowGraph:
    areas = [zones[i:i + zones_per_area] for i in range(0, len(zones), zones_per_area)]
    parent, children, role, zone_of, cmd = {"campus": None}, {"campus": []}, {"campus": "coordinator"}, {}, {}
    gw_level, area_level = [], []
    for k, zs in enumerate(areas):
        a = f"area{k}"
        parent[a] = "campus"
        children["campus"].append(a)
        children[a] = []
        role[a] = "coordinator"
        cmd[a] = list(zs)
        area_level.append(a)
        for z in zs:
            g = f"gw_{z}"
            parent[g], role[g], zone_of[g] = a, "gateway", z
            children[a].append(g)
            gw_level.append(g)
    cmd["campus"] = []
    return FlowGraph(zones, parent, children, role, zone_of, cmd, [gw_level, area_level, ["campus"]])
