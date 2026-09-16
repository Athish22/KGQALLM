"""SchemaPathFinder: bounded-depth path search over
the schema-as-graph representation (classes as nodes, properties as edges
via domain/range), for questions that name a start entity and a target
without naming the predicates connecting them -- e.g. "pressure readings
for Machine 6" implies `Machine -hasTool-> Tool -sosa:madeObservation->
Observation -sosa:hasSimpleResult-> result`, but only "Machine 6" and
"pressure readings" are grounded from the text. Relation-phrase grounding
only matches phrases that appear in the question, so it can't discover an
unstated intermediate hop -- that's what this component is for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from kgqa.adapters.base import KGAdapter


@dataclass
class PathHop:
    property_uri: str
    forward: bool  # True: traversed subject->object; False: object->subject


@dataclass
class PropertyPath:
    hops: list[PathHop]
    end_class: str

    @property
    def properties(self) -> list[str]:
        return [hop.property_uri for hop in self.hops]


class SchemaPathFinder:
    def __init__(self) -> None:
        self._adjacency: dict[str, list[tuple[str, bool, str]]] = {}
        self._property_uris: set[str] = set()
        self._class_uris: set[str] = set()

    def build_graph(self, adapter: KGAdapter) -> None:
        self._adjacency = {}
        self._property_uris = set()
        self._class_uris = {c.uri for c in adapter.get_classes()}

        for prop in adapter.get_properties():
            self._property_uris.add(prop.uri)
            domains = prop.domain or [None]
            ranges = prop.range or [None]
            for d in domains:
                for r in ranges:
                    if d is None or r is None:
                        continue
                    self._adjacency.setdefault(d, []).append((prop.uri, True, r))
                    self._adjacency.setdefault(r, []).append((prop.uri, False, d))

    def find_paths(self, source_class: str, target: str, max_hops: int = 3) -> list[PropertyPath]:
        """BFS from `source_class` to `target`, where `target` may be a
        class URI (path must *end* at that class) or a property URI (path
        must end at a node with an outgoing/incoming edge of that
        property -- the hop onto that property is included as the last
        hop). Returns shortest paths first; stops at the first hop-count
        that yields any match (doesn't keep searching deeper once found).
        """
        target_is_property = target in self._property_uris

        found: list[PropertyPath] = []
        visited: set[str] = {source_class}
        current_layer: list[tuple[str, list[PathHop]]] = [(source_class, [])]

        for depth in range(max_hops):
            next_layer: list[tuple[str, list[PathHop]]] = []
            for node, hops in current_layer:
                for prop_uri, forward, neighbor in self._adjacency.get(node, []):
                    new_hops = hops + [PathHop(property_uri=prop_uri, forward=forward)]
                    if target_is_property and prop_uri == target:
                        found.append(PropertyPath(hops=new_hops, end_class=neighbor))
                        continue
                    if not target_is_property and neighbor == target:
                        found.append(PropertyPath(hops=new_hops, end_class=neighbor))
                        continue
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_layer.append((neighbor, new_hops))

            if found:
                # Finish this full BFS layer (shortest paths may arrive
                # from multiple sibling nodes at the same depth) then
                # stop -- don't search deeper once the shallowest matches
                # are in hand.
                break
            current_layer = next_layer
            if not current_layer:
                break

        return found
