"""Multi-hop decomposition.

A sub-question only needs a grounded start entity (or the previous
sub-question's result) and a target class/property/literal shape; it does
NOT need to name every intermediate predicate itself -- that's
`SchemaPathFinder`'s job (`kgqa.construction.path_finder`) when the
question doesn't spell out the chain. Not implemented yet: real
decomposition needs the intent classifier and the grounding step
(retrieval strategy) wired up first so it has something to decompose
against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SubQuestion:
    text: str
    start_entity_mention: str | None
    target_shape: str


class MultiHopDecomposer:
    def decompose(self, question: str) -> list[SubQuestion]:
        raise NotImplementedError(
            "Multi-hop decomposition is not implemented yet."
        )
