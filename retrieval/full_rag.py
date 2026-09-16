"""Tier C retrieval strategy: embedding retrieval + a reranking pass
(cross-encoder or LLM-based) over the top candidates, since precision at
large scale matters more than recall alone.

Stubbed until a KG with nameable-entity count in the 10^5+ range (or high
polysemy) is plugged in.
"""

from __future__ import annotations

from kgqa.adapters.base import KGAdapter
from kgqa.retrieval.base import EntityCandidate, PropertyCandidate, RetrievalStrategy


class FullRAGStrategy(RetrievalStrategy):
    def build_index(self, adapter: KGAdapter) -> None:
        raise NotImplementedError(
            "Tier C (FullRAGStrategy) is not implemented yet -- "
            "no KG plugged in so far needs it."
        )

    def ground_entity(self, mention: str, context: str = "") -> list[EntityCandidate]:
        raise NotImplementedError

    def ground_property(self, phrase: str, context: str = "") -> list[PropertyCandidate]:
        raise NotImplementedError
