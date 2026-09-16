"""Tier B retrieval strategy: lightweight local embedding index, ANN
search at query time.

Stubbed until a KG with nameable-entity count in the thousands is plugged
in -- the reference KG is Tier A. Intended shape: embed all labels once
in `build_index` (e.g. sentence-transformers),
store in a local vector index (FAISS/hnswlib), retrieve top-k in
`ground_entity`/`ground_property`.
"""

from __future__ import annotations

from kgqa.adapters.base import KGAdapter
from kgqa.retrieval.base import EntityCandidate, PropertyCandidate, RetrievalStrategy


class LightEmbeddingStrategy(RetrievalStrategy):
    def build_index(self, adapter: KGAdapter) -> None:
        raise NotImplementedError(
            "Tier B (LightEmbeddingStrategy) is not implemented yet -- "
            "no KG plugged in so far needs it."
        )

    def ground_entity(self, mention: str, context: str = "") -> list[EntityCandidate]:
        raise NotImplementedError

    def ground_property(self, phrase: str, context: str = "") -> list[PropertyCandidate]:
        raise NotImplementedError
