"""RetrievalStrategy: swappable, one implementation per tier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from kgqa.adapters.base import KGAdapter


@dataclass
class EntityCandidate:
    uri: str
    label: str
    score: float  # 0..1, higher is better
    class_uri: str | None = None  # the entity's own rdf:type, when known


@dataclass
class PropertyCandidate:
    uri: str
    label: str
    score: float


class RetrievalStrategy(ABC):
    @abstractmethod
    def build_index(self, adapter: KGAdapter) -> None: ...

    @abstractmethod
    def ground_entity(self, mention: str, context: str = "") -> list[EntityCandidate]: ...

    @abstractmethod
    def ground_property(self, phrase: str, context: str = "") -> list[PropertyCandidate]: ...
