"""KGAdapter: the only KG-specific component.

Every other module in this package must be able to do its job knowing
only these interfaces -- never a specific ontology's URIs or naming
conventions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class QueryOutcome(Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"


@dataclass
class QueryResult:
    outcome: QueryOutcome
    rows: list[dict[str, str]] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    error: str | None = None
    raw_query: str = ""

    @property
    def is_empty(self) -> bool:
        return self.outcome == QueryOutcome.SUCCESS and not self.rows


@dataclass
class ClassInfo:
    uri: str
    label: str | None = None
    comment: str | None = None


@dataclass
class PropertyInfo:
    uri: str
    label: str | None = None
    comment: str | None = None
    domain: list[str] = field(default_factory=list)
    range: list[str] = field(default_factory=list)
    is_object_property: bool = True


@dataclass
class EntityInfo:
    uri: str
    class_uri: str | None = None
    labels: list[str] = field(default_factory=list)


@dataclass
class SeedQuery:
    question: str | None
    sparql: str
    name: str | None = None


class KGAdapter(ABC):
    """Abstract base for a single knowledge graph's connection + schema access.

    Concrete adapters (e.g. ManufacturingKGAdapter) are the *only* place
    that should know this graph's prefixes, class names, or property
    names.
    """

    # --- Connection ---
    @abstractmethod
    def execute_sparql(self, query: str) -> QueryResult: ...

    @abstractmethod
    def health_check(self) -> bool: ...

    # --- Schema introspection ---
    @abstractmethod
    def get_classes(self) -> list[ClassInfo]: ...

    @abstractmethod
    def get_properties(self) -> list[PropertyInfo]: ...

    @abstractmethod
    def get_prefixes(self) -> dict[str, str]: ...

    # --- Entity access ---
    @abstractmethod
    def get_entity_count(self, nameable_only: bool = False) -> int: ...

    @abstractmethod
    def get_entities(
        self,
        class_uri: str | None = None,
        nameable_only: bool = False,
        limit: int | None = None,
    ) -> Iterator[EntityInfo]: ...

    @abstractmethod
    def get_entity_labels(self, uri: str) -> list[str]: ...

    @abstractmethod
    def get_nameable_classes(self) -> list[str]:
        """Class URIs the adapter author has identified as stable,
        addressable-by-name entities (machines, sensors, operators, ...)
        as opposed to generated event/observation classes.

        This is adapter-specific knowledge with no generic heuristic.
        """

    # --- Optional, used when available ---
    def get_seed_queries(self) -> list[SeedQuery]:
        return []

    def detect_temporal_properties(self) -> list[str]:
        return []

    def get_entity_by_local_name(self, local_name: str) -> EntityInfo | None:
        """Exact lookup by an individual's local URI name, independent of
        `get_nameable_classes()` curation -- for when a question names an
        exact identifier verbatim rather than describing it in natural
        language. Default: unsupported. See `RDFLibAdapter`'s override."""
        return None
