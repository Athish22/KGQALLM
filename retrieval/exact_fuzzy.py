"""Tier A retrieval strategy: exact/fuzzy string match against a
hand-buildable alias table. No external dependencies --
uses stdlib `difflib` for fuzzy scoring, sufficient for the reference KG's
~100-150 nameable entities.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from kgqa.adapters.base import KGAdapter
from kgqa.retrieval.base import EntityCandidate, PropertyCandidate, RetrievalStrategy

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")


def _local_name(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _humanize(local_name: str) -> str:
    words = _CAMEL_SPLIT.sub(" ", local_name).replace("_", " ")
    return words.lower().strip()


class ExactFuzzyStrategy(RetrievalStrategy):
    def __init__(self, synonym_overrides: dict[str, list[str]] | None = None, min_score: float = 0.6):
        self.synonym_overrides = synonym_overrides or {}
        self.min_score = min_score
        self._entity_aliases: list[tuple[str, str, str]] = []  # (alias, uri, canonical_label)
        self._property_aliases: list[tuple[str, str, str]] = []

    def build_index(self, adapter: KGAdapter) -> None:
        self._entity_aliases = []
        for class_uri in adapter.get_nameable_classes():
            for entity in adapter.get_entities(class_uri=class_uri, nameable_only=True):
                # Always alias the humanized local URI name ("Machine 2")
                # alongside any data-level hasName/label ("Oval printing
                # Machine") -- they can differ, and users refer to either.
                labels = list(dict.fromkeys([*entity.labels, _humanize(_local_name(entity.uri))]))
                canonical = entity.labels[0] if entity.labels else labels[0]
                for label in labels:
                    self._entity_aliases.append((label.lower(), entity.uri, canonical))
                for extra in self.synonym_overrides.get(entity.uri, []):
                    self._entity_aliases.append((extra.lower(), entity.uri, canonical))

        self._property_aliases = []
        for prop in adapter.get_properties():
            local = _local_name(prop.uri)
            labels = {_humanize(local)}
            if prop.label:
                labels.add(prop.label.lower())
            canonical = prop.label or _humanize(local)
            for label in labels:
                self._property_aliases.append((label, prop.uri, canonical))
            for extra in self.synonym_overrides.get(prop.uri, []):
                self._property_aliases.append((extra.lower(), prop.uri, canonical))

    def ground_entity(self, mention: str, context: str = "") -> list[EntityCandidate]:
        return self._rank(mention, self._entity_aliases, EntityCandidate)

    def ground_property(self, phrase: str, context: str = "") -> list[PropertyCandidate]:
        return self._rank(phrase, self._property_aliases, PropertyCandidate)

    def _rank(self, query: str, aliases: list[tuple[str, str, str]], candidate_cls):
        query_l = query.lower().strip()
        scored: dict[str, tuple[float, str]] = {}
        for alias, uri, canonical in aliases:
            score = 1.0 if alias == query_l else SequenceMatcher(None, alias, query_l).ratio()
            if score < self.min_score:
                continue
            if uri not in scored or score > scored[uri][0]:
                scored[uri] = (score, canonical)
        ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)
        return [
            candidate_cls(uri=uri, label=canonical, score=round(score, 3))
            for uri, (score, canonical) in ranked
        ]
