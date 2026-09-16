"""KGProfiler: runs once per new KG, picks the retrieval tier.

Scored on nameable entity count (`get_entity_count(nameable_only=True)`),
not raw individual count: generated event/observation individuals would
otherwise push a Tier-A-shaped KG straight to Tier C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kgqa.adapters.base import KGAdapter

ENTITY_TIER_A_MAX = 500
ENTITY_TIER_B_MAX = 100_000
PROPERTY_EMBEDDING_THRESHOLD = 150


@dataclass
class KGProfile:
    nameable_entity_count: int
    total_individual_count: int
    property_count: int
    entity_tier: str  # "A" | "B" | "C"
    property_tier: str  # "in_context" | "embedding"
    schema_card_fits_in_context: bool
    detected_temporal_properties: list[str] = field(default_factory=list)


class KGProfiler:
    def profile(self, adapter: KGAdapter) -> KGProfile:
        nameable_count = adapter.get_entity_count(nameable_only=True)
        total_count = adapter.get_entity_count(nameable_only=False)
        properties = adapter.get_properties()
        property_count = len(properties)

        entity_tier = self._entity_tier(nameable_count)
        property_tier = (
            "embedding" if property_count > PROPERTY_EMBEDDING_THRESHOLD else "in_context"
        )

        return KGProfile(
            nameable_entity_count=nameable_count,
            total_individual_count=total_count,
            property_count=property_count,
            entity_tier=entity_tier,
            property_tier=property_tier,
            schema_card_fits_in_context=(property_tier == "in_context"),
            detected_temporal_properties=adapter.detect_temporal_properties(),
        )

    @staticmethod
    def _entity_tier(nameable_count: int) -> str:
        if nameable_count < ENTITY_TIER_A_MAX:
            return "A"
        if nameable_count < ENTITY_TIER_B_MAX:
            return "B"
        return "C"
