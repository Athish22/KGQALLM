"""Intent classification:
`lookup | list_filter | aggregate | multi_hop | compare`.

Rule-based baseline -- cheap, deterministic, and good enough to route into
the template bank for the reference KG's question shapes. Swap for an
LLM/ML classifier once an eval set shows the keyword rules under-perform
on real question variety; the `QuestionAnalyzer` contract (a single
`Intent` string) doesn't change either way.
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    LOOKUP = "lookup"
    LIST_FILTER = "list_filter"
    AGGREGATE = "aggregate"
    MULTI_HOP = "multi_hop"
    COMPARE = "compare"


_AGGREGATE = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal\b|\baverage\b|\bavg\b", re.I)
_COMPARE = re.compile(r"\bcompare\b|\bversus\b|\bvs\.?\b|\bmore than\b|\bless than\b|\bhigher than\b", re.I)
_LIST = re.compile(r"^\s*(which|what are|list|show me all|give me all)\b", re.I)
# Heuristic multi-hop signal: the question chains a relation off of
# something that is itself the result of a relation ("the X of the Y of
# Z", "that", "whose") rather than naming one entity + one relation.
_MULTI_HOP_HINT = re.compile(r"\bwhose\b|\bthat\s+\w+\s+the\b|\bof the\b.*\bof\b", re.I)


class IntentClassifier:
    def classify(self, question: str) -> Intent:
        if _AGGREGATE.search(question):
            return Intent.AGGREGATE
        if _COMPARE.search(question):
            return Intent.COMPARE
        if _MULTI_HOP_HINT.search(question):
            return Intent.MULTI_HOP
        if _LIST.search(question):
            return Intent.LIST_FILTER
        return Intent.LOOKUP
