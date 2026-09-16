"""Eval harness: execution-accuracy based, not string match -- run the
generated query and the expected query,
compare actual *results*, since equivalent SPARQL can be written many
different syntactic ways.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kgqa.adapters.base import KGAdapter, QueryOutcome


@dataclass
class EvalItem:
    question: str
    expected_sparql: str
    name: str | None = None


@dataclass
class EvalCaseResult:
    item: EvalItem
    passed: bool
    actual_query: str | None
    detail: str


def load_dataset(path: str | Path) -> list[EvalItem]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalItem(question=d["question"], expected_sparql=d["sparql"], name=d.get("name")) for d in data]


def _rows_as_sets(rows: list[dict[str, str]]) -> set[frozenset]:
    return {frozenset(row.items()) for row in rows}


class EvalHarness:
    def __init__(self, adapter: KGAdapter):
        self.adapter = adapter

    def run(self, dataset: list[EvalItem], pipeline_fn: Callable[[str], str]) -> list[EvalCaseResult]:
        """`pipeline_fn(question) -> sparql` is the system under test (e.g.
        `pipeline.answer_to_sparql`); this harness only owns execution +
        comparison, not question answering itself.
        """
        results = []
        for item in dataset:
            expected = self.adapter.execute_sparql(item.expected_sparql)
            try:
                actual_query = pipeline_fn(item.question)
            except Exception as exc:
                results.append(EvalCaseResult(item=item, passed=False, actual_query=None, detail=f"pipeline error: {exc}"))
                continue

            actual = self.adapter.execute_sparql(actual_query)
            if expected.outcome != QueryOutcome.SUCCESS:
                detail = f"expected query itself failed: {expected.outcome.value}"
                results.append(EvalCaseResult(item=item, passed=False, actual_query=actual_query, detail=detail))
                continue

            passed = actual.outcome == QueryOutcome.SUCCESS and _rows_as_sets(actual.rows) == _rows_as_sets(expected.rows)
            detail = "match" if passed else f"expected {len(expected.rows)} row(s), got {len(actual.rows)} ({actual.outcome.value})"
            results.append(EvalCaseResult(item=item, passed=passed, actual_query=actual_query, detail=detail))
        return results

    @staticmethod
    def accuracy(results: list[EvalCaseResult]) -> float:
        if not results:
            return 0.0
        return sum(r.passed for r in results) / len(results)
