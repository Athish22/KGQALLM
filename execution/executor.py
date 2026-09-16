"""Executor: runs the query, classifies the outcome, and on
`empty`/`syntax_error` routes back to the query constructor with a
diagnostic payload and a bounded retry budget. On repeated failure,
returns a clear "not found in this KG" result rather than a fabricated
answer.

Constructor-agnostic: takes a `retry_fn` callback rather than depending on
`TemplateBank`/`LLMFallbackConstructor` directly, so it works the same way
whichever one produced the failing query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kgqa.adapters.base import KGAdapter, QueryOutcome, QueryResult

DEFAULT_MAX_RETRIES = 2


@dataclass
class DiagnosticPayload:
    outcome: QueryOutcome
    error: str | None
    failed_query: str
    attempt: int
    # Best-effort schema hint for the retry: without the constructor
    # threading through which entity types were used (it knows; the raw
    # SPARQL text alone doesn't reliably), this falls back to "some real
    # predicates in this KG" rather than a true neighborhood search.
    known_predicates: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    outcome: QueryOutcome
    query_result: QueryResult | None
    attempts: int
    final_query: str

    @property
    def not_found(self) -> bool:
        return self.outcome != QueryOutcome.SUCCESS


class Executor:
    def __init__(self, adapter: KGAdapter, max_retries: int = DEFAULT_MAX_RETRIES):
        self.adapter = adapter
        self.max_retries = max_retries

    def run(
        self,
        initial_query: str,
        retry_fn: Callable[[DiagnosticPayload], str] | None = None,
    ) -> ExecutionResult:
        query = initial_query
        attempt = 0

        while True:
            attempt += 1
            result = self.adapter.execute_sparql(query)

            if result.outcome == QueryOutcome.SUCCESS and not result.is_empty:
                return ExecutionResult(
                    outcome=QueryOutcome.SUCCESS,
                    query_result=result,
                    attempts=attempt,
                    final_query=query,
                )

            effective_outcome = QueryOutcome.EMPTY if result.is_empty else result.outcome

            if retry_fn is None or attempt > self.max_retries:
                return ExecutionResult(
                    outcome=effective_outcome,
                    query_result=result,
                    attempts=attempt,
                    final_query=query,
                )

            diagnostic = DiagnosticPayload(
                outcome=effective_outcome,
                error=result.error,
                failed_query=query,
                attempt=attempt,
                known_predicates=[p.uri for p in self.adapter.get_properties()[:30]],
            )
            query = retry_fn(diagnostic)
