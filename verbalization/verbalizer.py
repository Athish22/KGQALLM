"""Verbalizer: template-based NL generation for common result shapes
(single value, list, count, boolean). Always attaches the SPARQL query
executed. Distinguishes "no such fact" (the predicates used are real,
declared properties in this KG -- the data just doesn't have a match)
from "not represented in this KG" (a predicate in the failed query isn't
a declared property at all) -- this needs schema knowledge, so it's a
best-effort regex extraction of prefixed-name tokens against
`adapter.get_properties()` rather than a real query-plan walk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kgqa.adapters.base import KGAdapter, QueryOutcome
from kgqa.execution.executor import ExecutionResult

_PREFIXED_TOKEN = re.compile(r"\b([a-zA-Z][\w-]*):([A-Za-z_][\w-]*)\b")
_NON_PROPERTY_PREFIXES = {"xsd", "d"}  # data-value / individual namespaces, not predicates


@dataclass
class VerbalizedAnswer:
    text: str
    sparql_query: str
    result_shape: str  # "single_value" | "list" | "count" | "boolean" | "not_found"
    not_found_reason: str | None = None  # "no_such_fact" | "not_represented" | None


class Verbalizer:
    def verbalize(self, execution_result: ExecutionResult, adapter: KGAdapter | None = None) -> VerbalizedAnswer:
        if execution_result.outcome == QueryOutcome.SUCCESS and execution_result.query_result:
            return self._verbalize_success(execution_result)
        return self._verbalize_not_found(execution_result, adapter)

    # --- success shapes ---

    def _verbalize_success(self, execution_result: ExecutionResult) -> VerbalizedAnswer:
        result = execution_result.query_result
        rows, variables = result.rows, result.variables

        if variables == ["boolean"]:
            value = rows[0]["boolean"] if rows else "false"
            text = "Yes." if value.lower() == "true" else "No."
            return VerbalizedAnswer(text=text, sparql_query=result.raw_query, result_shape="boolean")

        count_vars = [v for v in variables if "count" in v.lower()]
        if count_vars and len(rows) <= 1:
            row = rows[0] if rows else {}
            parts = [f"{v.replace('_', ' ')}: {row.get(v, '0')}" for v in count_vars]
            text = "; ".join(parts) + "."
            return VerbalizedAnswer(text=text, sparql_query=result.raw_query, result_shape="count")

        if len(rows) == 1 and len(variables) == 1:
            var = variables[0]
            text = f"{rows[0][var]}"
            return VerbalizedAnswer(text=text, sparql_query=result.raw_query, result_shape="single_value")

        lines = [", ".join(f"{v}={row.get(v, '')}" for v in variables) for row in rows]
        text = f"{len(rows)} result(s):\n" + "\n".join(f"- {line}" for line in lines)
        return VerbalizedAnswer(text=text, sparql_query=result.raw_query, result_shape="list")

    # --- not-found ---

    def _verbalize_not_found(self, execution_result: ExecutionResult, adapter: KGAdapter | None) -> VerbalizedAnswer:
        if execution_result.outcome == QueryOutcome.SYNTAX_ERROR:
            error = execution_result.query_result.error if execution_result.query_result else None
            text = f"Couldn't construct a valid query for this question (syntax error: {error})."
            return VerbalizedAnswer(
                text=text,
                sparql_query=execution_result.final_query,
                result_shape="not_found",
                not_found_reason=None,
            )

        reason = self._classify_not_found(execution_result.final_query, adapter) if adapter else None
        if reason == "not_represented":
            text = "This isn't represented in this knowledge graph -- no such property/class is modeled."
        else:
            text = "No matching facts found in this knowledge graph for this question."
        return VerbalizedAnswer(
            text=text,
            sparql_query=execution_result.final_query,
            result_shape="not_found",
            not_found_reason=reason or "no_such_fact",
        )

    @staticmethod
    def _classify_not_found(query: str, adapter: KGAdapter) -> str:
        declared = {p.uri for p in adapter.get_properties()}
        prefixes = adapter.get_prefixes()
        referenced_local_names = {
            (prefix, local)
            for prefix, local in _PREFIXED_TOKEN.findall(query)
            if prefix not in _NON_PROPERTY_PREFIXES and prefix in prefixes
        }
        for prefix, local in referenced_local_names:
            full_uri = prefixes[prefix] + local
            if full_uri not in declared:
                return "not_represented"
        return "no_such_fact"
