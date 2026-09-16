"""Wires all components together. KG Adapter + Profiler, Tier A
retrieval, intent + temporal analysis, template bank + SchemaPathFinder,
executor, and verbalizer are wired and runnable against the reference KG.
Multi-hop decomposition is NOT implemented -- see
`kgqa.analysis.decomposition`.

LLM fallback IS wired when a Hugging Face API token is supplied -- see
`kgqa.construction.llm_fallback`. Without one, `answer()` still works for
anything the template bank covers and raises clearly for anything else,
same as before.

Adapter selection: `ManufacturingKGAdapter` (curated `get_nameable_classes`)
is used automatically when the uploaded file declares the reference KG's
namespace; any other file gets `GenericRDFAdapter` (generic name-coverage
heuristic, lower precision -- see `RDFLibAdapter.get_nameable_classes`).
The template bank is seeded from this one KG's example queries, so on a
different KG it simply never matches and every question goes through LLM
fallback, which is the intended long-tail path, not a special case.

Entity/property-mention extraction here is a minimal placeholder (regex
n-grams + fuzzy grounding), not a fully specified "entity/relation mention
spans" component -- it exists only so `answer()` is runnable end-to-end.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path

from kgqa.adapters.base import KGAdapter
from kgqa.adapters.generic_rdf import GenericRDFAdapter
from kgqa.adapters.manufacturing_kg import SMO, ManufacturingKGAdapter
from kgqa.analysis.intent import Intent, IntentClassifier
from kgqa.analysis.temporal import TemporalParser
from kgqa.construction.llm_fallback import LLMFallbackConstructor, LLMFallbackContext
from kgqa.construction.path_finder import SchemaPathFinder
from kgqa.construction.template_bank import TemplateBank
from kgqa.execution.executor import DiagnosticPayload, Executor
from kgqa.profiler.profiler import KGProfile, KGProfiler
from kgqa.retrieval.base import EntityCandidate, PropertyCandidate, RetrievalStrategy
from kgqa.retrieval.exact_fuzzy import ExactFuzzyStrategy
from kgqa.retrieval.full_rag import FullRAGStrategy
from kgqa.retrieval.light_embedding import LightEmbeddingStrategy
from kgqa.verbalization.verbalizer import Verbalizer, VerbalizedAnswer

STRATEGY_BY_TIER = {
    "A": ExactFuzzyStrategy,
    "B": LightEmbeddingStrategy,
    "C": FullRAGStrategy,
}

_MENTION_CANDIDATE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\s+\d+)?")
_MIN_TEMPLATE_MENTION_SCORE = 0.75  # strict: feeds a template slot verbatim
_MIN_CONTEXT_MENTION_SCORE = 0.5  # looser: the LLM can weigh/discard noise itself
_MAX_CONTEXT_CANDIDATES = 8
_MAX_SCHEMA_PROPERTIES = 60  # placeholder cap until Tier B/C embedding retrieval exists

# Templates 3/4 use a generic "tool_uri" for any hasTool object (motor,
# sensor, die, heater, ...); only Processing/AssemblingMachine gets its
# own slot. Manufacturing-KG-specific mapping, only meaningful when that
# adapter is in use.
_MACHINE_CLASSES = {str(SMO.ProcessingMachine), str(SMO.AssemblingMachine)}

_MANUFACTURING_NAMESPACE_HINT = "manufacturingproductionline"


def _local_name(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _pick_adapter_class(ttl_path: Path) -> type[KGAdapter]:
    try:
        head = ttl_path.open("rb").read(65536).decode("utf-8", errors="ignore")
    except OSError:
        return GenericRDFAdapter
    return ManufacturingKGAdapter if _MANUFACTURING_NAMESPACE_HINT in head else GenericRDFAdapter


class KGQAPipeline:
    def __init__(
        self,
        ttl_path: str | Path,
        seed_query_dir: str | Path | None = None,
        adapter_cls: type[KGAdapter] | None = None,
        llm_token: str | None = None,
        llm_model: str | None = None,
        llm_provider: str | None = None,
    ):
        ttl_path = Path(ttl_path)
        adapter_cls = adapter_cls or _pick_adapter_class(ttl_path)
        self.adapter: KGAdapter = adapter_cls(ttl_path, seed_query_dir=seed_query_dir)
        self.profile: KGProfile = KGProfiler().profile(self.adapter)

        strategy_cls = STRATEGY_BY_TIER[self.profile.entity_tier]
        self.retrieval: RetrievalStrategy = strategy_cls()
        self.retrieval.build_index(self.adapter)

        self.intent_classifier = IntentClassifier()
        self.temporal_parser = TemporalParser()
        self.template_bank = TemplateBank()
        self.executor = Executor(self.adapter)
        self.verbalizer = Verbalizer()

        self.path_finder = SchemaPathFinder()
        self.path_finder.build_graph(self.adapter)

        self.llm_fallback: LLMFallbackConstructor | None = None
        if llm_token or os.environ.get("HF_TOKEN"):
            kwargs = {"token": llm_token} if llm_token else {}
            if llm_model:
                kwargs["model"] = llm_model
            if llm_provider:
                kwargs["provider"] = llm_provider
            self.llm_fallback = LLMFallbackConstructor(**kwargs)

        self._entity_class: dict[str, str] = {
            entity.uri: class_uri
            for class_uri in self.adapter.get_nameable_classes()
            for entity in self.adapter.get_entities(class_uri=class_uri, nameable_only=True)
        }
        self._schema_card = self._build_schema_card()

    def answer(self, question: str) -> VerbalizedAnswer:
        intent = self.intent_classifier.classify(question)
        template_params = self._ground_template_params(question)

        temporal_range = self.temporal_parser.parse(question)
        if temporal_range and temporal_range.start:
            template_params["start_time"] = temporal_range.start
        if temporal_range and temporal_range.end:
            template_params["end_time"] = temporal_range.end

        template = self.template_bank.find(intent, set(template_params), question=question)
        if template is not None:
            sparql = template.render(template_params)
            execution_result = self.executor.run(sparql)  # no LLM retry needed: template already grounded
            return self.verbalizer.verbalize(execution_result, adapter=self.adapter)

        if self.llm_fallback is None:
            raise NotImplementedError(
                f"No template matches intent={intent.value} with params={sorted(template_params)}, "
                "and no LLM fallback is configured (pass llm_token=... / set HF_TOKEN)."
            )

        context = self._build_llm_context(question, temporal_range)
        sparql = self.llm_fallback.construct(context)

        def retry_fn(diagnostic: DiagnosticPayload) -> str:
            return self.llm_fallback.construct(replace(context, diagnostic=diagnostic))

        execution_result = self.executor.run(sparql, retry_fn=retry_fn)
        return self.verbalizer.verbalize(execution_result, adapter=self.adapter)

    # --- grounding ---

    def _exact_entity_candidates(self, question: str) -> list[EntityCandidate]:
        """A question that names an exact identifier verbatim (e.g.
        copy-pasted from a KG dump, like `Machine3_Power_1182`) deserves a
        direct lookup, independent of `get_nameable_classes()` curation --
        see `KGAdapter.get_entity_by_local_name`. Without this, such a
        mention is invisible to grounding entirely (Tier A only indexes
        nameable-class individuals), and the LLM fallback would have to
        hallucinate a query shape with no grounded entity to anchor on.
        """
        found: dict[str, EntityCandidate] = {}
        for match in _MENTION_CANDIDATE.finditer(question):
            entity = self.adapter.get_entity_by_local_name(match.group(0))
            if entity is not None:
                label = entity.labels[0] if entity.labels else _local_name(entity.uri)
                found[entity.uri] = EntityCandidate(
                    uri=entity.uri, label=label, score=1.0, class_uri=entity.class_uri
                )
        return list(found.values())

    def _ground_template_params(self, question: str) -> dict[str, str]:
        best: tuple[float, str, str] | None = None  # (score, slot_name, uri)

        def consider(candidate: EntityCandidate) -> None:
            nonlocal best
            if best is None or candidate.score > best[0]:
                class_uri = self._entity_class.get(candidate.uri)
                slot = "machine_uri" if class_uri in _MACHINE_CLASSES else "tool_uri"
                best = (candidate.score, slot, candidate.uri)

        for candidate in self._exact_entity_candidates(question):
            consider(candidate)
        for match in _MENTION_CANDIDATE.finditer(question):
            for candidate in self.retrieval.ground_entity(match.group(0)):
                if candidate.score >= _MIN_TEMPLATE_MENTION_SCORE:
                    consider(candidate)
        return {best[1]: best[2]} if best else {}

    def _ground_context_candidates(self, question: str):
        best_entities: dict[str, EntityCandidate] = {}
        best_properties: dict[str, PropertyCandidate] = {}

        for c in self._exact_entity_candidates(question):
            best_entities[c.uri] = c  # exact match always wins, score 1.0

        for match in _MENTION_CANDIDATE.finditer(question):
            mention = match.group(0)
            for c in self.retrieval.ground_entity(mention):
                if c.score >= _MIN_CONTEXT_MENTION_SCORE and (c.uri not in best_entities or c.score > best_entities[c.uri].score):
                    c.class_uri = c.class_uri or self._entity_class.get(c.uri)
                    best_entities[c.uri] = c
            for c in self.retrieval.ground_property(mention):
                if c.score >= _MIN_CONTEXT_MENTION_SCORE and (c.uri not in best_properties or c.score > best_properties[c.uri].score):
                    best_properties[c.uri] = c

        entities = sorted(best_entities.values(), key=lambda c: c.score, reverse=True)[:_MAX_CONTEXT_CANDIDATES]
        properties = sorted(best_properties.values(), key=lambda c: c.score, reverse=True)[:_MAX_CONTEXT_CANDIDATES]
        return entities, properties

    def _build_llm_context(self, question: str, temporal_range) -> LLMFallbackContext:
        entities, properties = self._ground_context_candidates(question)

        discovered_path = None
        if entities and properties:
            source_class = self._entity_class.get(entities[0].uri)
            if source_class:
                paths = self.path_finder.find_paths(source_class, properties[0].uri, max_hops=3)
                if paths:
                    discovered_path = paths[0]

        question_with_time = question
        if temporal_range and (temporal_range.start or temporal_range.end):
            question_with_time += (
                f"\n(Parsed time range: start={temporal_range.start}, end={temporal_range.end})"
            )

        return LLMFallbackContext(
            question=question_with_time,
            prefixes=self.adapter.get_prefixes(),
            schema_card=self._schema_card,
            grounded_entities=entities,
            grounded_properties=properties,
            discovered_path=discovered_path,
        )

    def _build_schema_card(self) -> str:
        lines = ["Classes:"]
        for c in self.adapter.get_classes():
            label = c.label or _local_name(c.uri)
            lines.append(f"- {label}: <{c.uri}>")

        lines.append("\nProperties:")
        props = self.adapter.get_properties()
        if not self.profile.schema_card_fits_in_context:
            props = props[:_MAX_SCHEMA_PROPERTIES]
        for p in props:
            label = p.label or _local_name(p.uri)
            domain = ", ".join(_local_name(d) for d in p.domain) or "?"
            range_ = ", ".join(_local_name(r) for r in p.range) or "?"
            lines.append(f"- {label}: <{p.uri}> ({domain} -> {range_})")
        return "\n".join(lines)


def _demo() -> None:
    root = Path(__file__).resolve().parent.parent
    ttl_path = root / "10-Days-KGs.owl"
    pipeline = KGQAPipeline(ttl_path, seed_query_dir=root)

    print(f"KG profile: {pipeline.profile}\n")

    questions = [
        "List all the processing and assembling machines by name.",
        "What tools does Machine 2 have?",
    ]
    for q in questions:
        print(f"Q: {q}")
        try:
            answer = pipeline.answer(q)
            print(f"A ({answer.result_shape}): {answer.text}\n")
        except NotImplementedError as exc:
            print(f"[not implemented] {exc}\n")


if __name__ == "__main__":
    _demo()
