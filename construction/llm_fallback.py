"""LLM fallback query construction: receives only (a) the schema card or
top-k retrieved schema elements, (b) the grounded candidate URIs from
retrieval, (c) any path found by `SchemaPathFinder`, (d) the question.
Must NOT have access to the full raw ontology dump -- only the grounded
(+ schema-connected) subset -- to keep the hallucination surface small.

Provider: Hugging Face Inference API, chat-completion against an
instruct-tuned Mistral model. Grammar-constrained decoding isn't
available through this API, so syntax validation is post-hoc: the model
is asked to emit only the SPARQL body (no PREFIX lines -- we generate
those from the adapter's real prefixes, not the model's guess), and
`Executor`'s retry loop is what actually catches and corrects a bad
query, not this module.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from huggingface_hub import InferenceClient

from kgqa.construction.path_finder import PropertyPath
from kgqa.execution.executor import DiagnosticPayload
from kgqa.retrieval.base import EntityCandidate, PropertyCandidate

DEFAULT_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
DEFAULT_PROVIDER = "featherless-ai"

_CODE_FENCE = re.compile(r"^```(?:sparql|SPARQL)?\s*|\s*```$", re.MULTILINE)

_SYSTEM_PROMPT = """You are a SPARQL query generator for a knowledge graph question-answering system.

Rules:
- Use ONLY the classes, properties, and entity URIs given to you below. NEVER invent a URI, class, or property that isn't listed.
- Output ONLY the SPARQL query body (starting with SELECT, ASK, or SELECT DISTINCT). Do NOT include PREFIX declarations -- they are added separately.
- Do NOT wrap the output in markdown code fences. Do NOT explain your answer. Output the query and nothing else.
- If the question needs an entity, class, or property that is not in the provided list, use the closest reasonable match from the list rather than inventing one.
"""


@dataclass
class LLMFallbackContext:
    question: str
    prefixes: dict[str, str]
    schema_card: str  # in-context schema, or top-k retrieved schema elements
    grounded_entities: list[EntityCandidate] = field(default_factory=list)
    grounded_properties: list[PropertyCandidate] = field(default_factory=list)
    discovered_path: PropertyPath | None = None
    diagnostic: DiagnosticPayload | None = None  # set on a retry attempt


def _strip_code_fences(text: str) -> str:
    return _CODE_FENCE.sub("", text.strip()).strip()


class LLMFallbackConstructor:
    def __init__(
        self,
        token: str | None = None,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
    ):
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.token:
            raise ValueError(
                "No Hugging Face API token found. Pass token=... explicitly, "
                "or set the HF_TOKEN environment variable."
            )
        self.model = model
        self.provider = provider
        # Which (model, provider) pairs are actually live on HF's router
        # shifts over time and depends on the token's own scope (a
        # fine-grained token may only be authorized for specific models).
        # If this combination 404s, check what's currently live for your
        # token with `huggingface_hub.HfApi().model_info(model,
        # expand="inferenceProviderMapping")` and pass a working
        # model/provider pair explicitly.
        self._client = InferenceClient(model=self.model, token=self.token, provider=self.provider)

    def construct(self, context: LLMFallbackContext) -> str:
        prefix_header = "\n".join(f"PREFIX {p}: <{uri}>" for p, uri in context.prefixes.items())
        user_content = self._build_user_content(context)

        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=500,
            temperature=0.0,
        )
        body = _strip_code_fences(response.choices[0].message.content)
        return f"{prefix_header}\n{body}"

    @staticmethod
    def _build_user_content(context: LLMFallbackContext) -> str:
        parts = [f"Schema (available classes and properties):\n{context.schema_card}"]

        if context.grounded_entities:
            entity_lines = "\n".join(
                f"- {c.label!r} -> <{c.uri}>"
                + (f", rdf:type <{c.class_uri}>" if c.class_uri else "")
                + f" (confidence {c.score})"
                for c in context.grounded_entities
            )
            parts.append(
                "Grounded entities mentioned in the question, with their actual type in this KG "
                "(if the entity's own type already matches what's being asked about, query its "
                "own properties directly -- don't assume another entity must point to it):\n"
                f"{entity_lines}"
            )

        if context.grounded_properties:
            prop_lines = "\n".join(
                f"- {c.label!r} -> <{c.uri}> (confidence {c.score})" for c in context.grounded_properties
            )
            parts.append(f"Grounded properties possibly relevant to the question:\n{prop_lines}")

        if context.discovered_path:
            path_lines = " -> ".join(
                f"{'<' + h.property_uri + '>'}{'(inverse)' if not h.forward else ''}"
                for h in context.discovered_path.hops
            )
            parts.append(
                "A property path was found connecting the grounded entity to the target:\n"
                f"{path_lines}\nending at class <{context.discovered_path.end_class}>. "
                "Consider using this chain of properties."
            )

        if context.diagnostic:
            d = context.diagnostic
            parts.append(
                "Your previous attempt failed. Fix it.\n"
                f"Previous query:\n{d.failed_query}\n"
                f"Outcome: {d.outcome.value}"
                + (f", error: {d.error}" if d.error else "")
                + "\nSome real predicates that do exist in this KG (for reference): "
                + ", ".join(f"<{p}>" for p in d.known_predicates[:20])
            )

        parts.append(f"Question: {context.question}")
        return "\n\n".join(parts)
