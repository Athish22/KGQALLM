"""Gradio UI for the KGQA pipeline (kgqa.pipeline.KGQAPipeline): upload
any Turtle/RDF-XML/N3/N-Triples knowledge graph file, wait for it to
load, then ask questions against it in natural language.

Adapter selection is automatic (kgqa.pipeline._pick_adapter_class): the
reference Industry 4.0 production-line ontology gets the curated
`ManufacturingKGAdapter`; anything else gets `GenericRDFAdapter`, which
grounds entities via a generic name-coverage heuristic instead of
hand-curated knowledge (see `RDFLibAdapter.get_nameable_classes`) --
lower precision, but genuinely KG-agnostic.

Questions that don't match one of the 5 templates seeded from the
reference KG's example queries go through LLM fallback
(kgqa.construction.llm_fallback) when a Hugging Face API token is
supplied -- this is the only path available at all for a KG other than
the reference one, since the templates are specific to that one
ontology's predicates.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from kgqa.pipeline import KGQAPipeline, _pick_adapter_class

MAX_DISPLAY_ROWS = 200
RDF_FILE_TYPES = [".owl", ".ttl", ".rdf", ".xml", ".n3", ".nt", ".jsonld"]


def load_kg(file, hf_token: str, hf_model: str, hf_provider: str):
    if file is None:
        return None, "Upload a knowledge graph file first.", ""

    ttl_path = Path(file if isinstance(file, str) else file.name)
    adapter_cls = _pick_adapter_class(ttl_path)

    try:
        pipeline = KGQAPipeline(
            ttl_path,
            llm_token=hf_token.strip() or None,
            llm_model=hf_model.strip() or None,
            llm_provider=hf_provider.strip() or None,
        )
    except Exception as exc:
        return None, f"**Failed to load KG:** {exc}\n\n```\n{traceback.format_exc()}\n```", ""

    p = pipeline.profile
    llm_line = (
        f"LLM fallback: **on** (model `{pipeline.llm_fallback.model}` via `{pipeline.llm_fallback.provider}`)"
        if pipeline.llm_fallback
        else "LLM fallback: **off** (no HF token given -- only the 5 built-in templates will answer)"
    )
    status = (
        f"**Loaded `{ttl_path.name}`** using `{adapter_cls.__name__}`\n\n"
        f"- Nameable entities: {p.nameable_entity_count} (total individuals: {p.total_individual_count:,})\n"
        f"- Properties: {p.property_count}\n"
        f"- Entity retrieval tier: **{p.entity_tier}**  \n"
        f"- Property retrieval mode: **{p.property_tier}**  (schema fits in-context: {p.schema_card_fits_in_context})\n"
        f"- Detected temporal properties: {len(p.detected_temporal_properties)}\n"
        f"- {llm_line}\n"
    )
    return pipeline, status, "Ready. Ask a question below."


def ask(pipeline: KGQAPipeline | None, question: str):
    if pipeline is None:
        return "Load a knowledge graph first.", "", ""
    if not question or not question.strip():
        return "Type a question first.", "", ""

    try:
        answer = pipeline.answer(question)
    except NotImplementedError as exc:
        return f"**Not answerable yet:** {exc}", "", ""
    except Exception as exc:
        return f"**Error:** {exc}\n\n```\n{traceback.format_exc()}\n```", "", ""

    text = answer.text
    lines = text.splitlines()
    if len(lines) > MAX_DISPLAY_ROWS:
        text = "\n".join(lines[:MAX_DISPLAY_ROWS]) + f"\n... ({len(lines) - MAX_DISPLAY_ROWS} more rows truncated)"

    meta = f"shape: `{answer.result_shape}`"
    if answer.not_found_reason:
        meta += f" | reason: `{answer.not_found_reason}`"

    return text, answer.sparql_query, meta


with gr.Blocks(title="KGQA") as demo:
    gr.Markdown("# Knowledge Graph Question Answering")
    gr.Markdown(
        "Upload any Turtle/RDF-XML/N3/N-Triples knowledge graph file, then ask "
        "a question. The Industry 4.0 reference KG gets curated entity "
        "grounding and 5 built-in query templates automatically; any other KG "
        "is grounded generically and answered entirely through the LLM "
        "fallback below. Loading can take a minute or two for large files."
    )

    pipeline_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Knowledge graph file", file_types=RDF_FILE_TYPES)
            with gr.Accordion("LLM fallback (Hugging Face) -- optional but required for anything outside the 5 built-in templates", open=False):
                hf_token_input = gr.Textbox(
                    label="Hugging Face API token",
                    type="password",
                    placeholder="hf_...",
                )
                hf_model_input = gr.Textbox(
                    label="Model",
                    value="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                )
                hf_provider_input = gr.Textbox(
                    label="Provider",
                    value="featherless-ai",
                    info="Which (model, provider) pairs are live on HF's router shifts over time and "
                    "depends on your token's scope -- if loading fails with a 'not a chat model' or "
                    "404 error, check huggingface_hub.HfApi().model_info(model, "
                    "expand='inferenceProviderMapping') for a currently-live pair.",
                )
            load_btn = gr.Button("Load knowledge graph", variant="primary")
            profile_output = gr.Markdown()

        with gr.Column(scale=2):
            load_status = gr.Markdown("No knowledge graph loaded yet.")
            question_input = gr.Textbox(
                label="Question",
                placeholder="e.g. What tools does Machine 2 have?",
                lines=1,
            )
            ask_btn = gr.Button("Ask")
            answer_output = gr.Markdown(label="Answer")
            meta_output = gr.Markdown()
            sparql_output = gr.Code(label="SPARQL query executed", language="sql")

    load_btn.click(
        fn=load_kg,
        inputs=[file_input, hf_token_input, hf_model_input, hf_provider_input],
        outputs=[pipeline_state, profile_output, load_status],
    )
    ask_btn.click(
        fn=ask,
        inputs=[pipeline_state, question_input],
        outputs=[answer_output, sparql_output, meta_output],
    )
    question_input.submit(
        fn=ask,
        inputs=[pipeline_state, question_input],
        outputs=[answer_output, sparql_output, meta_output],
    )

if __name__ == "__main__":
    demo.launch()
