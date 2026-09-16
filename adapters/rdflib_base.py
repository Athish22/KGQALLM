"""RDFLibAdapter: generic rdflib-backed `KGAdapter` mechanics shared by
every adapter that loads a Turtle/RDF/OWL file into an in-memory graph and
runs SPARQL through rdflib's own engine.

This is the KG-agnostic half of the split: everything here is pure
RDF/OWL convention (rdf:type, rdfs:label, owl:Class, domain/range, ...),
never a specific ontology's vocabulary. A concrete KG's adapter (e.g.
`ManufacturingKGAdapter`) subclasses this and overrides only the handful
of methods that benefit from curated, KG-specific knowledge --
`get_nameable_classes()` most of all, since "which classes are
stable/addressable-by-name" has no fully reliable generic signal (see the
heuristic below). `GenericRDFAdapter` subclasses this with zero
overrides, for a KG nobody has hand-curated an adapter for yet.
"""

from __future__ import annotations

from pathlib import Path

from rdflib import RDF, RDFS, OWL, Graph, Literal, URIRef
from rdflib.namespace import DC, DCTERMS, SKOS
from rdflib.plugins.sparql import prepareQuery
from rdflib.plugins.sparql.sparql import SPARQLError

from kgqa.adapters.base import (
    ClassInfo,
    EntityInfo,
    KGAdapter,
    PropertyInfo,
    QueryOutcome,
    QueryResult,
    SeedQuery,
)

_FIXED_NAME_PREDICATES = {RDFS.label, SKOS.prefLabel, DC.title, DCTERMS.title}
_TEMPORAL_XSD_RANGES = {
    "http://www.w3.org/2001/XMLSchema#dateTime",
    "http://www.w3.org/2001/XMLSchema#dateTimeStamp",
    "http://www.w3.org/2001/XMLSchema#date",
    "http://www.w3.org/2001/XMLSchema#time",
}
NAMEABLE_MIN_LABEL_RATIO = 0.5  # fraction of a class's individuals that must carry a name-like literal


def _local_name(uri: str) -> str:
    return uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


class RDFLibAdapter(KGAdapter):
    def __init__(
        self,
        ttl_path: str | Path,
        seed_query_dir: str | Path | None = None,
        rdf_format: str | None = None,
    ):
        self.ttl_path = Path(ttl_path)
        self.seed_query_dir = Path(seed_query_dir) if seed_query_dir else None
        self.graph = Graph()
        self._parse(rdf_format)
        self._entities_by_class: dict[str, set[str]] | None = None
        self._named_subjects_cache: set[str] | None = None
        self._nameable_classes_cache: list[str] | None = None
        self._local_name_index_cache: dict[str, str] | None = None

    def _parse(self, rdf_format: str | None) -> None:
        """A file extension (notably `.owl`) doesn't reliably say which
        RDF serialization is inside -- the reference KG, for instance,
        uses Turtle syntax despite the `.owl` extension. Sniff the actual
        content instead of trusting the extension, then fall back through
        the common serializations if that guess is wrong.
        """
        if rdf_format:
            self.graph.parse(str(self.ttl_path), format=rdf_format)
            return

        head = self.ttl_path.open("rb").read(2048).lstrip()
        if head.startswith(b"<?xml") or head.startswith(b"<rdf:RDF") or head.startswith(b"<"):
            candidates = ["xml", "turtle", "n3", "nt", "json-ld"]
        else:
            candidates = ["turtle", "n3", "nt", "xml", "json-ld"]

        last_error: Exception | None = None
        for fmt in candidates:
            try:
                self.graph.parse(str(self.ttl_path), format=fmt)
                return
            except Exception as exc:  # try the next serialization
                last_error = exc
                self.graph.remove((None, None, None))  # clear any partial parse
        raise ValueError(
            f"Could not parse {self.ttl_path} as any of {candidates}: {last_error}"
        )

    # --- Connection ---

    def execute_sparql(self, query: str) -> QueryResult:
        try:
            prepared = prepareQuery(query)
        except Exception as exc:  # rdflib raises plain Exception/ParseException
            return QueryResult(outcome=QueryOutcome.SYNTAX_ERROR, error=str(exc), raw_query=query)

        try:
            result = self.graph.query(prepared)
        except SPARQLError as exc:
            return QueryResult(outcome=QueryOutcome.SYNTAX_ERROR, error=str(exc), raw_query=query)

        if result.type == "ASK":
            rows = [{"boolean": str(result.askAnswer)}]
            return QueryResult(outcome=QueryOutcome.SUCCESS, rows=rows, variables=["boolean"], raw_query=query)

        variables = [str(v) for v in (result.vars or [])]
        rows = [
            {var: str(row[var]) for var in variables if row[var] is not None}
            for row in result
        ]
        outcome = QueryOutcome.SUCCESS if rows else QueryOutcome.EMPTY
        return QueryResult(outcome=outcome, rows=rows, variables=variables, raw_query=query)

    def health_check(self) -> bool:
        return len(self.graph) > 0

    # --- Schema introspection ---

    def get_classes(self) -> list[ClassInfo]:
        out = []
        for c in self.graph.subjects(RDF.type, OWL.Class):
            if not isinstance(c, URIRef):  # skip blank-node class expressions
                continue
            out.append(
                ClassInfo(
                    uri=str(c),
                    label=self._first(self.graph.objects(c, RDFS.label)),
                    comment=self._first(self.graph.objects(c, RDFS.comment)),
                )
            )
        return out

    def get_properties(self) -> list[PropertyInfo]:
        out = []
        for prop_type, is_obj in ((OWL.ObjectProperty, True), (OWL.DatatypeProperty, False)):
            for p in self.graph.subjects(RDF.type, prop_type):
                out.append(
                    PropertyInfo(
                        uri=str(p),
                        label=self._first(self.graph.objects(p, RDFS.label)),
                        comment=self._first(self.graph.objects(p, RDFS.comment)),
                        domain=[str(d) for d in self.graph.objects(p, RDFS.domain)],
                        range=[str(r) for r in self.graph.objects(p, RDFS.range)],
                        is_object_property=is_obj,
                    )
                )
        return out

    def get_prefixes(self) -> dict[str, str]:
        prefixes = {prefix: str(ns) for prefix, ns in self.graph.namespaces() if prefix}
        prefixes.setdefault("xsd", "http://www.w3.org/2001/XMLSchema#")
        prefixes.setdefault("rdf", str(RDF))
        prefixes.setdefault("rdfs", str(RDFS))
        prefixes.setdefault("owl", str(OWL))
        return prefixes

    # --- Entity access ---

    def get_entity_count(self, nameable_only: bool = False) -> int:
        if nameable_only:
            return sum(len(v) for v in self._class_index().values())
        return sum(1 for _ in self.graph.subjects(RDF.type, OWL.NamedIndividual))

    def get_entities(
        self,
        class_uri: str | None = None,
        nameable_only: bool = False,
        limit: int | None = None,
    ):
        if class_uri is not None:
            uris = {str(s) for s in self.graph.subjects(RDF.type, URIRef(class_uri))}
            if nameable_only:
                uris &= self._class_index().get(class_uri, set())
        elif nameable_only:
            uris = set().union(*self._class_index().values()) if self._class_index() else set()
        else:
            uris = {str(s) for s in self.graph.subjects(RDF.type, OWL.NamedIndividual)}

        count = 0
        for uri in sorted(uris):
            if limit is not None and count >= limit:
                break
            types = [str(t) for t in self.graph.objects(URIRef(uri), RDF.type) if t != OWL.NamedIndividual]
            yield EntityInfo(uri=uri, class_uri=types[0] if types else None, labels=self.get_entity_labels(uri))
            count += 1

    def get_entity_labels(self, uri: str) -> list[str]:
        ref = URIRef(uri)
        labels = set()
        for pred in self._name_predicates():
            for obj in self.graph.objects(ref, pred):
                labels.add(str(obj))
        return sorted(labels)

    def get_entity_by_local_name(self, local_name: str) -> EntityInfo | None:
        """Exact lookup by an individual's local URI name (case-
        insensitive), across ALL named individuals -- deliberately not
        limited to `get_nameable_classes()`. A question that names an
        exact identifier verbatim (e.g. copy-pasted from a KG dump, like
        `Machine3_Power_1182`) is asking for that one triple-store row,
        not describing something in natural language for fuzzy retrieval
        -- curation for *which classes are worth indexing for fuzzy
        matching* shouldn't also gate an exact match.
        """
        if self._local_name_index_cache is None:
            index: dict[str, str] = {}
            for s in self.graph.subjects(RDF.type, OWL.NamedIndividual):
                index[_local_name(str(s)).lower()] = str(s)
            self._local_name_index_cache = index

        uri = self._local_name_index_cache.get(local_name.lower())
        if uri is None:
            return None
        ref = URIRef(uri)
        types = [str(t) for t in self.graph.objects(ref, RDF.type) if t != OWL.NamedIndividual]
        return EntityInfo(uri=uri, class_uri=types[0] if types else None, labels=self.get_entity_labels(uri))

    def get_nameable_classes(self) -> list[str]:
        """Generic default: a class counts as nameable if at least
        `NAMEABLE_MIN_LABEL_RATIO` of its individuals carry a name-like
        literal (rdfs:label, skos:prefLabel, dc:title, or a datatype
        property whose local name contains "name"/"label"/"title").

        This is a real but imprecise heuristic -- it has no way to know
        that, say, every row of a bulk-generated log table happens to
        carry a formulaic "name" field and so isn't actually a stable,
        addressable entity. A KG-specific adapter should override this
        with curated knowledge when precision here matters (see
        `ManufacturingKGAdapter.get_nameable_classes`).
        """
        if self._nameable_classes_cache is None:
            named = self._named_subjects()
            class_to_individuals: dict[str, set[str]] = {}
            for s, o in self.graph.subject_objects(RDF.type):
                o_str = str(o)
                if o_str == str(OWL.NamedIndividual) or not isinstance(o, URIRef):
                    continue
                class_to_individuals.setdefault(o_str, set()).add(str(s))

            nameable = [
                class_uri
                for class_uri, individuals in class_to_individuals.items()
                if individuals and len(individuals & named) / len(individuals) >= NAMEABLE_MIN_LABEL_RATIO
            ]
            self._nameable_classes_cache = sorted(nameable)
        return self._nameable_classes_cache

    # --- Optional, used when available ---

    def get_seed_queries(self) -> list[SeedQuery]:
        if not self.seed_query_dir or not self.seed_query_dir.exists():
            return []
        seeds = []
        for path in sorted(self.seed_query_dir.glob("Query *")):
            seeds.append(SeedQuery(question=None, sparql=path.read_text(encoding="utf-8"), name=path.name))
        return seeds

    def detect_temporal_properties(self) -> list[str]:
        temporal = set()
        for p in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            ranges = {str(r) for r in self.graph.objects(p, RDFS.range)}
            if ranges & _TEMPORAL_XSD_RANGES:
                temporal.add(str(p))
        return sorted(temporal)

    # --- internals ---

    def _candidate_name_predicates(self) -> set[URIRef]:
        candidates = set(_FIXED_NAME_PREDICATES)
        for p in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            local = _local_name(str(p)).lower()
            if "name" in local or "title" in local or "label" in local:
                candidates.add(p)
        return candidates

    def _name_predicates(self) -> set[URIRef]:
        return self._candidate_name_predicates()

    def _named_subjects(self) -> set[str]:
        if self._named_subjects_cache is None:
            named = set()
            for pred in self._candidate_name_predicates():
                for s, o in self.graph.subject_objects(pred):
                    if isinstance(o, Literal):
                        named.add(str(s))
            self._named_subjects_cache = named
        return self._named_subjects_cache

    def _class_index(self) -> dict[str, set[str]]:
        if self._entities_by_class is None:
            nameable = self.get_nameable_classes()
            index: dict[str, set[str]] = {c: set() for c in nameable}
            nameable_set = set(nameable)
            for subj, obj in self.graph.subject_objects(RDF.type):
                obj_str = str(obj)
                if obj_str in nameable_set:
                    index[obj_str].add(str(subj))
            self._entities_by_class = index
        return self._entities_by_class

    @staticmethod
    def _first(objects) -> str | None:
        for o in objects:
            return str(o)
        return None
