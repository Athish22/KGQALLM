"""Generic KGAdapter: any Turtle/RDF/OWL file, with zero KG-specific
overrides -- pure `RDFLibAdapter` defaults (name-coverage heuristic for
`get_nameable_classes`, prefixes read
straight from the file's own `@prefix` declarations, generic temporal
property detection by xsd:date*/time range).

Use this for a KG nobody has written a curated adapter for yet. Expect
lower grounding precision than a curated adapter like
`ManufacturingKGAdapter` -- see the heuristic's caveats in
`RDFLibAdapter.get_nameable_classes`.
"""

from __future__ import annotations

from kgqa.adapters.rdflib_base import RDFLibAdapter


class GenericRDFAdapter(RDFLibAdapter):
    pass
