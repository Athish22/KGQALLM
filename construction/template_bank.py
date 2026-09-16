"""Template bank: keyed by intent + hop pattern, seeded from the 5
provided example queries and generalized
into parametrized templates. Uses `string.Template`'s `$name` syntax
rather than `str.format`, since SPARQL's own `{`/`}` braces would collide
with `.format()`'s placeholder syntax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from string import Template

from kgqa.analysis.intent import Intent

PREFIX_HEADER = """PREFIX smo: <http://www.semanticweb.org/manufacturingproductionline#>
PREFIX d: <http://www.semanticweb.org/manufacturingproductionline/data/>
PREFIX tm: <http://www.w3.org/2006/time#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX sosa: <http://www.w3.org/ns/sosa#>
"""


@dataclass
class Template_:
    name: str
    intent: Intent
    required_params: list[str]
    sparql_body: str  # $-placeholders, no PREFIX header
    # At least one of these must appear in the question (case-insensitive
    # substring) for this template to be eligible. Without this, matching
    # was intent + "are the required slots filled?" only -- a question
    # that grounds a machine but asks about something templates 3/4 don't
    # cover would still match template 2 just because machine_uri was
    # available -- a wrong template match is exactly the kind of
    # over-confident wrong answer this is trying to prevent.
    trigger_keywords: list[str] = field(default_factory=list)

    def render(self, params: dict[str, str]) -> str:
        missing = [p for p in self.required_params if p not in params]
        if missing:
            raise KeyError(f"template {self.name!r} missing params: {missing}")
        return PREFIX_HEADER + Template(self.sparql_body).substitute(params)


# Seeded from Query 1..5 -- URIs are substituted as full IRIs (`<...>`)
# since grounded candidates from the retrieval strategy come back as full
# URIs, not prefixed names.
TEMPLATES: list[Template_] = [
    Template_(
        name="machines_by_name",  # Query 1
        intent=Intent.LIST_FILTER,
        required_params=[],
        trigger_keywords=["machine", "machines"],
        sparql_body="""SELECT ?Machine ?Name
WHERE {
  { ?Machine a smo:ProcessingMachine. } UNION { ?Machine a smo:AssemblingMachine. }
  ?Machine smo:hasName ?Name.
  FILTER(?Name != "")
}""",
    ),
    Template_(
        name="tool_of_machine",  # Query 2
        intent=Intent.LOOKUP,
        required_params=["machine_uri"],
        trigger_keywords=["tool", "tools"],
        sparql_body="""SELECT ?Machine ?tool
WHERE {
  ?machine a smo:ProcessingMachine ;
           smo:hasName ?Machine ;
           smo:hasTool ?tool .
  FILTER (?machine = <$machine_uri> && ?Machine != "")
}""",
    ),
    Template_(
        name="motor_state_in_timerange",  # Query 3
        intent=Intent.LIST_FILTER,
        required_params=["machine_uri", "start_time", "end_time"],
        trigger_keywords=["motor", "motors", "state", "status"],
        sparql_body="""SELECT DISTINCT ?Motor_Name ?Status ?Start_time
WHERE {
  <$machine_uri> smo:hasTool ?motor .
  ?motor smo:hasName ?Motor_Name ;
         smo:hasMotorState ?state .
  ?process tm:hasTime ?time .
  ?state smo:hasState ?Status .
  ?time tm:hasStartTime ?Start_time .
  FILTER (?Start_time >= "$start_time"^^xsd:dateTime && ?Start_time <= "$end_time"^^xsd:dateTime)
}""",
    ),
    Template_(
        name="observations_of_tool",  # Query 4
        intent=Intent.LIST_FILTER,
        required_params=["tool_uri"],
        trigger_keywords=["reading", "readings", "observation", "observations", "result", "results", "sensor", "pressure", "temperature"],
        sparql_body="""SELECT DISTINCT ?machine ?Start_time ?result
WHERE {
  ?machine smo:hasTool ?tool ;
           smo:performsProcess ?process .
  ?process tm:hasTime ?time .
  ?tool sosa:madeObservation ?observation .
  ?observation sosa:hasSimpleResult ?result .
  ?time tm:hasStartTime ?Start_time .
  FILTER (?tool = <$tool_uri>)
}""",
    ),
    Template_(
        name="process_and_tool_counts_in_timerange",  # Query 5
        intent=Intent.AGGREGATE,
        required_params=["start_time", "end_time"],
        trigger_keywords=["process", "processes", "tool", "tools"],
        sparql_body="""SELECT DISTINCT ?machine (COUNT(DISTINCT ?process) AS ?process_count) (COUNT(DISTINCT ?tool) AS ?tool_count)
WHERE {
  ?machine smo:performsProcess ?process ;
           smo:hasTool ?tool .
  ?process tm:hasTime ?time .
  ?time tm:hasStartTime ?start_time ;
         tm:hasFinishTime ?finish_time .
  FILTER (?start_time >= "$start_time"^^xsd:dateTime && ?finish_time <= "$end_time"^^xsd:dateTime)
}
GROUP BY ?machine
ORDER BY ?machine""",
    ),
]


class TemplateBank:
    def __init__(self, templates: list[Template_] | None = None):
        self.templates = templates if templates is not None else list(TEMPLATES)

    def find(self, intent: Intent, available_params: set[str], question: str = "") -> Template_ | None:
        """Most specific match: the template for this intent whose
        required params are all satisfied and whose trigger keywords
        actually appear in the question, preferring the one that uses the
        most of what's available (rather than the first/loosest fit).
        """
        question_l = question.lower()
        candidates = [
            t
            for t in self.templates
            if t.intent == intent
            and set(t.required_params) <= available_params
            and (not t.trigger_keywords or any(kw in question_l for kw in t.trigger_keywords))
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda t: len(t.required_params))
