"""Reference KGAdapter implementation for the Industry 4.0 production-line
KG. Subclasses `RDFLibAdapter` (the generic rdflib
mechanics) and overrides only the pieces where curated, KG-specific
knowledge beats the generic heuristic: which classes are nameable, which
literal is the "real" name when more than one is attached, and which
temporal properties don't carry an explicit xsd:dateTime range in this
particular ontology. This is the *only* module that should hardcode
`smo:`/`d:`/`tm:` URIs -- everything else in `kgqa` talks to it through
the generic `KGAdapter` interface.
"""

from __future__ import annotations

from rdflib import Namespace

from kgqa.adapters.rdflib_base import RDFLibAdapter

SMO = Namespace("http://www.semanticweb.org/manufacturingproductionline#")
D = Namespace("http://www.semanticweb.org/manufacturingproductionline/data/")
TM = Namespace("http://www.w3.org/2006/time#")
SOSA = Namespace("http://www.w3.org/ns/sosa#")

# Adapter-specific knowledge: classes whose individuals are stable,
# addressable-by-name entities, as opposed to
# generated event/observation classes (ManufacturingProcess, MotorState,
# sosa:Observation, tm:Time, WorkStationState, Sit-*, ...) that are only
# ever reached by traversal from one of these, filtered by time. More
# precise than the generic name-coverage heuristic in `RDFLibAdapter`
# (e.g. it deliberately excludes classes that happen to have *a* labeled
# individual or two without being stable entities).
NAMEABLE_CLASSES = [
    SMO.ProcessingMachine,
    SMO.AssemblingMachine,
    SMO.Sensor,
    SMO.PressureSensor,
    SMO.TemperatureSensor,
    SMO.VibrationSensor,
    SMO.ViscositySensor,
    SMO.Motor,
    SMO.Actuator,
    SMO.Operator,
    SMO.MaintenanceTechnician,
    SMO.Manager,
    SMO.Staff,
    SMO.Location,
    SMO.Die,
    SMO.BallShappingDie,
    SMO.FormingMoldDie,
    SMO.HighFrequencyDie,
    SMO.LaserDie,
    SMO.Tool,
    SMO.Heater,
    SMO.Conveyor,
    SMO.Feeder,
    SMO.Squeegee,
    SMO.Bed,
    SMO.Bar,
    SMO.GlueSprayingNeedle,
    SMO.WorkStation,
    SMO.StorageHall,
    SMO.ManufacturingFacility,
    SMO.Cell,
    SMO.Line,
    SMO.Supplier,
    SMO.Customer,
]


class ManufacturingKGAdapter(RDFLibAdapter):
    def get_nameable_classes(self) -> list[str]:
        return [str(c) for c in NAMEABLE_CLASSES]

    def get_entity_labels(self, uri: str) -> list[str]:
        # smo:hasName is this ontology's real "display name" property;
        # prefer it over incidental name-like datatype properties the
        # generic heuristic would also pick up (e.g. hasMaterialName on
        # an unrelated individual sharing no relation to this one).
        from rdflib import RDFS, URIRef

        ref = URIRef(uri)
        labels = {str(o) for o in self.graph.objects(ref, SMO.hasName)}
        labels |= {str(o) for o in self.graph.objects(ref, RDFS.label)}
        return sorted(labels)

    def detect_temporal_properties(self) -> list[str]:
        temporal = set(super().detect_temporal_properties())
        # tm:hasStartTime / tm:hasFinishTime are declared without an
        # explicit rdfs:range in this ontology -- they're well-known by
        # name/namespace instead, so fold them in explicitly.
        temporal.add(str(TM.hasStartTime))
        temporal.add(str(TM.hasFinishTime))
        return sorted(temporal)
