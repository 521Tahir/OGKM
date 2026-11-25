"""
Export the tourism ontology to RDF/Turtle and JSON-LD formats using rdflib.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.logging_utils import get_logger
from ..utils.io import ensure_dir, save_json

logger = get_logger("ontology.exporters")

NAMESPACE = "http://tourism.ontology.org/"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"


def _safe_uri(text: str) -> str:
    """Convert a label to a URI-safe fragment."""
    return text.strip().replace(" ", "_").replace("/", "_").replace("'", "")


class OntologyExporter:
    """Export ontology data to Turtle (.ttl) and JSON-LD formats."""

    def __init__(self, namespace: str = NAMESPACE, output_dir: str | Path = "artifacts/ontologies"):
        self.namespace = namespace
        self.output_dir = Path(output_dir)
        ensure_dir(self.output_dir)

    def export_turtle(
        self,
        concepts: list[dict],
        relations: list[dict],
        rules: list[dict],
        version: str = "0.1.0",
        output_path: str | Path | None = None,
    ) -> str:
        """Generate Turtle (.ttl) representation. Return path to saved file."""
        try:
            from rdflib import Graph, URIRef, Literal, Namespace, RDF, RDFS, OWL, XSD

            ns = Namespace(self.namespace)
            g = Graph()
            g.bind("tourism", ns)
            g.bind("owl", OWL)
            g.bind("rdfs", RDFS)

            # Ontology header
            onto_uri = URIRef(self.namespace + "TourismOntology")
            g.add((onto_uri, RDF.type, OWL.Ontology))
            g.add((onto_uri, RDFS.label, Literal(f"Tourism Ontology v{version}")))

            # Add concepts as OWL Named Individuals
            for concept in concepts:
                name = concept.get("name", "")
                cclass = concept.get("concept_class", "Thing")
                uri = ns[_safe_uri(name)]
                class_uri = ns[_safe_uri(cclass)]
                g.add((uri, RDF.type, class_uri))
                g.add((uri, RDF.type, OWL.NamedIndividual))
                g.add((uri, RDFS.label, Literal(name)))
                conf = concept.get("provenance", {}).get("confidence", 1.0)
                g.add((uri, ns.confidence, Literal(conf, datatype=XSD.float)))
                for alias in concept.get("aliases", []):
                    g.add((uri, ns.alias, Literal(alias)))

            # Add relations
            for rel in relations:
                subj_uri = ns[_safe_uri(rel.get("subject", ""))]
                obj_uri = ns[_safe_uri(rel.get("object", ""))]
                pred_uri = ns[_safe_uri(rel.get("predicate", ""))]
                g.add((subj_uri, pred_uri, obj_uri))

            # Serialize
            out_path = Path(output_path) if output_path else self.output_dir / f"tourism_v{version}.ttl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            g.serialize(destination=str(out_path), format="turtle")
            logger.info(f"Turtle export saved → {out_path}")
            return str(out_path)

        except ImportError:
            logger.warning("rdflib not available — writing minimal Turtle manually.")
            return self._export_turtle_minimal(concepts, relations, version, output_path)

    def _export_turtle_minimal(
        self,
        concepts: list[dict],
        relations: list[dict],
        version: str,
        output_path: str | Path | None,
    ) -> str:
        """Fallback minimal Turtle serializer (no rdflib dependency)."""
        lines = [
            f"@prefix tourism: <{self.namespace}> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "",
            f"tourism:TourismOntology a owl:Ontology ;",
            f'    rdfs:label "Tourism Ontology v{version}" .',
            "",
        ]
        for c in concepts:
            uri = _safe_uri(c.get("name", ""))
            cclass = _safe_uri(c.get("concept_class", "Thing"))
            lines.append(f"tourism:{uri} a tourism:{cclass} ;")
            lines.append(f'    rdfs:label "{c.get("name","")}" .')
            lines.append("")
        for r in relations:
            s = _safe_uri(r.get("subject", ""))
            p = _safe_uri(r.get("predicate", ""))
            o = _safe_uri(r.get("object", ""))
            lines.append(f"tourism:{s} tourism:{p} tourism:{o} .")

        out_path = Path(output_path) if output_path else self.output_dir / f"tourism_v{version}.ttl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Minimal Turtle export saved → {out_path}")
        return str(out_path)

    def export_jsonld(
        self,
        concepts: list[dict],
        relations: list[dict],
        version: str = "0.1.0",
        output_path: str | Path | None = None,
    ) -> str:
        """Generate JSON-LD representation. Return path."""
        jsonld: dict = {
            "@context": {
                "tourism": self.namespace,
                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                "owl": "http://www.w3.org/2002/07/owl#",
                "label": "rdfs:label",
                "type": "@type",
            },
            "@graph": [],
        }

        for c in concepts:
            node: dict = {
                "@id": f"tourism:{_safe_uri(c.get('name', ''))}",
                "@type": f"tourism:{_safe_uri(c.get('concept_class', 'Thing'))}",
                "label": c.get("name", ""),
                "provenance": c.get("provenance", {}),
            }
            if c.get("aliases"):
                node["aliases"] = c["aliases"]
            jsonld["@graph"].append(node)

        for r in relations:
            edge: dict = {
                "@id": f"tourism:{_safe_uri(r.get('subject', ''))}",
                f"tourism:{_safe_uri(r.get('predicate', ''))}": {
                    "@id": f"tourism:{_safe_uri(r.get('object', ''))}"
                },
            }
            jsonld["@graph"].append(edge)

        out_path = Path(output_path) if output_path else self.output_dir / f"tourism_v{version}.jsonld"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(jsonld, out_path)
        logger.info(f"JSON-LD export saved → {out_path}")
        return str(out_path)
