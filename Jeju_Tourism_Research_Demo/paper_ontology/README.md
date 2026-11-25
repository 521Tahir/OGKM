# Jeju Tourism Ontology v1.3.0: paper figures, data and tables

This folder holds the complete ontology embedded in the demo application, saved for use in the paper.
Everything here is generated from `../ontology/original/*.json`. No number was entered by hand.

To regenerate it, run this from the project root:

```bash
python -m experiments.exp041_demo_traces.export_paper_ontology
```

## The ontology at a glance

| Metric | Value |
|---|---|
| Concepts (graph nodes) | 1,585 |
| Instances (excluding the 32 class nodes) | 1,553 |
| Ontology classes | 32 (one root: `TourismEntity`) |
| Subclass edges / maximum depth | 31 / 3 |
| Relation types used | 19 (25 declared in the schema) |
| Asserted relations | 8,324 |
| RDF triples in the Turtle export | 18,374 |

The full list is in `tables/ontology_summary.csv`.

## figures/

Each figure comes in three formats:
- `.pdf`: vector. Use this one in LaTeX (`\includegraphics{...pdf}`).
- `.svg`: vector, editable in Inkscape or Illustrator. The text stays as real text.
- `.png`: 600 dpi, for Word or slides.

| File | Shows |
|---|---|
| `fig1_class_hierarchy` | All 32 classes as a tree. Each class shows its direct instances / instances including subclasses. |
| `fig2_full_ontology_network` | **The complete ontology**: all 1,585 concepts and 8,324 relations, one cluster per class. Node size is degree. The centre ring holds the class nodes. |
| `fig3_relation_schema` | Declared domain → relation type → range for the 9 relation types the schema constrains (3,353 of the 8,324 triples). |
| `fig4_instances_per_class` | Instance count per class. |
| `fig5_relations_per_predicate` | Asserted triples per relation type. |

Colours mark 8 semantic groups of classes, plus grey for season, weather and other context.
The colours come from a colourblind-checked palette, and every figure also has text labels and a legend, so it still reads in greyscale print.

## data/

| File | Contents |
|---|---|
| `jeju_tourism_ontology_v1.3.0.ttl` | Full OWL ontology in Turtle: classes, `rdfs:subClassOf`, object properties with domain and range, every individual with its labels, aliases, coordinates, confidence and provenance, and every relation. |
| `jeju_tourism_ontology_v1.3.0.owl` | The same graph in RDF/XML. It opens in Protégé. |
| `jeju_tourism_ontology_v1.3.0.jsonld` | The same graph in JSON-LD. |
| `concepts.csv` | One row per concept: class, degree, aliases, lat/lon, confidence, source, validator status. |
| `relations.csv` | One row per relation, with its per-triple confidence and provenance. The RDF files keep only the triple itself. |
| `class_hierarchy.csv` | The 31 parent → child class edges. |
| `relation_schema.csv` | Declared domain and range of each relation type, with its triple count. |
| `original_json/` | Verbatim copies of the source files. |

The three RDF files reload with exactly the same 18,374 triples, so no format loses information.
The namespace is `http://jejunu.ac.kr/ontology/jeju-tourism#`, the same as the project's formal deliverable in `wbs/deliverables/task2_formalization/`.

## tables/

Each table is available as `.csv`, and as `.tex` ready to `\input{}`. The LaTeX tables need `\usepackage{booktabs}`.

- `ontology_summary`: the headline numbers above.
- `class_statistics`: every class with its superclass, depth, direct instances and instances including subclasses.
- `predicate_statistics`: every relation type with its triple count and declared domain and range.

## Notes for the paper text

- **Version:** the demo and this export use **v1.3.0**, with 1,585 concepts and 8,324 relations. The normalised Turtle file in `wbs/deliverables/task2_formalization/` is v1.3.0-norm, with 1,564 individuals and 7,906 relations. If you report numbers from both, name the version each one comes from.
- **Stale schema file:** `schema.json` still carries the v1.2.0 schema header and a 14-edge class tree. The 31-edge tree used here comes from the instance paths in `hierarchy.json`, which match the app exactly.
- **Relations without a declared domain and range:** 10 of the 19 relation types in use have none in the schema, including `connected_to`, `is_a` and `near`. That's why Fig. 3 covers only 3,353 of the 8,324 triples.
