# Discarded H1 run - 2026-10-17

This run reported **0 relations for every strategy**. `hypothesis_h1.py` read
`r.object` from `OntologyRelation`, whose field is named `obj` (only
`OntologyRelation.to_dict()` spells it `"object"`), so every record raised
`AttributeError` after its concepts had been appended but before its relations
were. Concept counts in these files are real; every relation-derived number -
L1/L2 findings per 100 relations, hierarchy depth, cycles, richness, density,
degree - is an artefact of a zero denominator and must not be quoted.

Kept because the broken numbers were visible in the session log and the record
should say why they are not in the report. The corrected run replaces them.
