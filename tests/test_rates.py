"""R15: a rate over an empty set must be reported as undefined, not as zero.

The H1 extraction run that motivated this shipped per-100 defect rates of 0.0
for every defect class because every record had failed and the relation list was
empty - `100 * n / max(len(relations), 1)` returns a plausible number from no
data. These tests pin the two behaviours that stop that recurring: the helper
returns None on an empty denominator, and the quality-metric block that feeds
the deliverables does too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wbs.rates import rate


def test_rate_normal():
    assert rate(3, 95) == 0.0316
    assert rate(3, [0] * 95) == 0.0316
    assert rate(1, [0, 0]) == 0.5
    assert rate(0, 10) == 0.0          # a measured zero is still zero


def test_rate_undefined_on_empty():
    assert rate(3, 0) is None
    assert rate(3, []) is None
    assert rate(0, []) is None         # the dangerous case: 0/0 is not 0.0


def test_rate_precision_is_settable():
    assert rate(1, 3, 8) == 0.33333333


def test_quality_metrics_report_null_on_an_empty_ontology():
    from wbs.task2_formalization.validate_ontology import quality_metrics

    q = quality_metrics([], [], [])
    for k in ("relationship_richness", "attribute_richness", "inheritance_richness",
              "graph_density", "average_degree", "isolated_fraction",
              "mean_hierarchy_depth", "provenance_coverage", "alias_coverage"):
        assert q[k] is None, f"{k} is {q[k]!r} for an empty ontology, not None"
    assert q["concepts"] == 0 and q["relations"] == 0


def test_quality_metrics_still_compute_on_real_input():
    from wbs.task2_formalization.validate_ontology import quality_metrics

    concepts = [{"concept_name": "A", "concept_class": "Attraction",
                 "provenance": {"confidence": 0.9}, "aliases": ["a"]},
                {"concept_name": "B", "concept_class": "Region"}]
    rels = [{"subject": "A", "predicate": "located_in", "object": "B"}]
    q = quality_metrics(concepts, rels, [])
    assert q["concepts"] == 2 and q["relations"] == 1
    assert q["provenance_coverage"] == 0.5
    assert q["average_degree"] == 1.0


# ── proposal 10.3: the two constraint families added 2026-10-18 ──────────────
# Both were implemented to close an audit gap, so the tests pin the cases that
# made the first implementation wrong: inverted polarity on `avoid_during`,
# alternative facts treated as contradictions, and adjectives read as classes.

def _kb():
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    return KnowledgeBase()


def _types(text, kb):
    from wbs.task3_reasoning.constrained_reasoner import validate
    return [v["type"] for v in validate(text, kb)]


def test_temporal_conflict_fires_outside_asserted_months():
    kb = _kb()
    assert "TEMPORAL_CONFLICT" in _types("Cherry blossoms bloom in Jeju in July.", kb)
    assert "TEMPORAL_CONFLICT" not in _types(
        "Cherry blossoms bloom in Jeju in early April.", kb)


def test_temporal_polarity_is_respected():
    """`avoid_during` lists months to stay away from, not months the fact holds."""
    kb = _kb()
    # recommending the subject inside its avoid window is the violation
    assert "TEMPORAL_CONFLICT" in _types(
        "Hiking on Hallasan Mountain is best in January.", kb)
    # merely mentioning it outside that window is not
    assert "TEMPORAL_CONFLICT" not in _types(
        "Hallasan Mountain trails may be closed in May due to flooding.", kb)
    # and warning about crowds in peak season is not a recommendation
    assert "TEMPORAL_CONFLICT" not in _types(
        "Jeju Island is busy and accommodations fill up during peak season "
        "(June-August).", kb)


def test_cardinality_needs_two_different_head_nouns():
    kb = _kb()
    assert "CARDINALITY_VIOLATION" in _types(
        "Seongsan Ilchulbong is a beach. Seongsan Ilchulbong is a temple.", kb)
    # same class, different adjectives, is not two classes
    assert "CARDINALITY_VIOLATION" not in _types(
        "Jeju Island is a popular destination. Jeju Island is a beautiful "
        "destination.", kb)


def test_functional_predicates_are_derived_from_the_ontology():
    kb = _kb()
    assert "is_a" in kb.functional and kb.functional["is_a"] >= 0.95
    # located_in is multi-valued in this ontology and must not be called functional
    assert "located_in" not in kb.functional


# ── R2: the span detector the repair loop reads ──────────────────────────────
# The validation layer cost 4.9 pp of token F1 for no measurable grounding gain
# because the detector put English capitalisation to it as fabrication. The fix
# is gated so every published number is produced with the old behaviour, and
# these tests pin both sides of the gate.

def test_raw_detector_is_unchanged_by_the_gate():
    """The default path must be byte-identical to the pre-fix behaviour."""
    from wbs.task3_reasoning.constrained_reasoner import ungrounded_entities
    kb = _kb()
    txt = ("Visit Yongduam Rock in the morning. Additional Tips: wear layers.\n"
           "Then try Jeju's black pork at Nonexistent Pork Palace.")
    assert (ungrounded_entities(txt, kb)
            == [s for s in kb.spans(txt) if not kb.known(s)])


def test_audit_grade_detector_drops_scaffolding():
    from wbs.task3_reasoning.constrained_reasoner import ungrounded_entities
    kb = _kb()
    txt = ("Additional Tips: consider Jeju's weather.\n"
           "Stay at the Zarnathex Grotto Resort.")
    raw = ungrounded_entities(txt, kb)
    aud = ungrounded_entities(txt, kb, audit_grade=True)
    low = {s.lower() for s in aud}
    # a section label and a possessive of a known entity are not claims
    assert "additional tips" not in low
    assert not any(s.lower().startswith("jeju's") for s in aud)
    assert len(aud) < len(raw)


def test_containment_exemption_is_generous_and_strict_mode_bounds_it():
    """The published rule exempts an invention that ends in a known generic word.

    This is not a wish about how it should behave: `Resort` is a concept name in
    this ontology, so the containment rule reads "Zarnathex Grotto Resort" as a
    phrasing of it. `strict=True` is the other end of the band - it counts the
    invention, and also counts section labels, which is why it is a sensitivity
    knob rather than a replacement.
    """
    from wbs.task4_evaluation.hallucination_report import audit_spans
    kb = _kb()
    txt = "Stay at the Zarnathex Grotto Resort."
    assert audit_spans(txt, kb)[1] == []
    assert any("zarnathex" in s.lower() for s in audit_spans(txt, kb, strict=True)[1])
    # and the knob defaults to the published rule everywhere
    import inspect
    assert inspect.signature(audit_spans).parameters["strict"].default is False


def test_audit_grade_detector_is_opt_in_everywhere():
    """`validate` and `constrained_answer` must default to the published path."""
    import inspect

    from wbs.task3_reasoning import constrained_reasoner as cr
    for fn in (cr.validate, cr.constrained_answer, cr.ungrounded_entities):
        assert inspect.signature(fn).parameters["audit_grade"].default is False


# ── R8: the 7-gram index replacing the validator's substring fallback ────────
# The fallback was a scan over every surface form, which made the validation
# pass the one query-time stage whose cost grew with ontology size. The index
# must be an index and not an approximation, so the test is equivalence with
# the scan it replaces rather than a speed assertion.

def _scan_known(kb, span):
    """The implementation the 7-gram index replaced, verbatim."""
    from wbs.task3_reasoning.constrained_reasoner import key
    k = key(span)
    if k in kb.surface:
        return True
    return any(k == s or (len(k) > 6 and k in s) for s in kb.surface)


def test_gram_index_agrees_with_the_scan_it_replaced():
    kb = _kb()
    probes = ["Hallasan", "Hallasan Mountain", "Seongsan Ilchulbong",
              "Zarnathex", "Zarnathex Grotto Resort", "Jeju", "Jejudo",
              "Seogwipo City Hall", "Late Morning", "abc", "abcdefg",
              "Cheonjiyeon", "Cheonjiyeon Waterfall Trail", "Manjanggul Cave"]
    # plus every surface form, which is the set the index is built from
    probes += list(kb.surface)[:400]
    for sp in probes:
        assert kb.known(sp) == _scan_known(kb, sp), sp


def test_gram_index_is_built_once_and_covers_the_surface_set():
    kb = _kb()
    idx = kb._gram_index()
    assert kb._gram_index() is idx          # cached, not rebuilt per call
    long_forms = [s for s in kb.surface if len(s) >= kb.GRAM]
    assert long_forms, "no surface form long enough to index"
    for s in long_forms[:200]:
        assert s[:kb.GRAM] in idx


# ── the selective validation gate ───────────────────────────────────────────
# `selective_validation.md` measured a policy that dominates validating every
# answer. The gate that implements it has to reproduce that policy exactly, and
# must leave the published behaviour alone at its default.

def test_selective_gate_defaults_to_validating_everything():
    import inspect

    from wbs.task3_reasoning.constrained_reasoner import constrained_answer
    assert inspect.signature(constrained_answer).parameters[
        "min_violations"].default == 0


def test_selective_gate_returns_the_draft_untouched_below_threshold():
    """A draft with nothing detectably ungrounded must come back unchanged."""
    from wbs.task3_reasoning.constrained_reasoner import constrained_answer
    kb = _kb()
    clean = "Hallasan Mountain is on Jeju Island."

    def regen(_):
        raise AssertionError("the gate must not spend a generation pass")

    r = constrained_answer("Where is Hallasan?", clean, kb, regenerate=regen,
                           min_violations=1)
    assert r["answer"] == clean
    assert r["selective_skip"] is True and r["repair_rounds"] == 0
    assert r["uncertainty_fallback"] is False
    # and the same draft with the gate off is free to be rewritten or replaced
    off = constrained_answer("Where is Hallasan?", clean, kb, min_violations=0)
    assert off["selective_skip"] is False


def test_selective_gate_lets_a_flagged_draft_through_to_the_layer():
    from wbs.task3_reasoning.constrained_reasoner import (constrained_answer,
                                                          ungrounded_entities)
    kb = _kb()
    bad = ("Visit the Zarnathex Grotto Resort and the Qemillon Botanical "
           "Gardens, then dine at Vrelmoor Pavilion.")
    assert ungrounded_entities(bad, kb, audit_grade=True), "test text not flagged"
    r = constrained_answer("What should I see?", bad, kb, min_violations=1)
    assert r["selective_skip"] is False


def test_shipped_gate_decides_exactly_as_the_offline_analysis_did():
    """The gate must select the same answers the measurement selected.

    `selective_validation.py` built its arm from `audit_spans(draft, kb)[1]`,
    and `constrained_answer(min_violations=...)` gates on
    `ungrounded_entities(draft, kb, audit_grade=True)`. If those two ever
    diverge, the policy that ships is not the policy that was measured, and
    nothing else in the suite would notice.
    """
    from wbs.task3_reasoning.constrained_reasoner import ungrounded_entities
    from wbs.task4_evaluation.hallucination_report import audit_spans
    kb = _kb()
    probes = [
        "Hallasan Mountain is on Jeju Island.",
        "Additional Tips: wear layers. Visit the Zarnathex Grotto Resort.",
        "Day 2: Cultural Exploration\n\nTry Jeju's black pork for dinner.",
        "",
        "Seongsan Ilchulbong offers sunrise views; Manjanggul Cave is nearby.",
    ]
    for txt in probes:
        assert (ungrounded_entities(txt, kb, audit_grade=True)
                == audit_spans(txt, kb)[1]), txt[:40]


def test_shipped_gate_reproduces_the_measured_selection_on_the_real_drafts():
    """End to end on the stored run the policy was fitted on, if present."""
    import json
    from pathlib import Path

    from wbs.task3_reasoning.constrained_reasoner import ungrounded_entities
    EVAL = Path(__file__).resolve().parents[1] / "wbs/deliverables/task4_evaluation"
    sel_p = EVAL / "selective_validation__llama3.1-8b.json"
    v4_p = EVAL / "variant__llama3.1-8b__V4_ontology_rag__hybded.jsonl"
    if not (sel_p.exists() and v4_p.exists()):
        import pytest
        pytest.skip("the 8B hybded run is not present in this checkout")
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    if sel["best_signal"] != "n_ungrounded_draft":
        import pytest
        pytest.skip("the winning signal is not the one the gate implements")
    ths = sel["cross_validated"]["n_ungrounded_draft"]["thresholds_by_fold"]
    th = max(set(ths), key=ths.count)
    kb = _kb()
    n = sum(1 for line in open(v4_p, encoding="utf-8") if line.strip()
            and len(ungrounded_entities(json.loads(line).get("answer") or "",
                                        kb, audit_grade=True)) >= th)
    assert n == sel["cross_validated"]["n_ungrounded_draft"]["n_validated"]


def test_selective_axis_labels_match_their_parameters():
    """The 2x2's labels say what each cell does; drift would misreport it.

    `min_coverage=0.0` does not mean "never abstain" - the unsupportable-claim
    trigger ignores min_coverage entirely - so the labels name the behaviour
    rather than the parameters, and nothing but this test ties the two
    together. A cell relabelled without its parameters being changed would be
    reported as the opposite configuration.
    """
    from wbs.task4_evaluation.ablate_generation import SELECTIVE_CONFIGS

    cells = {}
    for name, mr, mc, mv in SELECTIVE_CONFIGS:
        if mr is None:                      # the draft row runs no layer at all
            assert mc is None and mv == 0, name
            continue
        gated = name.startswith("gated")
        assert (mv >= 1) is gated, f"{name}: gate label disagrees with min_violations"
        floor = "coverage floor" in name
        assert (mc == 0.6) is floor, f"{name}: abstain label disagrees with min_coverage"
        assert floor or mc == 0.0, f"{name}: unsupportable-only cell must pass 0.0"
        cells[(gated, floor)] = name

    assert len(cells) == 4, f"the grid is not a full 2x2: {sorted(cells)}"
    assert len({c[0] for c in SELECTIVE_CONFIGS}) == len(SELECTIVE_CONFIGS), \
        "duplicate configuration labels would collide in the answers file"


def test_rule_attribute_availability_is_measured_not_asserted():
    """Each rule attribute's availability must come from the data.

    `rule_compliance.py` hardcoded four of these to False. One was wrong:
    `suitable_for` carries traveler type on the overwhelming majority of its
    triples, in the seed and in the version the validator loads, so the rule
    that needed it was reported as blocked on an attribute the ontology has.
    This pins the two directions that matter - an attribute carried by a
    predicate that is mostly about it counts as available, and a class with
    instances that nothing meaningfully points at does not.
    """
    import json
    from pathlib import Path

    f = (Path(__file__).resolve().parents[1]
         / "wbs/deliverables/task2_formalization/rule_compliance.json")
    if not f.exists():
        import pytest
        pytest.skip("rule_compliance.json has not been generated in this checkout")
    sup = json.loads(f.read_text(encoding="utf-8")).get("attribute_support")
    assert sup, "availability is not recorded, so it cannot be checked"

    share = sup["min_linked_share"]
    for scope in ("seed", "production"):
        tt = sup[scope].get("traveler type")
        if not tt:
            continue
        # carried by a predicate that is overwhelmingly about traveler types
        assert tt["linked_share"] >= share, (scope, tt)
        assert tt["available"] is True, f"{scope}: traveler type is carried, {tt}"

        wc = sup[scope]["weather state"]
        # the class may exist; what matters is whether anything points at it
        if wc["linked_share"] is not None and wc["linked_share"] < share:
            assert wc["available"] is False, f"{scope}: not queryable, {wc}"


def test_the_four_unenforceable_rules_are_the_named_ones():
    """The enforceable count is quoted in several documents; pin it here.

    A change to the availability logic that silently moved this count would
    leave every document that quotes "11 of 15" or "4 of 8" wrong.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables/task2_formalization"
    rc, carry = D / "rule_compliance.json", D / "rule_layer_carry.json"
    if not (rc.exists() and carry.exists()):
        import pytest
        pytest.skip("the rule deliverables are not present in this checkout")
    e = json.loads(rc.read_text(encoding="utf-8"))["enforceability"]
    blocked = sorted(k for k, v in e.items() if not v.get("enforceable"))
    assert blocked == ["R001", "R003", "R006", "R007"], blocked
    assert sum(1 for v in e.values() if v.get("enforceable")) == 4, "4 of 8 seed"

    c = json.loads(carry.read_text(encoding="utf-8"))
    assert c["n_enforceable"] == 11 and c["n_rules"] == 15, (c["n_enforceable"],
                                                             c["n_rules"])
    # and every blocked rule must say why, in terms a reader can check
    for rid in blocked:
        why = (c["per_rule"][rid].get("reason") or "")
        assert why and "needs" in why, (rid, why)


def test_ontology_only_verdict_follows_its_own_holm_rule():
    """The ontology-only arm's verdict must be the rule applied, not a label.

    `report_ontology_only_validated.py` decides whether the arm beats Simple
    RAG from Holm-corrected paired tests, and the answer it reached is
    negative. A later change that widened the family, flipped a sign
    convention or stopped correcting would move that verdict silently, and
    both the final report and R17 quote it. This recomputes the rule from the
    stored tests and requires the recorded verdict to match.
    """
    import json
    from pathlib import Path

    D = (Path(__file__).resolve().parents[1]
         / "wbs/deliverables/task4_evaluation")
    files = sorted(D.glob("ontology_only_validated__*.json"))
    if not files:
        import pytest
        pytest.skip("the ontology-only arm has not been run in this checkout")
    for f in files:
        _check_one_verdict(json.loads(f.read_text(encoding="utf-8")), f.name)


def _check_one_verdict(d: dict, name: str) -> None:
    vr = d["vs_rag"]
    # lower is better for exactly one of the two groundedness measures
    def wins(cell: str, metric: str) -> bool:
        x = vr[cell][metric]
        if x["holm_p"] is None or x["holm_p"] >= 0.05:
            return False
        return x["delta"] < 0 if metric == "ungrounded_rate" else x["delta"] > 0

    expect = sorted(c for c in vr
                    if wins(c, "ungrounded_rate") and wins(c, "clean"))
    assert sorted(d["beats_rag_on_both_groundedness"]) == expect, (
        name, d["beats_rag_on_both_groundedness"], expect)
    # Holm must actually be a correction: no adjusted p below its own raw p
    for cell, mets in vr.items():
        for m, x in mets.items():
            if x["p"] is not None and x["holm_p"] is not None:
                assert x["holm_p"] >= x["p"] - 1e-12, (name, cell, m, x)


def test_the_groundedness_credit_carries_its_qualification():
    """If retrieval dominates the graph-only context, §9 must say so.

    The report's conclusion 2 credits the ontology with the groundedness win.
    §5.1e measures whether the ontology supplies it alone; it does not. The
    qualification is generated from that run, so a regeneration that dropped
    it would leave the conclusion overstating what was measured.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables"
    f = D / "task4_evaluation/ontology_only_validated__llama3.1-8b.json"
    rep = D / "FINAL_REPORT.md"
    if not (f.exists() and rep.exists()):
        import pytest
        pytest.skip("the arm or the final report is not present in this checkout")
    d = json.loads(f.read_text(encoding="utf-8"))
    txt = rep.read_text(encoding="utf-8")
    if d.get("retrieved_prose_dominates"):
        assert "5.1e" in txt, "the run is not reported in the final report"
        assert "does not supply it alone" in txt, \
            "conclusion 2 credits the ontology without the measured qualification"
    # the verdict is size dependent, so neither half of it may be quoted bare
    sizes = d.get("beats_rag_at_sizes")
    if sizes:
        every = [q.stem.split("__")[1]
                 for q in (D / "task4_evaluation").glob("ontology_only_validated__*.json")]
        if [m for m in every if m not in sizes]:
            assert "depends on model size" in txt, \
                "the final report states one verdict for a size-dependent result"
            for m in sizes:
                assert m in txt, f"{m} wins and is not named in the report"



def test_selective_axis_cross_size_claims_match_the_grids():
    """The transfer claims must be the per-size grids, recomputed.

    §5.1d and R2 say the gate recovers token F1 at every size, that the
    interaction is negative at every size, and that the coverage floor can be
    left alone once the gate is on at some of them but not all. Each of those
    is a count over the per-size json, so each is checkable here rather than
    trusted.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables/task4_evaluation"
    files = sorted(D.glob("selective_axis__*.json"))
    if len(files) < 2:
        import pytest
        pytest.skip("the 2x2 has not been run at more than one size here")
    loaded = {f.stem.split("__")[1]: json.loads(f.read_text(encoding="utf-8"))
              for f in files}
    for model, d in loaded.items():
        cs = d.get("cross_size")
        if not cs:
            continue
        # every size this file claims something about must have its own run
        assert set(cs["sizes"]) <= set(loaded), (model, cs["sizes"])
        gate = [m for m, o in loaded.items()
                if o["marginal_token_f1"]["gate_only"] > 0]
        neg = [m for m, o in loaded.items()
               if o["marginal_token_f1"]["interaction"] < 0]
        ns = [m for m, o in loaded.items()
              if (o["tests"]["floor_given_gate_on"]["token_f1"]["p"] or 1) >= .05]
        assert sorted(cs["gate_recovers_token_f1_at"]) == sorted(gate), model
        assert sorted(cs["negative_interaction_at"]) == sorted(neg), model
        assert sorted(cs["floor_adds_nothing_once_gated_at"]) == sorted(ns), model


def test_routing_verdicts_are_the_measurements_not_a_story():
    """§5.1f and R7 quote three numbers that must come from the run.

    The per-item gold-span coverage, the stratum deltas and the oracle
    headroom each close off a direction the report would otherwise leave open,
    so each is pinned to the json rather than to prose that could drift from
    it. The oracle must also dominate every single arm by construction - if it
    does not, the selector is being computed wrong.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables/task4_evaluation"
    files = sorted(D.glob("coverage_routing__*.json"))
    if not files:
        import pytest
        pytest.skip("the routing analysis has not been run in this checkout")
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        g = d["gold_span_coverage"]
        assert 0 <= g["at_1.0"] <= g["n_scored"], (f.name, g)
        assert 0.0 <= g["median"] <= 1.0 and 0.0 <= g["mean"] <= 1.0, (f.name, g)
        o = d["oracle_selector"]
        # an oracle picking the best stored answer per item cannot be worse
        # than the best single arm it picks from
        best_arm = max(d["arms"][k]["token_f1"] for k in d["arms"])
        assert o["token_f1"] >= best_arm - 1e-9, (f.name, o["token_f1"], best_arm)
        assert o["headroom_token_f1_pp"] >= 0, (f.name, o)
        # the router is cross-validated, so it cannot be credited with the
        # oracle's headroom
        r = d["router"]["vs_rag"]["token_f1"]
        assert r["delta"] <= o["headroom_token_f1_pp"] + 1e-9, (f.name, r, o)


def test_the_candidate_ontology_does_not_displace_the_published_one():
    """v1.5.0 must be additive, and measured beside the published version.

    Validating or rule-checking a non-default version used to overwrite the
    published deliverables, because the output names carried no version. Both
    scripts now refuse. This pins the two halves a regression would break: the
    published record still says what it said, and the candidate's own record
    exists separately and says more rules fire.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables/task2_formalization"
    pub = D / "rule_compliance.json"
    cand = D / "rule_compliance__1.5.0.json"
    if not (pub.exists() and cand.exists()):
        import pytest
        pytest.skip("the candidate ontology has not been built in this checkout")
    pj = json.loads(pub.read_text(encoding="utf-8"))
    cj = json.loads(cand.read_text(encoding="utf-8"))
    n_pub = sum(1 for v in pj["enforceability"].values() if v["enforceable"])
    n_cand = sum(1 for v in cj["enforceability"].values() if v["enforceable"])
    # the published record is the one every reported number was measured on
    assert n_pub == 4, n_pub
    assert pj["attribute_support"]["production_version"] == "1.4.0"
    # and the candidate must be strictly better, or it is not worth carrying
    assert n_cand > n_pub, (n_pub, n_cand)
    assert cj["attribute_support"]["production_version"] == "1.5.0"

    ws = json.loads((D / "weather_state_extraction.json").read_text(encoding="utf-8"))
    # every accepted pair must clear the support threshold it claims
    assert all(t["support"] >= ws["min_support"] for t in ws["top"]), ws["min_support"]
    assert ws["accepted_pairs"] + ws["rejected_for_support"] == ws["candidate_pairs"]
    # and the unblocked set must be exactly the difference between the records
    was = {k for k, v in pj["enforceability"].items() if not v["enforceable"]}
    now = {k for k, v in cj["enforceability"].items() if not v["enforceable"]}
    assert sorted(was - now) == sorted(ws["unblocked"]), (was, now, ws["unblocked"])


def test_paraphrase_acceptance_is_accounted_for():
    """Every paraphrase must be accepted or counted as rejected, with a reason.

    The experiment's claim rests on accepted paraphrases differing from their
    originals in wording only. If a paraphrase that dropped an entity or
    changed length could slip into the scored set, an arm that names fewer
    places would be flattered by the filter rather than by its answers.
    """
    import json
    from pathlib import Path

    D = Path(__file__).resolve().parents[1] / "wbs/deliverables/task4_evaluation"
    files = sorted(D.glob("reference_paraphrase__*.json"))
    if not files:
        import pytest
        pytest.skip("the paraphrase experiment has not been run in this checkout")
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        assert (d["n_accepted"] + d["rejected_entities_lost"] + d["rejected_length"]
                == d["n_paraphrased"]), (f.name, d)
        # the scored set cannot exceed what survived the filter
        assert d["n_scored"] <= d["n_accepted"], (f.name, d)
        # and a drop is a drop: scoring against a reworded reference cannot
        # raise overlap for an arm that never saw the paraphrase
        for arm, r in d["arms"].items():
            assert r["token_f1_paraphrased"] <= r["token_f1_original"], (arm, r)


def test_the_deck_quotes_the_runs_it_names():
    """Nothing on a slide may be prose written over the numbers.

    The deck and the film both read presentation/demo_evidence.json. These
    checks pin the three things a reader would be misled by if the generator
    drifted: a quoted answer that is not the stored one, a headline comparison
    that does not hold in the scores it claims to come from, and the edit
    story's "no retraining" claim.
    """
    import json
    from pathlib import Path

    R = Path(__file__).resolve().parents[1]
    p = R / "wbs/deliverables/presentation/demo_evidence.json"
    if not p.exists():
        import pytest
        pytest.skip("the presentation has not been built in this checkout")
    ev = json.loads(p.read_text(encoding="utf-8"))
    T4 = R / "wbs/deliverables/task4_evaluation"

    # (a) the before/after answers are the stored ones, character for character
    for case in ev["cases"]:
        for arm, f in (("before", "V1_llm"),
                       ("after", "V5_ontology_rag_validated__hybded")):
            rows = {json.loads(l)["bench_id"]: json.loads(l) for l in
                    (T4 / f"variant__llama3.2-1b__{f}.jsonl").read_text(
                        encoding="utf-8").splitlines() if l.strip()}
            assert case[arm]["answer"] == rows[case["bench_id"]]["answer"], case["bench_id"]

    # (a2) the excerpt a slide quotes is a window of that same answer, not a
    #      paraphrase of it, and it contains the claim it is shown to prove
    for case in ev["cases"]:
        body = case["before_excerpt"].strip("… ").strip()
        full = " ".join(case["before"]["answer"].split())
        assert body in full, case["bench_id"]
        assert case["anchor"].lower() in body.lower(), case["bench_id"]

    # (b) the small-beats-big headline is a comparison of two stored runs
    sb, V = ev["small_beats_big"], ev["variants"]
    small, big = ev["sizes"][0], ev["sizes"][-1]
    assert sb["small_with_ontology"] == V[small]["V4_ontology_rag"]
    assert sb["big_without_ontology"] == V[big]["V1_llm"]
    assert sb["holds"] == (sb["small_with_ontology"]["token_f1"]
                           > sb["big_without_ontology"]["token_f1"]
                           and sb["small_with_ontology"]["ungrounded"]
                           < sb["big_without_ontology"]["ungrounded"])

    # (c) the deck may claim "no retraining" only if the digests say so
    live = ev.get("live_edit")
    if live:
        r = live["retraining"]
        assert r["weights_unchanged"] == (r["adapter_sha_before"] == r["adapter_sha_after"])
        assert r["weights_unchanged"], r
        assert r["gpu_hours_spent_on_the_edit"] == 0.0
        # and the claim counter is a count of pairs, not a verdict
        for v, tot in live["claim_check"]["totals"].items():
            assert 0 <= tot["asserted"] <= tot["claimed"], (v, tot)


def test_the_deck_keeps_the_result_that_goes_against_it():
    """The honest half is generated, not optional.

    A business deck is exactly where a losing number gets quietly dropped, so
    the evidence file must still carry the sizes where the ontology arm trails
    on reference overlap, and the deck source must still print them.
    """
    import json
    from pathlib import Path

    R = Path(__file__).resolve().parents[1]
    p = R / "wbs/deliverables/presentation/demo_evidence.json"
    if not p.exists():
        import pytest
        pytest.skip("the presentation has not been built in this checkout")
    ev = json.loads(p.read_text(encoding="utf-8"))
    trails = [m for m in ev["sizes"]
              if ev["gated"][m]["vs_rag"]["token_f1"]["delta"] < 0]
    assert ev["limits"]["token_f1_trails_at"] == trails
    assert trails, "if no size trails any more, the deck's honesty slide must be rewritten"
    src = (R / "wbs/task5_demo/make_slides.py").read_text(encoding="utf-8")
    assert "the claim we will not make" in src
    assert "token_f1" in src.split("the claim we will not make")[1][:1200]


# ── R19: surgical repair ─────────────────────────────────────────────────────

def _kb_and_items():
    import json
    import sys
    from pathlib import Path
    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    from wbs.task3_reasoning.constrained_reasoner import KnowledgeBase
    p = R / "wbs/deliverables/task4_evaluation/tourism_benchmark_categorized.jsonl"
    if not p.exists():
        return None, None, R
    items = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return KnowledgeBase(), items, R


def test_surgical_repair_never_adds_a_word_the_draft_did_not_have():
    """The layer's licence is to assert less, never something new.

    A repair that re-prompts the model can introduce a place name, a number or
    a hedge that nothing in the draft or the graph supports - which is how a
    validation layer becomes a second generator. The surgical pass is only
    allowed to delete, so every word in its output must already be in the
    draft. Punctuation is exempt: closing a hole leaves a full stop behind.
    """
    import re
    import json
    import pytest
    kb, items, R = _kb_and_items()
    if kb is None:
        pytest.skip("benchmark not present in this checkout")
    from wbs.task3_reasoning.constrained_reasoner import surgical_repair
    p = R / "wbs/deliverables/task4_evaluation/variant__llama3.1-8b__V4_ontology_rag__hybded.jsonl"
    if not p.exists():
        pytest.skip("V4 drafts not present in this checkout")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    word = re.compile(r"[A-Za-z0-9']+")
    touched = 0
    for r in rows:
        out, cut = surgical_repair(r["answer"], kb, audit_grade=True)
        if cut:
            touched += 1
        extra = set(word.findall(out.lower())) - set(word.findall(r["answer"].lower()))
        assert not extra, (r["bench_id"], sorted(extra)[:8])
    assert touched, "the pass fired on no draft at all - the detector or the drafts moved"


def test_surgical_repair_leaves_nothing_its_own_audit_counts():
    """Published 0.00% must be a property of the answers, not of the report.

    This is the constructional half of the claim and it has to hold exactly:
    if a single span survives, the rate the deliverables print is wrong.
    `test_the_strict_band_is_reported_beside_every_zero` is the other half.
    """
    import json
    import pytest
    kb, items, R = _kb_and_items()
    if kb is None:
        pytest.skip("benchmark not present in this checkout")
    from wbs.task3_reasoning.constrained_reasoner import surgical_repair
    from wbs.task4_evaluation.hallucination_report import audit_spans
    p = R / "wbs/deliverables/task4_evaluation/variant__llama3.1-8b__V4_ontology_rag__hybded.jsonl"
    if not p.exists():
        pytest.skip("V4 drafts not present in this checkout")
    for l in p.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        out, _ = surgical_repair(r["answer"], kb, audit_grade=True)
        assert not audit_spans(out, kb)[1], r["bench_id"]


def test_the_repair_policy_cells_match_the_answers_they_were_scored_on():
    """Every mean in the report is recomputable from the stored answers."""
    import json
    import statistics as st
    import pytest
    kb, items, R = _kb_and_items()
    if kb is None:
        pytest.skip("benchmark not present in this checkout")
    from wbs.task4_evaluation.score_variants import score_item
    EV = R / "wbs/deliverables/task4_evaluation"
    checked = 0
    for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b"):
        jp, ap = EV / f"repair_policy__{m}.json", EV / f"repair_policy__{m}__answers.jsonl"
        if not (jp.exists() and ap.exists()):
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        rows = [json.loads(l) for l in ap.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == d["n_items"]
        for cell, c in d["cells"].items():
            per = [score_item({"bench_id": r["bench_id"], "category": r["category"],
                               "reference": r["reference"],
                               "answer": r["answers"][cell]}, kb) for r in rows]
            for f, want in c["mean"].items():
                xs = [x[f] for x in per if x[f] is not None]
                assert len(xs) == c["n"][f], (m, cell, f)
                # the json stores means rounded to 4 decimal places
                assert abs(st.mean(xs) * 100 - want) < 1e-4, (m, cell, f, want)
            checked += 1
    if not checked:
        pytest.skip("repair_policy has not been run in this checkout")


def test_the_strict_band_is_reported_beside_every_zero():
    """A 0.00% the layer produced by construction may not stand on its own.

    The surgical cells remove exactly what the published detector counts, so
    their published rate is zero whatever the answers say. Both the report and
    the json must therefore carry the independent strict-band rate, and the
    report must say in words that the zero is constructional.
    """
    import json
    import pytest
    EV = __import__("pathlib").Path(__file__).resolve().parents[1] / "wbs/deliverables/task4_evaluation"
    md = EV / "repair_policy.md"
    if not md.exists():
        pytest.skip("repair_policy has not been run in this checkout")
    text = md.read_text(encoding="utf-8")
    assert "by construction" in text
    assert "strict" in text.lower()
    for m in ("llama3.2-1b", "llama3.2-3b", "llama3.1-8b"):
        jp = EV / f"repair_policy__{m}.json"
        if not jp.exists():
            continue
        d = json.loads(jp.read_text(encoding="utf-8"))
        for cell, c in d["cells"].items():
            if c["mean"].get("ungrounded_rate") == 0.0:
                assert c.get("ungrounded_rate_strict") is not None, (m, cell)
                assert c["ungrounded_rate_strict"] > 0, (m, cell)


def test_the_cross_model_verifier_table_is_a_transformation_of_stored_answers():
    """§5.1i must be recomputable, not a table someone typed.

    The claim is that the verifier is ordinary post-processing and therefore
    model-agnostic, so the check is that re-running it on the same stored
    generations reproduces the reported means - and that the strict-rule rate,
    which the layer does not optimise, really does fall for every model.
    """
    import json
    import statistics as st
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    jp = R / "wbs/deliverables/task4_evaluation/verifier_across_models__bench.json"
    if not jp.exists():
        pytest.skip("verifier_across_models has not been run in this checkout")
    d = json.loads(jp.read_text(encoding="utf-8"))
    from wbs.task3_reasoning.constrained_reasoner import (KnowledgeBase,
                                                          surgical_repair,
                                                          ungrounded_entities)
    from wbs.task4_evaluation.hallucination_report import audit_spans
    kb = KnowledgeBase()
    GEN = R / "experiments/exp024_rag_vs_ontology/outputs/generations_ded"
    for m, row in d["models"].items():
        p = GEN / f"bench__{m}__rag.jsonl"
        if not p.exists():
            continue
        rates = []
        for l in p.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            draft = json.loads(l).get("answer") or ""
            fixed = (surgical_repair(draft, kb, audit_grade=True)[0]
                     if ungrounded_entities(draft, kb, audit_grade=True) else draft)
            sp, bad, _ = audit_spans(fixed, kb, strict=True)
            if sp:
                rates.append(len(bad) / len(sp))
        assert abs(st.mean(rates) * 100 - row["after"]["ungrounded_strict"]) < 1e-3, m
        # and the point of the table: the rule the layer does not optimise moves
        assert row["after"]["ungrounded_strict"] < row["before"]["ungrounded_strict"], m


def test_every_policy_triple_is_in_the_version_it_claims_and_nowhere_before():
    """The demo's whole claim is "one fact was added, and nothing else changed".

    If a triple the deck shows as new at v1.6.0 were already in v1.5.0, the
    before/after would be theatre. This checks both directions: every accepted
    pair is asserted at the new version and absent from the one before it, and
    the new version is otherwise a superset of the old.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    rep = R / "wbs/deliverables/task2_formalization/policy_constraint_extraction.json"
    new = R / "artifacts/ontologies/1.6.0/relations.json"
    old = R / "artifacts/ontologies/1.5.0/relations.json"
    if not (rep.exists() and new.exists() and old.exists()):
        pytest.skip("the policy layer has not been extracted in this checkout")
    d = json.loads(rep.read_text(encoding="utf-8"))

    def triples(p):
        return {(str(r["subject"]), str(r["predicate"]), str(r["object"]))
                for r in json.loads(p.read_text(encoding="utf-8"))}

    tn, to = triples(new), triples(old)
    assert to <= tn, "the new version must be additive"
    assert len(tn - to) == d["accepted_pairs"], (len(tn - to), d["accepted_pairs"])
    for t in d["top"]:
        s, p_, o = t["triple"].split(f" {t['predicate']} ")[0], t["predicate"], \
            t["triple"].split(f" {t['predicate']} ")[1]
        assert (s, p_, o) in tn, t["triple"]
        assert (s, p_, o) not in to, f"{t['triple']} was already in v1.5.0"
    # and every one of them carries the evidence it was read from
    byt = {f"{r['subject']} {r['predicate']} {r['object']}": r
           for r in json.loads(new.read_text(encoding="utf-8"))}
    for t in d["top"]:
        prov = byt[t["triple"]]["provenance"]
        assert prov.get("evidence"), t["triple"]
        assert prov.get("source_record_ids"), t["triple"]
        assert len(prov["source_record_ids"]) >= 1


def test_the_policy_demo_slide_numbers_are_the_runs_own_counts():
    """A before/after slide is only worth the run behind it.

    The deck reads `policy_edit` out of demo_evidence.json; this recomputes the
    four numbers it puts on the counted slide straight from the raw run file,
    and checks that every case the deck tells is one where the rule really did
    reach the answer after the edit and not before.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    ev = R / "wbs/deliverables/presentation/demo_evidence.json"
    if not ev.exists():
        pytest.skip("demo evidence has not been built in this checkout")
    pe = json.loads(ev.read_text(encoding="utf-8")).get("policy_edit")
    if not pe:
        pytest.skip("the policy before/after demo has not been run")
    raw = sorted((R / "wbs/deliverables/presentation")
                 .glob("demo_policy_before_after__*.json"))
    d = next(json.loads(q.read_text(encoding="utf-8")) for q in raw
             if json.loads(q.read_text(encoding="utf-8"))["model"] == pe["model"])
    b, a = d["before"], d["after"]
    tgt = [i for i in d["items"] if i["kind"] != "control"]
    ctl = [i for i in d["items"] if i["kind"] == "control"]
    sm = pe["summary"]
    assert sm["rule_stated_before"] == sum(i[b]["states_the_rule"] for i in tgt)
    assert sm["rule_stated_after"] == sum(i[a]["states_the_rule"] for i in tgt)
    assert sm["control_answers_changed"] == sum(i["changed"] for i in ctl)
    assert sm["targeted"] == len(tgt) and sm["controls"] == len(ctl)
    for case in pe["cases"]:
        assert case["kind_of_change"] in ("corrected", "backed")
        if case["kind_of_change"] == "corrected":
            assert case["after"]["states_the_rule"]
            assert not case["before"]["states_the_rule"]
        assert case["after"]["rule_asserted_in_graph"]
        assert not case["before"]["rule_asserted_in_graph"]


def test_the_deck_does_not_claim_the_edit_was_local():
    """R21. The controls moved, and the deck has to say so.

    The temptation on a before/after slide is to report only the questions that
    moved the right way. This pins that the control count and the overlap are
    both carried into the evidence file the deck reads, and that the generator
    actually prints them.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    ev = R / "wbs/deliverables/presentation/demo_evidence.json"
    if not ev.exists():
        pytest.skip("demo evidence has not been built in this checkout")
    pe = json.loads(ev.read_text(encoding="utf-8")).get("policy_edit")
    if not pe:
        pytest.skip("the policy before/after demo has not been run")
    assert "control_overlap_mean" in pe
    assert pe["controls"] and all("overlap" in c for c in pe["controls"])
    src = (R / "wbs/task5_demo/make_slides.py").read_text(encoding="utf-8")
    assert "control_answers_changed" in src
    assert "control_overlap_mean" in src
    if pe["summary"]["control_answers_changed"]:
        assert "Controls moved a little" in src, (
            "controls moved in the run, so the deck must say they moved")


def test_the_film_labels_the_briefs_examples_as_illustrations():
    """The two generic examples in the film are the brief's, not our runs.

    They are the only answers in either artefact that no model produced, so the
    one thing that must never be edited away is the line that says so.
    """
    import sys
    from pathlib import Path

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    src = (R / "wbs/task5_demo/make_demo_video.py").read_text(encoding="utf-8")
    assert src.count("not a run of this system") >= 2, (
        "each of the brief's two examples must carry the disclaimer")
    # and the four parts the brief asks for are the scene's actual structure
    for part in ("THE PROBLEM", "THE ONTOLOGY FIX", "WHY IT IMPROVED"):
        assert part in src, part
    assert "scene_generic1" in src and "scene_generic2" in src
    assert "draw_compat" in src and "draw_boundary" in src


# ── proposal §8.7: role-based access control over knowledge changes ────────

def test_the_publishing_permissions_stay_with_the_administrator():
    """§8.7's point is that *some* changes must be refusable.

    An RBAC layer where every role can do everything is documentation, not
    control. This pins the four permissions that must never leave `admin` -
    writing a production version, moving the published pointer, rolling back,
    and overriding a protected element - plus least privilege for a principal
    the registry has never heard of.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wbs.task3_governance import access_control as ac

    for perm in ("proposal.merge.production", "pointer.set", "version.rollback",
                 "element.modify.protected"):
        assert ac.holders(perm) == {"admin"}, (perm, ac.holders(perm))
    # the pipeline may propose and stage, and that is all
    auto = ac.permissions("automation")
    assert {"proposal.create", "proposal.validate", "proposal.merge.staging"} <= auto
    assert not ({"proposal.approve", "proposal.merge.production", "pointer.set"} & auto)
    # an expert may approve but not publish
    exp = ac.permissions("expert")
    assert "proposal.approve" in exp and "pointer.set" not in exp
    # inheritance is real: admin holds everything
    assert ac.permissions("admin") == set(ac.PERMISSIONS)
    # least privilege for the unregistered
    assert ac.role_of("no-such-principal-in-the-registry") == "reader"
    assert ac.permissions("reader") == {"ontology.read"}
    assert not ac._self_check(), ac._self_check()


def test_protected_elements_are_derived_from_the_ontology_not_hand_listed():
    """The protected set has to follow the ontology, or it goes stale.

    Recomputes the registry from the published version and checks that every
    protected element is justified by its own provenance or its own predicate,
    and that an ordinary fact is not protected - otherwise "protected" would
    mean "everything" and the permission would never bite.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    from wbs.task3_governance import access_control as ac

    pub = ac._published_version()
    if not pub or not (ac.ONTO_DIR / pub / "relations.json").exists():
        pytest.skip("no published ontology version on disk")
    reg = ac.derive_protected(pub)
    rels = json.loads((ac.ONTO_DIR / pub / "relations.json").read_text(encoding="utf-8"))
    concepts = json.loads((ac.ONTO_DIR / pub / "concepts.json").read_text(encoding="utf-8"))

    assert reg["counts"]["total"] == len(reg["items"]) > 0
    for i in reg["items"]:
        assert i["rule"] in reg["rules"]
        if i["rule"] == "safety_constraint":
            assert i["key"].split("|")[1] in ac.SAFETY_PREDICATES, i["label"]

    # every safety triple in the version is in the registry, none missed
    want = {f"{ac._norm(r['subject'])}|{r['predicate']}|{ac._norm(r['object'])}"
            for r in rels if isinstance(r, dict) and r.get("subject")
            and r.get("object") and r["predicate"] in ac.SAFETY_PREDICATES}
    have = {i["key"] for i in reg["items"]}
    assert want <= have, sorted(want - have)[:5]

    # and an ordinary fact is not protected
    plain = next((r for r in rels if isinstance(r, dict) and r.get("object")
                  and r["predicate"] not in ac.SAFETY_PREDICATES
                  and not ac._human_approved(r.get("provenance"))), None)
    assert plain is not None
    assert (f"{ac._norm(plain['subject'])}|{plain['predicate']}|"
            f"{ac._norm(plain['object'])}") not in have
    assert reg["counts"]["total"] < len(rels) + len(concepts)

    # the human-approved rule is the one §8.4 asks for, and it found something
    assert reg["counts"]["by_rule"].get("human_approved", 0) >= 1


def test_every_governance_entry_point_asks_the_policy_before_it_writes():
    """A bypass is one missing line, so the lines are pinned.

    Each function in `version_governance.py` that can change knowledge, or
    change what the deployed system serves, must call into `access_control`
    before it touches a file.
    """
    import ast
    import sys
    from pathlib import Path

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    src = (R / "wbs/task3_governance/version_governance.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ("propose", "transition", "validate_proposal", "merge_proposal",
                 "set_current", "rollback"):
        assert name in fns, name
        body = ast.unparse(fns[name])
        assert "ac.require" in body, f"{name} writes without asking the policy"
    # and the two that are specifically about protection and about publishing
    assert "ac.guard_protected" in ast.unparse(fns["merge_proposal"])
    assert "ac.merge_permission" in ast.unparse(fns["merge_proposal"])
    assert "ac.require_distinct_approver" in ast.unparse(fns["transition"])


def test_the_governed_path_refuses_the_pipeline_and_admits_the_administrator():
    """The whole §8.7 path, run for real against a sandbox ontology.

    Not a check of the permission table - a check that the governance functions
    actually stop. The pipeline proposes, is refused approval twice over (it
    holds no `proposal.approve`, and it authored the proposal), the expert
    approves, the pipeline cannot write production, the admin can, and a
    protected triple cannot be overridden without the elevated permission.
    Every refusal has to land in the audit trail.
    """
    import importlib
    import json
    import os
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    from wbs.task3_governance import access_control as ac
    from wbs.task3_governance import version_governance as vg

    tmp = Path(os.environ.get("PYTEST_TMP") or "") if os.environ.get("PYTEST_TMP") else None
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="gov_rbac_"))
    onto, gov = tmpdir / "ontologies", tmpdir / "gov"
    (onto / "9.0.0").mkdir(parents=True)
    gov.mkdir(parents=True)
    # a three-fact ontology: one protected safety triple, one ordinary fact
    (onto / "9.0.0" / "concepts.json").write_text(json.dumps([
        {"name": "Testsite", "concept_class": "Attraction", "attributes": {}},
        {"name": "Helmet", "concept_class": "Equipment", "attributes": {}},
        {"name": "Nextdoor", "concept_class": "Attraction", "attributes": {}}]))
    (onto / "9.0.0" / "relations.json").write_text(json.dumps([
        {"subject": "Testsite", "predicate": "requires", "object": "Helmet"},
        {"subject": "Testsite", "predicate": "near", "object": "Nextdoor"}]))
    (onto / "9.0.0" / "hierarchy.json").write_text("[]")

    saved = {m: {k: getattr(m, k) for k in
                 ("ONTO_DIR", "GOV_DIR", "AUDIT")} for m in (ac, vg)}
    saved_extra = {"PRINCIPALS": ac.PRINCIPALS, "PROTECTED": ac.PROTECTED,
                   "MANUAL": ac.MANUAL,
                   "PROPOSALS": vg.PROPOSALS, "POINTER": vg.POINTER}
    prev_actor = os.environ.get("WBS_ACTOR")

    def as_(who):
        os.environ["WBS_ACTOR"] = who

    try:
        for m in (ac, vg):
            m.ONTO_DIR, m.GOV_DIR, m.AUDIT = onto, gov, gov / "audit_trail.jsonl"
        ac.PRINCIPALS, ac.PROTECTED = gov / "principals.json", gov / "protected.json"
        vg.PROPOSALS, vg.POINTER = gov / "proposals.json", gov / "current_version.json"
        ac.PRINCIPALS.write_text(json.dumps(ac.SEED_PRINCIPALS))

        prot = ac.derive_protected("9.0.0")
        assert prot["counts"]["total"] == 1                    # the `requires` triple
        assert prot["items"][0]["rule"] == "safety_constraint"

        # the pipeline proposes an ordinary addition
        as_("nlp-lab-pipeline")
        adds = [{"kind": "relation", "subject": "Nextdoor",
                 "predicate": "near", "object": "Testsite"}]
        p = vg.propose("add a reciprocal near edge", adds, "symmetry")
        v = vg.validate_proposal(p["id"], "9.0.0")
        vg.transition(p["id"], "validated", "ok")
        # ... and cannot approve its own work
        with pytest.raises(ac.AccessDenied):
            vg.transition(p["id"], "approved", "self-approval")
        # a reader cannot even propose
        as_("guest")
        with pytest.raises(ac.AccessDenied):
            vg.propose("nope", adds, "nope")
        # the expert can approve what it did not author
        as_("domain-expert-1")
        vg.transition(p["id"], "approved", "reviewed")
        # separation of duties only bites for a principal that *holds* approve:
        # the pipeline above was stopped one step earlier, by the permission
        # itself. So the expert authors one and is refused its own approval.
        q = vg.propose("expert's own addition",
                       [{"kind": "relation", "subject": "Helmet",
                         "predicate": "near", "object": "Testsite"}], "by hand")
        vg.validate_proposal(q["id"], "9.0.0")
        vg.transition(q["id"], "validated", "ok")
        with pytest.raises(ac.AccessDenied) as sod:
            vg.transition(q["id"], "approved", "self-approval")
        assert "separation of duties" in str(sod.value)
        assert vg.load_proposals()[-1]["state"] == "validated"
        # the pipeline may stage but not publish
        as_("nlp-lab-pipeline")
        with pytest.raises(ac.AccessDenied):
            vg.merge_proposal(p["id"], "9.0.0", "9.1.0")
        assert not (onto / "9.1.0").exists(), "a denied merge still wrote a version"
        # the admin may
        as_("ontology-admin")
        res = vg.merge_proposal(p["id"], "9.0.0", "9.1.0")
        assert res["relations"] == 3 and (onto / "9.1.0" / "relations.json").exists()

        # overriding the protected safety triple: expert refused, admin allowed
        override = [{"kind": "relation", "subject": "Testsite",
                     "predicate": "requires", "object": "Nothing At All"}]
        as_("domain-expert-1")
        with pytest.raises(ac.AccessDenied):
            ac.guard_protected(override, "9.0.0")
        as_("ontology-admin")
        assert len(ac.guard_protected(override, "9.0.0")) == 1

        # manual protection is a real operation, not a row in a table: an
        # expert may add one, only an admin may lift it, and the key a
        # protection is stored under has to be the key a lookup uses
        ac.MANUAL = gov / "protected_manual.json"
        as_("guest")
        with pytest.raises(ac.AccessDenied):
            ac.protect("relation", "Testsite|near|Nextdoor", "nope")
        as_("domain-expert-1")
        ac.protect("relation", "Testsite|near|Nextdoor", "reviewer judgement")
        key, label = ac.element_key("relation", "Testsite|near|Nextdoor")
        assert key in ac.load_manual() and label == "Testsite near Nextdoor"
        reg = ac.derive_protected("9.0.0")
        assert reg["counts"]["total"] == 2                 # derived one + manual one
        assert key in {i["key"] for i in reg["items"]}
        # and it now bites: overriding it needs the admin permission
        over = [{"kind": "relation", "subject": "Testsite",
                 "predicate": "near", "object": "Elsewhere"}]
        with pytest.raises(ac.AccessDenied):
            ac.guard_protected(over, "9.0.0")
        with pytest.raises(ac.AccessDenied):               # expert cannot lift it
            ac.unprotect(key, "no")
        as_("ontology-admin")
        assert ac.unprotect(key, "done") is True
        assert ac.load_manual() == {}
        assert ac.derive_protected("9.0.0")["counts"]["total"] == 1
        # a relation given as a sentence is refused rather than stored wrong
        with pytest.raises(ValueError):
            ac.element_key("relation", "Testsite near Nextdoor")

        # accountability: every refusal is on the record, not just the successes
        lines = [json.loads(x) for x in
                 open(gov / "audit_trail.jsonl", encoding="utf-8") if x.strip()]
        denies = [x for x in lines if x["op"] == "access.deny"]
        assert len(denies) >= 8, [x["permission"] for x in denies]
        assert {"element.protect", "element.unprotect",
                "element.modify.protected"} <= {x["permission"] for x in denies}
        assert {"nlp-lab-pipeline", "guest", "domain-expert-1"} <= {
            x["actor"] for x in denies}
        assert any(x.get("reason") == "separation_of_duties" for x in denies)
        assert [x for x in lines if x["op"] == "access.allow"]
    finally:
        for m, kv in saved.items():
            for k, val in kv.items():
                setattr(m, k, val)
        ac.PRINCIPALS, ac.PROTECTED = saved_extra["PRINCIPALS"], saved_extra["PROTECTED"]
        ac.MANUAL = saved_extra["MANUAL"]
        vg.PROPOSALS, vg.POINTER = saved_extra["PROPOSALS"], saved_extra["POINTER"]
        if prev_actor is None:
            os.environ.pop("WBS_ACTOR", None)
        else:
            os.environ["WBS_ACTOR"] = prev_actor
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_the_human_review_scorer_matches_krippendorffs_published_example():
    """§11.4.2 and §11.6b demand an agreement figure before any mean.

    The protocols said so for weeks and nothing computed one. This runs the
    scorer's own verification suite, which checks the alpha implementation
    against Krippendorff's published four-observer example (0.743), and pins
    the ordering rule: one rater sheet must yield no mean and no unblinding.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wbs.task4_evaluation.score_human_review import (
        krippendorff_alpha, cohen_kappa, self_test)

    r = self_test()
    failed = [c["name"] for c in r["checks"] if not c["passed"]]
    assert not failed, failed
    assert r["passed"] == r["total"] >= 9

    # and the two properties the protocols actually turn on
    A = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    B = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, None]
    C = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    D = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [[v for v in col if v is not None] for col in zip(A, B, C, D)]
    assert abs(krippendorff_alpha(units, "nominal") - 0.743) < 0.001
    assert krippendorff_alpha([[1], [2], [3]]) is None       # nothing pairable
    assert cohen_kappa([]) is None


def test_the_unfilled_instruments_report_no_score_rather_than_a_zero():
    """R15's rule, applied to human ratings.

    An unfilled sheet must come back as undefined, not as a mean of nothing.
    This checks the generated report says `unfilled` for both instruments and
    carries no fabricated number, for as long as no panel has filled them.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    p = R / "wbs/deliverables/task4_evaluation/human_review_scores.json"
    if not p.exists():
        pytest.skip("the human-review scorer has not been run in this checkout")
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("expert_11_4_2", "manual_11_6b"):
        r = d[k]
        assert r["items"] > 0, k
        if r["n_raters"] < 2:
            assert r["status"] in ("unfilled", "single-rater"), k
            assert r["agreement"] is None, k
            assert r.get("means") is None and r.get("by_variant") is None, k
        else:
            assert r["status"] == "scored" and r["agreement"] is not None, k


def test_the_filled_workbook_claims_only_what_the_audit_found():
    """The deliverable the funder reads is the workbook, so pin it.

    Every progress cell must be a number the on-disk audit supports, the
    evidence column must name a file that exists for each requirement counted,
    and the summary sheet's headline must be the audit's own count rather than
    a figure typed in beside it.
    """
    import json
    import sys
    from pathlib import Path
    import pytest

    R = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(R))
    openpyxl = pytest.importorskip("openpyxl")
    from wbs.task4_scalability.fill_wbs_workbook import (
        OUT, ROW_OF, PLAN_ROW, PROGRESS_COL, PATH_COL, SHEET, SUMMARY_SHEET,
        PLAN_SHEET)

    if not OUT.exists():
        pytest.skip("the filled workbook has not been generated in this checkout")
    audit = json.loads((R / "wbs/deliverables/WBS_WORKBOOK_AUDIT.json")
                       .read_text(encoding="utf-8"))
    prop = json.loads((R / "wbs/deliverables/PROPOSAL_COMPLIANCE.json")
                      .read_text(encoding="utf-8"))
    by_code = {i["code"]: i for i in audit["items"]}

    wb = openpyxl.load_workbook(OUT, data_only=True)
    assert SHEET in wb.sheetnames and SUMMARY_SHEET in wb.sheetnames
    ws = wb[SHEET]
    for row, code in ROW_OF.items():
        a = by_code[code]
        want = len(a["met"]) / max(len(a["met"]) + len(a["missing"]), 1)
        got = ws[f"{PROGRESS_COL}{row}"].value
        assert isinstance(got, (int, float)), (code, got)
        assert abs(got - round(want, 2)) < 1e-9, (code, got, want)
        ev = str(ws[f"{PATH_COL}{row}"].value or "")
        assert ev, code
        # a row at 1.00 may not name a MISSING artifact
        if got == 1:
            assert "MISSING" not in ev, (code, ev)
        for line in ev.splitlines():
            if ":" in line and "MISSING" not in line:
                q = line.split(":", 1)[1].split(" (+")[0].strip()
                assert (R / q).exists(), (code, q)

    # the proposal-version Gantt got the same treatment, on the same rule
    plan = wb[PLAN_SHEET]
    for row, (_name, _maps, paths) in PLAN_ROW.items():
        got = plan[f"K{row}"].value
        have = [q for q in paths if (R / q).exists()]
        assert isinstance(got, (int, float))
        assert abs(got - round(len(have) / len(paths), 2)) < 1e-9, (row, got)
        if got == 1:
            assert "MISSING" not in str(plan[f"M{row}"].value or "")

    # and the headline figures are the audits', not typed in
    flat = [str(c.value) for r in wb[SUMMARY_SHEET].iter_rows() for c in r
            if c.value is not None]
    assert f"{audit['fully_satisfied']} / {audit['total']} complete" in flat
    assert f"{prop['met']} / {prop['total']} met" in flat
    # the three requirements that need people are named, not folded into a total
    oos = [i["section"] for i in prop["items"] if i["status"] == "out-of-scope"]
    assert len(oos) == prop["out_of_scope"]
    for sec in oos:
        assert sec in flat, sec


# ── Task 4.1.2: the ontology-only arm across eight models ──────────────────
# cross_model_results.py reads another project's stored answers. Three things can
# go wrong and none of them would look wrong in the output: the training budget
# could be mislabelled, so an unequal comparison reads as equal; a "win" could be
# published on one test when the other contradicts it; or the pairing could drift,
# so two different question sets get compared as if they were one.

_XROOT = Path(__file__).resolve().parents[1]
CROSS = _XROOT / "wbs" / "deliverables" / "task4_evaluation" / "cross_model_results.json"


def _cross():
    import json

    assert CROSS.exists(), "run python -m wbs.task4_evaluation.cross_model_results"
    return json.loads(CROSS.read_text(encoding="utf-8"))


def test_the_training_budget_of_every_arm_is_read_off_its_own_receipt():
    """A pass count asserted in our registry and not confirmed by the companion
    project's own training receipt is the one error that would turn an unequal
    comparison into a headline. verify_budgets() has to agree for all twelve."""
    from wbs.task4_evaluation.cross_model_results import REGISTRY, verify_budgets

    budgets = verify_budgets()
    assert len(budgets) == len(REGISTRY) == 16
    for key, b in budgets.items():
        if b.get("status") == "pending":
            # exp060 is still training this arm; nothing is claimed for it yet
            assert key.endswith("m"), key
            continue
        assert b["receipt_agrees"], (key, b["receipts"])

    # and the tier label has to follow from the counts, not be chosen by hand
    for spec in REGISTRY:
        o, g = spec["onto_passes"], spec["rag_passes"]
        if spec["tier"] == "equal":
            assert o == g == 1, spec["key"]
        elif spec["tier"] == "extended":
            assert o > g == 1, spec["key"]
        elif spec["tier"] == "matched":
            assert o == g > 1, spec["key"]
        else:
            assert spec["tier"] == "untuned_rag" and g == 0, spec["key"]


def test_the_four_8b_pairs_really_have_no_trained_rag_arm():
    """The claim that excludes those four pairs from every conclusion is a claim
    about the filesystem. If an adapter has since appeared, the exclusion is wrong
    and the report has to be rewritten rather than quietly kept."""
    from wbs.task4_evaluation.cross_model_results import E58, REGISTRY

    untuned = [s for s in REGISTRY if s["tier"] == "untuned_rag"]
    assert len(untuned) == 4
    for spec in untuned:
        onto = E58 / "adapters" / f"{spec['model']}__onto_graph" / "adapter_config.json"
        rag = E58 / "adapters" / f"{spec['model']}__rag"
        assert onto.exists(), f"{spec['model']}: ontology adapter vanished"
        assert not rag.exists(), f"{spec['model']}: a RAG adapter now exists"


def test_a_win_needs_both_tests_to_agree():
    """Wilcoxon and the bootstrap disagree on real rows in both directions. The
    rule is that a verdict of 'better' requires Holm p < 0.05 AND an interval that
    excludes zero; anything else is indistinguishable. Checked against every cell
    rather than trusting the summary."""
    payload = _cross()
    checked = 0
    for pair in payload["pairs"]:
        for name, m in pair["metrics"].items():
            if not m["countable_n"] or m["delta"] is None:
                continue
            checked += 1
            p, ci = m.get("p_holm"), m.get("ci95")
            sig = (p is not None and p < 0.05) and (
                ci is None or (ci[0] > 0) == (ci[1] > 0))
            if m["verdict"] == "indistinguishable":
                assert not sig, (pair["key"], name, p, ci)
            else:
                assert sig, (pair["key"], name, p, ci, m["verdict"])
                better = (m["delta"] < 0) if m.get("lower_is_better") \
                    else (m["delta"] > 0)
                assert m["verdict"] == ("ontology better" if better
                                        else "simple RAG better"), (pair["key"], name)
    assert checked >= len(payload["pairs"]) * len(payload["metrics_remeasured"]) - 4


def test_every_pair_compares_one_question_set_to_itself():
    """Two arms are only paired if they answered the same questions. A silent
    drift here would compare different splits and report the difference as an
    effect."""
    payload = _cross()
    for pair in payload["pairs"]:
        assert pair["paired_n"] == 4607, (pair["key"], pair["paired_n"])
        assert pair["onto_only_ids"] == 0 and pair["rag_only_ids"] == 0, pair["key"]
        for arm, src in pair["sources"].items():
            assert src["rows"] == 4607, (pair["key"], arm, src["rows"])
            assert len(src["sha256"]) == 64
        # the two arms must be different files; one file scored twice would
        # produce a flawless zero delta everywhere
        assert (pair["sources"]["ontology"]["sha256"]
                != pair["sources"]["simple_rag"]["sha256"]), pair["key"]


def test_the_rollup_counts_match_the_cells_they_summarise():
    """The per-metric roll-up is what the report and GOAL_STATUS read. It must be
    a count of the table, not a second opinion about it."""
    payload = _cross()
    for tier, per_metric in payload["metric_rollup"].items():
        rows = [p for p in payload["pairs"] if p["tier"] == tier]
        for name, roll in per_metric.items():
            assert roll["models"] == len(rows), (tier, name)
            got = [r["model"] for r in rows
                   if r["metrics"][name]["verdict"] == "ontology better"]
            assert roll["ontology_better_at"] == got, (tier, name)
            assert roll["ontology_better"] == len(got)
            total = (roll["ontology_better"] + roll["indistinguishable"]
                     + roll["simple_rag_better"])
            assert total == len([r for r in rows
                                 if r["metrics"][name]["countable_n"]]), (tier, name)


def test_no_equal_budget_model_is_credited_with_a_clean_sweep():
    """The standing goal asks for the ontology arm to beat Simple RAG on every
    metric. It does not, at equal budget, and this pins that: if a future run
    changes it the test fails and the claim gets rewritten deliberately."""
    payload = _cross()
    eq = payload["summary"]["equal"]
    assert eq["pairs"] == 4
    assert eq["clean_sweeps"] == 0, eq["per_pair"]
    # every one of the four gives up token precision - the specific shape the
    # report states
    for pair in payload["pairs"]:
        if pair["tier"] != "equal":
            continue
        tp = pair["metrics"]["token_p"]
        assert tp["verdict"] == "simple RAG better", (pair["key"], tp)


def test_the_router_never_fires_on_the_held_out_split():
    """The hybrid arm is only a separate system variant where the router replaces
    answers. On the test split it replaces none, so counting hybrid and
    ontology-only as two results would be counting one twice."""
    payload = _cross()
    checks = [r for r in payload["router_firing"] if r.get("status") != "absent"]
    assert len(checks) >= 4
    for r in checks:
        if r["split"] == "test":
            assert r["fired"] == 0, r
        else:
            assert r["fired_rate"] > 0.9, r


def test_the_cited_companion_files_still_exist_and_still_hash_the_same():
    """The table is only re-derivable while the files it names are unchanged. A
    stale sha256 means the published numbers no longer describe what is on disk."""
    import hashlib

    import pytest

    payload = _cross()
    codex = Path(payload["companion_project"])
    if not codex.exists():
        pytest.skip("companion project not mounted")
    # Not every arm lives in the companion project any more: the matched-budget
    # baselines are trained here, so each source is resolved against the root it
    # declares rather than against one root assumed for all of them.
    roots = {"codex": codex, "jejufull": _XROOT}
    seen = {}
    for pair in payload["pairs"]:
        for src in pair["sources"].values():
            project = src.get("project", "codex")
            assert project in roots, (src["path"], project)
            seen[(project, src["path"])] = src["sha256"]
    assert len(seen) == payload["distinct_arms_rescored"]
    assert len(seen) >= 20
    for (project, rel), want in seen.items():
        p = roots[project] / rel
        assert p.exists(), rel
        h = hashlib.sha256()
        with p.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                h.update(chunk)
        assert h.hexdigest() == want, rel


def test_the_cross_model_claims_in_the_report_are_the_ones_measured():
    """GOAL_STATUS and the final report both state cross-model counts in prose.
    They are generated from the JSON, and this checks the generation rather than
    the wording: the numbers in the files have to be the numbers in the payload."""
    payload = _cross()
    eq, ex = payload["summary"]["equal"], payload["summary"]["extended"]
    deliv = _XROOT / "wbs" / "deliverables"
    goal = (deliv / "GOAL_STATUS.md").read_text(encoding="utf-8")
    report = (deliv / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert f"{eq['clean_sweeps']} of {eq['pairs']} models" in goal
    assert f"{ex['never_worse']} of {ex['pairs']} models are then worse on nothing" in goal
    assert "### 5.1k" in report
    assert f"{payload['distinct_arms_rescored']} distinct stored arms" in report
    assert (f"{ex['never_worse']} of {ex['pairs']} models are worse than Simple RAG "
            "on nothing measured here") in report
    # the exclusion of the untuned pairs has to survive into both documents
    assert "the RAG adapter directory does not exist" in goal
    assert "only one arm was trained" in report


def test_we_scored_the_same_files_the_published_comparison_scored():
    """Two of the companion project's comparison files record the sha256 of the
    generation each arm was scored from. If our hash differs we are reading a
    different run under the same filename, and nothing in the table reconciles
    with what that project published."""
    payload = _cross()
    checked = [p for p in payload["pairs"]
               if p["declared_sha_check"]["status"] == "checked"]
    assert len(checked) == 2, [p["key"] for p in checked]
    for pair in checked:
        c = pair["declared_sha_check"]
        assert c["all_agree"], (pair["key"], c["declared"], c["computed"])
        for arm in ("ontology", "simple_rag"):
            assert c["declared"][arm] == c["computed"][arm] == \
                pair["sources"][arm]["sha256"], (pair["key"], arm)


# ── exp060: the matched-budget RAG baseline ───────────────────────────────
# This is the first thing in the cross-model work that WRITES. Two mistakes would
# be quiet and expensive: writing into the companion project, whose adapters back
# every published equal-budget number, and continuing from the wrong adapter, which
# would produce a baseline that looks matched and is not.

def _exp060():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "exp060_train",
        _XROOT / "experiments" / "exp060_matched_budget_rag" / "train_rag_extended.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_exp060_refuses_to_write_into_the_companion_project():
    """Every output path is asserted to be inside this repository before any GPU
    work starts. A matched baseline is worthless if producing it damaged the
    evidence it is meant to be compared against."""
    import pytest

    mod = _exp060()
    for model, plan in mod.PLAN.items():
        dest = mod.dest_for(model)
        assert mod.ROOT in dest.parents, (model, dest)
        assert mod.CODEX not in dest.parents, (model, dest)
        # and the guard itself has to fire, not merely be documented
        with pytest.raises(SystemExit, match="companion project"):
            mod.guard_paths(mod.CODEX / "experiments" / "stolen", plan["source"])


def test_exp060_only_continues_the_published_one_pass_rag_adapter():
    """The source has to be the RAG arm at exactly one pass. Continuing from an
    ontology adapter, or from an already-extended one, silently changes what the
    tier means."""
    import json

    import pytest

    mod = _exp060()
    for model, plan in mod.PLAN.items():
        source = plan["source"]
        assert (source / "adapter_config.json").exists(), model
        receipt = json.loads((source / "train_summary.json").read_text(encoding="utf-8"))
        assert receipt["arm"] == "rag", (model, receipt.get("arm"))
        assert receipt["train_cfg"]["epochs"] == 1, (model, receipt["train_cfg"])
        # the guard rejects an ontology adapter in the source slot
        onto = source.parent / source.name.replace("__rag", "__onto_graph")
        if (onto / "train_summary.json").exists():
            with pytest.raises(SystemExit):
                mod.guard_paths(mod.dest_for(model), onto)


def test_exp060_trains_each_model_to_its_own_ontology_pass_count():
    """The whole point is parity. The plan's target has to equal the ontology
    pass count the registry records for that model's extended pair."""
    from wbs.task4_evaluation.cross_model_results import REGISTRY

    mod = _exp060()
    extended = {s["model"]: s for s in REGISTRY if s["tier"] == "extended"}
    matched = {s["model"]: s for s in REGISTRY if s["tier"] == "matched"}
    assert set(mod.PLAN) == set(extended) == set(matched)
    for model, plan in mod.PLAN.items():
        assert plan["onto_passes"] == extended[model]["onto_passes"], model
        assert matched[model]["rag_passes"] == plan["onto_passes"], model
        assert matched[model]["onto_passes"] == plan["onto_passes"], model
        # the matched pair must reuse the SAME ontology answers as the extended
        # pair, or it is answering a different question
        assert matched[model]["onto"] == extended[model]["onto"], model


def test_exp060_generation_settings_match_the_published_baseline():
    """A different prompt, context file or decoding setting would make the new
    baseline incomparable with every stored answer, and nothing downstream would
    notice. The generator checks itself against the companion project's manifest;
    this checks that the constants it checks with are the published ones."""
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "exp060_gen",
        _XROOT / "experiments" / "exp060_matched_budget_rag" / "generate_rag_extended.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)

    assert gen.MAX_NEW == 512 and gen.MAX_PROMPT == 1536 and gen.SPLIT == "test"
    checked = 0
    for model, path in gen.MANIFEST.items():
        if not path.exists():
            continue
        man = json.loads(path.read_text(encoding="utf-8"))
        assert man["decoding"] == "greedy", model
        assert man["max_new_tokens"] == gen.MAX_NEW, model
        assert man["max_prompt_tokens"] == gen.MAX_PROMPT, model
        checked += 1
    assert checked == 2, "both llama manifests should be on record"


# The matched tier's whole argument is that a `matched` row and its `extended`
# counterpart differ in exactly one thing: the baseline's training budget. The
# report says so in as many words ("the ontology arm is the same answers in both
# rows"), and every per-metric give-back it quotes is computed on that assumption.
# If the ontology arm ever drifted between the two rows - a re-generated file, a
# re-pointed registry entry, a different adapter - the give-backs would silently
# become differences of two variables and the claim would be unfounded.


def test_a_matched_pair_changes_only_the_baseline():
    payload = _cross()
    pairs = {p["key"]: p for p in payload["pairs"]}
    matched = [p for p in payload["pairs"] if p["tier"] == "matched"]
    assert matched, "no matched pair scored yet; nothing to check"
    for row in matched:
        base = pairs.get(row["key"][:-1])
        assert base is not None, f"{row['key']} has no extended counterpart"
        assert base["tier"] == "extended", base["key"]
        # same ontology generation file, byte for byte
        assert row["sources"]["ontology"]["sha256"] == \
            base["sources"]["ontology"]["sha256"], row["key"]
        # same ontology passes, and strictly more for the baseline
        assert row["onto_passes"] == base["onto_passes"], row["key"]
        assert row["rag_passes"] > base["rag_passes"], row["key"]
        # A metric scored on every item must therefore come out bit-identical. A
        # metric that is undefined where an arm produces nothing countable is scored
        # on the intersection, so a better baseline moves its denominator and the
        # ontology aggregate with it - allowed, but only for that reason and only by
        # a margin small enough not to carry a verdict.
        drift = {d["metric"]: d for d in payload["budget_effect"][row["key"]]["ontology_drift"]}
        for metric, now in row["metrics"].items():
            was = base["metrics"][metric]
            if not isinstance(now.get("ontology"), float):
                continue
            if now["countable_n"] == was["countable_n"]:
                assert abs(now["ontology"] - was["ontology"]) < 1e-12, \
                    (row["key"], metric, now["ontology"], was["ontology"])
                assert metric not in drift, (row["key"], metric)
            else:
                assert metric in drift, (row["key"], metric)
                assert abs(now["ontology"] - was["ontology"]) < 0.005, \
                    (row["key"], metric, now["ontology"], was["ontology"])


def test_the_matched_baseline_is_this_project_s_own_output():
    """The companion project is read-only evidence; the matched baselines are ours.
    Recording both under one root would publish our own training run as external
    corroboration, which is the one provenance error this table exists to avoid."""
    payload = _cross()
    for row in payload["pairs"]:
        onto = row["sources"]["ontology"]
        rag = row["sources"]["simple_rag"]
        assert onto["project"] == "codex", row["key"]
        if row["tier"] == "matched":
            assert rag["project"] == "jejufull", row["key"]
            assert rag["path"].startswith("experiments/exp060_matched_budget_rag/"), \
                rag["path"]
        else:
            assert rag["project"] == "codex", row["key"]


def test_a_win_that_becomes_a_tie_is_not_reported_as_a_loss():
    """`never_worse` counts losses, so a win decaying to a tie leaves it unchanged
    and only `clean_sweep` drops. Without budget_effect that shows up as "0 lead on
    every metric", which reads like a reversal rather than one metric reaching
    parity. The distinction is the reason the tier exists."""
    payload = _cross()
    effect = payload["budget_effect"]
    assert effect, "no matched pair scored yet"
    pairs = {p["key"]: p for p in payload["pairs"]}
    for key, e in effect.items():
        row = pairs[key]
        for lost in e["lost"]:
            # `lost` is a win that became a TIE; a win that became a defeat belongs in
            # `to_loss` and is narrated differently, because calling it a tie would
            # describe the worst movement in the table as the mildest kind.
            assert row["metrics"][lost["metric"]]["verdict"] == "indistinguishable", \
                (key, lost["metric"], row["metrics"][lost["metric"]]["verdict"])
        for fell in e["to_loss"]:
            assert row["metrics"][fell["metric"]]["verdict"] == "simple RAG better", \
                (key, fell["metric"])
            assert fell["metric"] in {d["metric"] for d in e["newly_lost"]}, \
                (key, fell["metric"])
        assert not ({d["metric"] for d in e["lost"]}
                    & {d["metric"] for d in e["to_loss"]}), key
        for kept in e["kept"]:
            assert row["metrics"][kept["metric"]]["verdict"] == "ontology better", \
                (key, kept["metric"])
        sign = {m: (-1 if lo else 1) for m, _, lo in
                __import__("wbs.task4_evaluation.cross_model_results",
                           fromlist=["METRICS"]).METRICS}
        for kept in e["kept"]:
            margin_was = sign[kept["metric"]] * kept["was"]
            margin_now = sign[kept["metric"]] * kept["now"]
            assert kept["narrowed"] == (margin_now < margin_was), kept
