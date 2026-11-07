import itertools
import numpy as np
import pytest

from eps_audit.data import synthetic_table, functionals
from eps_audit.graph import build_graph, reproduce_predictions
from eps_audit.units import make_queries, neighbourhood, audit_units, justification, certificate
from eps_audit.select import budget, eps, ppr, random_nodes, degree, type_aware, conflict_free
from eps_audit.readers import permissive, conservative
from eps_audit.replay import replay


@pytest.fixture(scope="module")
def setup():
    df = synthetic_table(80, seed=3)
    funcs = functionals(df)
    g = build_graph(df, funcs, "hse06", "synthetic", "sha-test", seed=1, contra_ev=0.8)
    qs = make_queries(df["qmof_id"].tolist(), ["hle17", "hse06"], 25, 8, seed=2)
    return df, funcs, g, qs


def all_methods(g, q, N, k):
    u = audit_units(g, q)
    return {"EPS": eps(g, q, k, units=u), "CF": conflict_free(g, q, k, units=u), "PPR": ppr(g, q, N, k),
            "Random": random_nodes(g, q, N, k, 0), "Degree": degree(g, q, N, k),
            "TA": type_aware(g, q, N, k), "CA": type_aware(g, q, N, k, condition_aware=True)}


def test_exclusion_iff_no_dft(setup):
    df, funcs, g, _ = setup
    for i, m in enumerate(df["qmof_id"]):
        for f, col in funcs.items():
            has = not np.isnan(df.at[i, col])
            assert (f"C:{m}:{f}" in g.ntype) == has
            assert (f"X:{m}:{f}" in g.ntype) == (not has)
            if not has:
                x = g.attr[f"X:{m}:{f}"]
                assert x["outcome"] == "absent" and x["sha256"] == "sha-test" and x["condition"] == f


def test_exclusion_unit_never_uses_claims(setup):
    _, _, g, qs = setup
    for q in qs:
        for u in audit_units(g, q):
            if u["kind"] == "exclusion":
                assert not any(n.startswith("C:") for n in u["nodes"])
                assert not g.has_dft(u["cand"], q["cond"])


def test_contradiction_units_include_the_anchor_and_contradicted_claim(setup):
    _, _, g, qs = setup
    for q in qs:
        for u in audit_units(g, q):
            if u["kind"] == "contradiction":
                assert f"M:{u['cand']}" in u["nodes"]
                assert u["parent"] in u["nodes"]


def test_certificate_shared_nodes_do_not_make_a_unit_partial(setup):
    _, _, g, qs = setup
    for q in qs:
        units = audit_units(g, q)
        contra = next((u for u in units if u["kind"] == "contradiction"), None)
        if contra is None:
            continue
        support = next(u for u in units if u["kind"] == "support" and u["cand"] == contra["cand"])
        status = certificate(units, set(support["nodes"]))["units"]
        record = next(x for x in status if x["kind"] == "contradiction" and x["cand"] == contra["cand"])
        assert record["status"] == "omitted"
        return
    pytest.fail("synthetic graph did not contain a contradiction unit")


def test_units_inside_neighbourhood(setup):
    _, _, g, qs = setup
    for q in qs:
        assert justification(audit_units(g, q)) <= neighbourhood(g, q)


def test_budget_compliance_and_distinct_node_accounting(setup):
    _, _, g, qs = setup
    for q in qs:
        N = neighbourhood(g, q)
        for frac in (0.05, 0.1, 0.3, 0.6, 1.0):
            k = budget(q, N, frac)
            for name, S in all_methods(g, q, N, k).items():
                assert len(S) <= k, name
                assert S <= N, name
                assert len(set(S)) == len(S)


def test_shared_hubs_counted_once(setup):
    _, _, g, qs = setup
    q = next(q for q in qs if sum(u["kind"] == "support" for u in audit_units(g, q)) >= 2)
    units = audit_units(g, q)
    sup = [u for u in units if u["kind"] == "support"]
    naive = sum(len(u["nodes"]) for u in sup)
    union = len(frozenset().union(*(u["nodes"] for u in sup)))
    assert union < naive  # P:snapshot and D:extract:f shared
    S = eps(g, q, 10 ** 6, units=units)
    assert len(S) == len(justification(units) | {f"M:{m}" for m in q["cands"]})


def test_minimum_audit_sufficient_set(setup):
    """Prop. 1: S is audit-sufficient (w.r.t. the canonical units) iff J* <= S; removing any
    node of J* breaks sufficiency, so J* is the unique minimum."""
    _, _, g, qs = setup
    for q in qs[:10]:
        units = audit_units(g, q)
        J = justification(units)
        assert certificate(units, set(J))["audit_sufficient"]
        for n in J:
            assert not certificate(units, set(J) - {n})["audit_sufficient"]


def test_eps_returns_justification_when_budget_allows(setup):
    """Prop. 2(i)."""
    _, _, g, qs = setup
    for q in qs:
        units = audit_units(g, q)
        J = justification(units) | {f"M:{m}" for m in q["cands"]}
        S = eps(g, q, len(J), units=units)
        assert S == J and certificate(units, S)["audit_sufficient"]


def test_soundness_under_insufficient_budget(setup):
    """Prop. 2(ii): every retained comparable claim has its complete support unit, and the
    certificate's unmet list is exactly the requirements of non-complete units."""
    _, _, g, qs = setup
    for q in qs:
        units = audit_units(g, q)
        N = neighbourhood(g, q)
        for frac in (0.05, 0.1, 0.2):
            S = eps(g, q, budget(q, N, frac), units=units)
            for u in units:
                if u["kind"] == "support" and any(n.startswith("C:") and n in S for n in u["nodes"]):
                    assert u["nodes"] <= S
            cert = certificate(units, S)
            incomplete = {u["kind"] for u in units if not u["nodes"] <= S}
            expect = set(itertools.chain(*[{"support": ["R1", "R2", "R3", "R5"], "contradiction": ["R4"],
                                            "exclusion": ["R2x"]}[k] for k in incomplete]))
            assert set(cert["unmet"]) == expect


def test_replay_on_justification_and_failure_without_exclusion(setup):
    _, _, g, qs = setup
    for q in qs:
        units = audit_units(g, q)
        J = set(justification(units))
        ref = conservative(g, q, set(g.ntype))
        rp = replay(g, q, J, units)
        assert rp is not None and rp["result"] == ref["result"] and rp["contested"] == ref["contested"]
        exc = [u for u in units if u["kind"] == "exclusion"]
        if exc:
            x = next(n for n in exc[0]["nodes"] if n.startswith("X:"))
            assert replay(g, q, J - {x}, units) is None


def test_conservative_reader_abstains_without_condition(setup):
    _, _, g, qs = setup
    for q in qs:
        units = audit_units(g, q)
        sup = [u for u in units if u["kind"] == "support"]
        if not sup:
            continue
        J = set(justification(units))
        k = next(n for n in sup[0]["nodes"] if n.startswith("K:"))
        assert conservative(g, q, J - {k})["answered"] is False
        assert permissive(g, q, J - {k})["answered"] is True
        break


def test_cross_fitting_no_leakage(setup):
    _, _, g, _ = setup
    f0, f1 = g.meta["folds"]
    assert not set(f0) & set(f1)
    assert reproduce_predictions(g) < 1e-9


# ---------------------------------------------------------------- extensions (revision)
from eps_audit.select import exclusion_aware_closure, eps_cost_first
from eps_audit.readers import claim_filtering, qualified
from eps_audit.auditor import independent_audit
from eps_audit.oracle import ilp_max_units, complete_units


def test_exclusion_aware_closure_respects_budget_and_keeps_anchors(setup):
    _, _, g, qs = setup
    for q in qs:
        N = neighbourhood(g, q)
        for frac in (0.1, 0.3, 1.0):
            k = budget(q, N, frac)
            S = exclusion_aware_closure(g, q, N, k)
            assert len(S) <= k and {f"M:{m}" for m in q["cands"]} <= S and S <= N


def test_claim_filter_never_abstains_and_agrees_on_full_graph(setup):
    _, _, g, qs = setup
    for q in qs:
        full = set(g.ntype)
        assert claim_filtering(g, q, full) == conservative(g, q, full)
        assert claim_filtering(g, q, {f"M:{m}" for m in q["cands"]})["answered"]


def test_independent_auditor_accepts_eps_whenever_canonical_audit_passes(setup):
    _, _, g, qs = setup
    for q in qs:
        units, N = audit_units(g, q), neighbourhood(g, q)
        ref = conservative(g, q, set(g.ntype))
        for frac in (0.1, 0.2, 0.3, 1.0):
            S = eps(g, q, budget(q, N, frac), units=units)
            if certificate(units, S)["audit_sufficient"]:
                assert independent_audit(g, q, S, ref)
        assert independent_audit(g, q, justification(units) | {f"M:{m}" for m in q["cands"]}, ref)


def test_qualified_reader_is_safe_on_eps(setup):
    """With EPS, an unflagged qualified answer is always correct (nothing unaccounted)."""
    _, _, g, qs = setup
    for q in qs:
        units, N = audit_units(g, q), neighbourhood(g, q)
        ref = conservative(g, q, set(g.ntype))
        for frac in (0.1, 0.2, 0.3):
            S = eps(g, q, budget(q, N, frac), units=units)
            a = qualified(g, q, S)
            if a["unaccounted"] == 0 and certificate(units, S)["audit_sufficient"]:
                assert a["result"] == ref["result"]


def test_ilp_oracle_bounds_greedy(setup):
    _, _, g, qs = setup
    for q in qs[:12]:
        units, N = audit_units(g, q), neighbourhood(g, q)
        for frac in (0.1, 0.2, 1.0):
            k = budget(q, N, frac)
            opt = ilp_max_units(q, units, k)
            assert complete_units(units, eps(g, q, k, units=units)) <= opt <= len(units)
            if k >= len(justification(units) | {f"M:{m}" for m in q["cands"]}):
                assert opt == len(units)


def test_cost_first_variant_budget_and_atomic_units(setup):
    _, _, g, qs = setup
    for q in qs:
        units, N = audit_units(g, q), neighbourhood(g, q)
        for frac in (0.1, 0.2, 1.0):
            k = budget(q, N, frac)
            S = eps_cost_first(g, q, k, units=units)
            assert len(S) <= k and complete_units(units, S) <= ilp_max_units(q, units, k)
            if k >= len(justification(units) | {f"M:{m}" for m in q["cands"]}):
                assert certificate(units, S)["audit_sufficient"]
