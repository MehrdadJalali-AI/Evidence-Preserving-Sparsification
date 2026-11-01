import pytest
from eps_audit.select import eps, InfeasibleBudget


def test_infeasible_budget_returns_certificate():
    q = {"cands": ["a", "b", "c"]}
    with pytest.raises(InfeasibleBudget) as e:
        eps(None, q, 2, units=[])
    c = e.value.certificate
    assert c["status"] == "infeasible" and c["k"] == 2 and c["min_feasible_k"] == 3


def test_feasible_anchor_only():
    q = {"cands": ["a", "b"]}
    assert eps(None, q, 2, units=[]) == {"M:a", "M:b"}
