"""Exact oracle for the budgeted unit-selection problem: maximise the number of complete audit
units subject to |S| <= k with all candidate anchors retained (a set-union knapsack instance),
solved as a 0-1 integer program with HiGHS (scipy.optimize.milp). Used to measure how far the
greedy EPS order is from the optimum at tight budgets."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def complete_units(units, S):
    return sum(u["nodes"] <= S for u in units)


def ilp_max_units(q, units, k):
    anchors = {f"M:{m}" for m in q["cands"]}
    nodes = sorted(set().union(*(u["nodes"] for u in units)) - anchors) if units else []
    nu, nn = len(units), len(nodes)
    if nu == 0:
        return 0
    ix = {n: i for i, n in enumerate(nodes)}
    rows = sum(len(u["nodes"] - anchors) for u in units) + 1
    A = lil_matrix((rows, nu + nn))
    lo, hi = [], []
    r = 0
    for j, u in enumerate(units):             # y_u <= x_n for every non-anchor node of u
        for n in u["nodes"] - anchors:
            A[r, j] = 1
            A[r, nu + ix[n]] = -1
            lo.append(-np.inf); hi.append(0); r += 1
    A[r, nu:] = 1                              # sum x_n <= k - |anchors|
    lo.append(-np.inf); hi.append(k - len(anchors))
    c = np.concatenate([-np.ones(nu), np.zeros(nn)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(nu + nn),
               bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(res.message)
    return int(round(-res.fun))
