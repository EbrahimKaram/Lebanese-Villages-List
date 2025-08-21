"""hungarian_assign.py

Small helper to compute a one-to-one optimal assignment between English rows and Arabic candidates
using scores you already computed.

API:
  assignments = optimal_assignment(eng_list, arabic_list, score_lookup, threshold=80)

- eng_list: list of identifiers (e.g., english row indices or unique keys)
- arabic_list: list of candidate arabic names (strings)
- score_lookup: dict mapping (eng_id, arabic_name) -> score (0..100)
- threshold: minimum acceptable score; matches below threshold are treated as disallowed

The module will try to use SciPy's linear_sum_assignment for an optimal solution. If SciPy isn't
available it will try the `munkres` (Munkres) implementation. If neither is available it falls
back to a safe greedy matching.

Returns a dict: eng_id -> (arabic_name, score) for accepted matches where score >= threshold.
"""

from typing import List, Dict, Tuple, Any

INF_COST = 10_000.0


def _use_scipy(cost_matrix):
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    except Exception:
        return None


def _use_munkres(cost_matrix):
    try:
        from munkres import Munkres
        m = Munkres()
        # munkres expects a list of lists
        indexes = m.compute([list(map(float, row)) for row in cost_matrix])
        row_ind = [r for r, c in indexes]
        col_ind = [c for r, c in indexes]
        return row_ind, col_ind
    except Exception:
        return None


def _greedy_assignment(cost_matrix):
    # cost_matrix is a square matrix (list of lists); we greedily pick the smallest cost
    n = len(cost_matrix)
    assigned_rows = set()
    assigned_cols = set()
    row_ind = []
    col_ind = []
    # flatten all entries and sort by cost
    entries = []
    for i in range(n):
        for j in range(n):
            entries.append((cost_matrix[i][j], i, j))
    entries.sort(key=lambda x: x[0])
    for cost, i, j in entries:
        if i in assigned_rows or j in assigned_cols:
            continue
        assigned_rows.add(i)
        assigned_cols.add(j)
        row_ind.append(i)
        col_ind.append(j)
        if len(assigned_rows) == n:
            break
    return row_ind, col_ind


def _square_cost_matrix(cost):
    # cost: 2D list (m x n). Return square matrix by padding with INF_COST
    m = len(cost)
    n = len(cost[0]) if m else 0
    size = max(m, n)
    sq = [[INF_COST for _ in range(size)] for _ in range(size)]
    for i in range(m):
        for j in range(n):
            sq[i][j] = cost[i][j]
    return sq


def _build_cost_matrix(eng_list: List[Any], arabic_list: List[str], score_lookup: Dict[Tuple[Any, str], float], threshold: float) -> List[List[float]]:
    # We build a cost matrix where higher scores become lower costs. Pairs below threshold get INF_COST
    m = len(eng_list)
    n = len(arabic_list)
    cost = [[INF_COST for _ in range(n)] for _ in range(m)]
    for i, e in enumerate(eng_list):
        for j, a in enumerate(arabic_list):
            s = score_lookup.get((e, a), 0.0)
            if s >= threshold:
                # convert score (0..100) to cost: we want to minimize cost, so use negative score
                cost[i][j] = -float(s)
            else:
                cost[i][j] = INF_COST
    return cost


def optimal_assignment(eng_list: List[Any], arabic_list: List[str], score_lookup: Dict[Tuple[Any, str], float], threshold: float = 80.0) -> Dict[Any, Tuple[str, float]]:
    """Return optimal one-to-one assignment eng -> (arabic, score) for matches with score >= threshold.

    If no feasible match exists for an English row (all candidates < threshold), that row is omitted.
    """
    if not eng_list or not arabic_list:
        return {}

    cost = _build_cost_matrix(eng_list, arabic_list, score_lookup, threshold)
    sq = _square_cost_matrix(cost)

    # try scipy
    res = None
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as _np
        arr = _np.array(sq)
        row_ind, col_ind = linear_sum_assignment(arr)
        res = (list(row_ind), list(col_ind))
    except Exception:
        # try munkres
        try:
            from munkres import Munkres
            m = Munkres()
            indexes = m.compute([list(map(float, row)) for row in sq])
            row_ind = [r for r, c in indexes]
            col_ind = [c for r, c in indexes]
            res = (row_ind, col_ind)
        except Exception:
            # fallback greedy
            res = _greedy_assignment(sq)

    row_ind, col_ind = res

    assignments: Dict[Any, Tuple[str, float]] = {}
    for r, c in zip(row_ind, col_ind):
        if r >= len(eng_list) or c >= len(arabic_list):
            continue
        # cost was -score
        assigned_cost = sq[r][c]
        if assigned_cost <= -threshold:
            assigned_score = -assigned_cost
            assignments[eng_list[r]] = (arabic_list[c], float(assigned_score))
    return assignments


if __name__ == '__main__':
    # tiny smoke test when invoked directly
    eng = ['e1', 'e2']
    arab = ['a1', 'a2']
    scores = {('e1','a1'): 90.0, ('e1','a2'): 80.0, ('e2','a1'): 85.0, ('e2','a2'): 70.0}
    print('Assignments:', optimal_assignment(eng, arab, scores, threshold=75))
