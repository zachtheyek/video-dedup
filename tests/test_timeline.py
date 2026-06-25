"""Timeline-solve and clustering tests: offset recovery up to gauge, weighting,
false-edge residual localisation, connected components, canonical intervals."""
import numpy as np
import pytest

from vdedup.cluster import solve_timeline, Edge, connected_components, cycle_check


def edges_from_truth(gt, pairs, weight=1.0):
    # edge (i,j) asserts beta = b_i - b_j
    return [Edge(i, j, gt[i] - gt[j], weight) for i, j in pairs]


def test_offset_recovery_up_to_gauge():
    gt = {"A": 0.0, "B": 10.0, "C": 25.0, "D": -5.0}
    nodes = list(gt)
    pairs = [("A", "B"), ("B", "C"), ("A", "D"), ("C", "D")]
    edges = edges_from_truth(gt, pairs)
    extents = {n: (0.0, 30.0) for n in nodes}
    sol = solve_timeline(nodes, edges, extents, reference="A")
    # offsets recovered relative to reference A (gt[A]=0, so absolute)
    for n in nodes:
        assert sol.offsets[n] == pytest.approx(gt[n] - gt["A"], abs=1e-6)
    assert sol.max_abs_residual < 1e-6


def test_gauge_independent_relative_offsets():
    gt = {"A": 3.0, "B": 13.0, "C": 28.0}
    nodes = list(gt)
    edges = edges_from_truth(gt, [("A", "B"), ("B", "C"), ("A", "C")])
    extents = {n: (0.0, 30.0) for n in nodes}
    sol = solve_timeline(nodes, edges, extents, reference="B")
    # differences must match ground truth regardless of gauge
    assert (sol.offsets["A"] - sol.offsets["B"]) == pytest.approx(gt["A"] - gt["B"], abs=1e-6)
    assert (sol.offsets["C"] - sol.offsets["B"]) == pytest.approx(gt["C"] - gt["B"], abs=1e-6)


def test_canonical_intervals_from_extents():
    gt = {"full": 0.0, "clip": 40.0}
    nodes = list(gt)
    edges = edges_from_truth(gt, [("full", "clip")])
    extents = {"full": (0.0, 100.0), "clip": (0.0, 20.0)}  # clip is 20s starting at canonical 40
    sol = solve_timeline(nodes, edges, extents, reference="full")
    assert sol.intervals["full"] == pytest.approx((0.0, 100.0))
    assert sol.intervals["clip"] == pytest.approx((40.0, 60.0))
    assert sol.canonical_span == pytest.approx((0.0, 100.0))


def test_false_edge_has_largest_residual():
    gt = {"A": 0.0, "B": 10.0, "C": 25.0}
    nodes = list(gt)
    good = edges_from_truth(gt, [("A", "B"), ("B", "C"), ("A", "C")], weight=4.0)
    false_edge = Edge("A", "C", gt["A"] - gt["C"] + 12.0, weight=1.0)  # wrong by 12s
    edges = good + [false_edge]
    extents = {n: (0.0, 30.0) for n in nodes}
    sol = solve_timeline(nodes, edges, extents, reference="A")
    residuals = {k: abs(v) for k, v in sol.residuals.items()}
    # the false A-C edge is the last one; it should carry by far the most residual
    worst = max(residuals.values())
    assert worst > 5.0
    bad = cycle_check(sol, tol=2.0)
    assert ("A", "C") in bad


def test_inverse_variance_weighting_trusts_precise_edges():
    # B's offset is constrained by one precise edge (A-B) and one noisy edge.
    gt = {"A": 0.0, "B": 10.0}
    precise = Edge("A", "B", gt["A"] - gt["B"], weight=100.0)         # says b_A-b_B = -10
    noisy = Edge("A", "B", gt["A"] - gt["B"] + 6.0, weight=0.5)       # says -4, low weight
    extents = {"A": (0.0, 20.0), "B": (0.0, 20.0)}
    sol = solve_timeline(["A", "B"], [precise, noisy], extents, reference="A")
    # weighted solution should sit very close to the precise edge (-10), not the midpoint
    assert sol.offsets["B"] == pytest.approx(10.0, abs=0.2)


def test_connected_components():
    nodes = ["a", "b", "c", "x", "y", "z"]
    edges = [Edge("a", "b", 0), Edge("b", "c", 0), Edge("x", "y", 0)]
    comps = connected_components(nodes, edges)
    sets = sorted((sorted(c) for c in comps), key=len, reverse=True)
    assert sorted(sets[0]) == ["a", "b", "c"]
    assert ["x", "y"] in [sorted(c) for c in comps]
    assert ["z"] in [sorted(c) for c in comps]


def test_single_node_no_edges():
    sol = solve_timeline(["solo"], [], {"solo": (0.0, 12.0)}, reference="solo")
    assert sol.offsets == {"solo": 0.0}
    assert sol.intervals["solo"] == (0.0, 12.0)
    assert sol.max_abs_residual == 0.0
