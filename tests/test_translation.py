"""The bet-size mappings and the deviation matrix in solver/translation.py."""

import math
import random

import pytest

from solver.translation import (
    DeviationMatrix,
    choose_anchor,
    pseudo_harmonic,
    randomized_geometric,
    threshold_pseudo_harmonic,
    wilson_lower,
)

MAPPINGS = [pseudo_harmonic, randomized_geometric, threshold_pseudo_harmonic]


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("a, b", [(0.5, 1.0), (1.0, 24.0), (0.0, 0.5), (0.1, 0.2)])
def test_exact_sizes_map_to_themselves(mapping, a, b):
    assert mapping(a, a, b) == pytest.approx(1)
    assert mapping(b, a, b) == pytest.approx(0)


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("a, b", [(0.5, 1.0), (1.0, 24.0), (0.0, 0.5)])
def test_a_bigger_bet_is_never_more_likely_to_be_read_small(mapping, a, b):
    steps = [a + (b - a) * i / 200 for i in range(201)]
    values = [mapping(x, a, b) for x in steps]
    assert all(0 <= v <= 1 for v in values)
    assert all(p >= q - 1e-12 for p, q in zip(values, values[1:]))


def test_geometric_is_even_at_the_geometric_mean():
    assert randomized_geometric(math.sqrt(0.5), 0.5, 1) == pytest.approx(0.5)


def test_pseudo_harmonic_is_even_at_its_published_boundary():
    a, b = 0.5, 1.0
    boundary = (a + b + 2 * a * b) / (a + b + 2)
    assert pseudo_harmonic(boundary, a, b) == pytest.approx(0.5)


def test_a_037_pot_bet_is_mostly_read_as_half_the_pot():
    assert pseudo_harmonic(0.37, 0, 0.5) == pytest.approx(0.19, abs=0.01)


def test_geometric_does_not_read_a_tiny_bet_as_a_big_one():
    assert randomized_geometric(0.01, 0, 0.5) > 0.95


def test_threshold_snaps_far_from_the_boundary_and_stays_soft_near_it():
    assert threshold_pseudo_harmonic(0.05, 0, 0.5) == 1
    assert threshold_pseudo_harmonic(0.45, 0, 0.5) == 0
    near = 0.2
    assert threshold_pseudo_harmonic(near, 0, 0.5) == pseudo_harmonic(near, 0, 0.5)


def test_choose_anchor_clamps_outside_the_tree():
    rng = random.Random(1)
    anchors = [0.0, 0.5, 1.0, 12.0]
    assert choose_anchor(-1, anchors, pseudo_harmonic, rng) == 0
    assert choose_anchor(99, anchors, pseudo_harmonic, rng) == 3


def test_choose_anchor_uses_only_the_two_neighbours_at_the_mapped_rate():
    rng = random.Random(2)
    anchors = [0.0, 0.5, 1.0, 12.0]
    picks = [choose_anchor(0.7, anchors, pseudo_harmonic, rng) for _ in range(20000)]
    assert set(picks) == {1, 2}
    assert picks.count(1) / len(picks) == pytest.approx(
        pseudo_harmonic(0.7, 0.5, 1.0), abs=0.015
    )


def test_wilson_lower_is_cautious_with_few_samples():
    assert wilson_lower(0, 0) == 0
    assert wilson_lower(3, 3) < 0.5 < wilson_lower(300, 300)


def _fill(matrix, street, column, count):
    for _ in range(count):
        matrix.record("them", street, column)


def test_a_fixed_weird_size_is_frequent_but_not_erratic():
    matrix = DeviationMatrix()
    _fill(matrix, 1, DeviationMatrix.size_column(0.37), 100)
    [flag] = matrix.flags("them")
    assert flag["street"] == "flop"
    assert flag["kinds"] == ["frequent"]
    assert flag["off_tree"] == 1


def test_scattered_sizes_are_flagged_erratic():
    matrix = DeviationMatrix()
    for column in range(len(DeviationMatrix.COLUMNS)):
        _fill(matrix, 2, column, 20)
    [flag] = matrix.flags("them")
    assert flag["street"] == "turn"
    assert set(flag["kinds"]) == {"frequent", "erratic"}


def test_on_tree_play_and_small_samples_raise_no_flag():
    matrix = DeviationMatrix()
    _fill(matrix, 0, DeviationMatrix.ON_TREE, 500)
    _fill(matrix, 3, DeviationMatrix.size_column(0.37), 10)
    assert matrix.flags("them") == []
    assert matrix.flags("nobody") == []


def test_size_columns_cover_the_range():
    labels = DeviationMatrix.COLUMNS
    assert labels[DeviationMatrix.size_column(0.1)] == "<0.25"
    assert labels[DeviationMatrix.size_column(0.37)] == "0.25-0.5"
    assert labels[DeviationMatrix.size_column(0.75)] == "0.5-1"
    assert labels[DeviationMatrix.size_column(1.5)] == "1-2"
    assert labels[DeviationMatrix.size_column(9)] == ">2"


def test_report_names_the_team_and_the_flag():
    matrix = DeviationMatrix()
    _fill(matrix, 1, DeviationMatrix.size_column(0.37), 100)
    report = matrix.report()
    assert "them:" in report and "FLAG flop: frequent" in report
