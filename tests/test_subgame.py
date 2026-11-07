"""
The real-time subgame solver in solver/subgame.cpp: known solutions, exact
equities against the engine's evaluator, and its tree against the real GameState.
Skipped without g++.
"""

import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from card_utils import Card, evaluate, full_deck
from game_state import GameState
from player_action import PlayerAction
from solver.reader import hole_index
from solver.subgame import CHANCE, FOLD, HOLES, PLAYER, SHOWDOWN, Subgame, load_library
from solver.translation import SubgameError, card_number

SOLVER = Path(__file__).resolve().parent.parent / "solver"


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is not installed")
    path = tmp_path_factory.mktemp("build") / "subgame.dll"
    subprocess.run(
        ["g++", "-std=c++20", "-O3", "-march=native", "-fopenmp-simd",
         "-fno-trapping-math", "-shared", "-static", "-pthread",
         str(SOLVER / "subgame.cpp"), "-o", str(path)],
        check=True,
    )
    return load_library(str(path))


@pytest.fixture(scope="module")
def solver(library):
    return Subgame(library, threads=2)


def cards(text):
    return [card_number(Card.parse(t)) for t in text.split()]


def weights(*hands):
    """A range from (cards text, weight) pairs."""
    out = [0.0] * HOLES
    for text, weight in hands:
        out[hole_index(*cards(text))] = weight
    return out


def combos(rank, exclude=""):
    gone = set(exclude.split())
    held = [rank + s for s in "CDHS" if rank + s not in gone]
    return [
        f"{held[i]} {held[j]}"
        for i in range(len(held)) for j in range(i)
    ]


def test_river_bluffing_game_has_its_known_solution(solver):
    # Half-pot all-in with nuts or air, against a pure bluff-catcher. Theory: the
    # caller calls P / (P + B) = 2/3 and bluffs are B / (P + 2B) = 1/4 of bets.
    board = "3C 4D 8H TS KC"
    kings = combos("K", exclude="KC")
    twos = combos("2")
    queens = combos("Q")
    assert (len(kings), len(twos)) == (3, 6)
    bettor = weights(*[(h, 1.0) for h in kings + twos])
    catcher = weights(*[(h, 1.0) for h in queens])
    result = solver.solve(
        cards(board), [100, 100], [100, 100], [0, 0], [False, True], 20, 20, 0,
        True, bettor, catcher, budget_ms=100, prune=0.0,
    )
    assert result.exploitability < 0.05  # chips, on a pot of 200
    kind, player, to = solver.node()
    assert (kind, player, to) == (PLAYER, 0, (0, 100))

    root = solver.strategy(2)
    for hand in kings:
        assert root[HOLES + hole_index(*cards(hand))] == pytest.approx(1, abs=0.01)
    bluffs = sum(root[HOLES + hole_index(*cards(h))] for h in twos)
    assert bluffs == pytest.approx(1, abs=0.03), "1 combo's weight for 3 of value"

    solver.advance(1)
    kind, player, to = solver.node()
    assert (kind, player, to) == (PLAYER, 1, (-1, 0))
    call = solver.strategy(2)
    for hand in queens:
        assert call[HOLES + hole_index(*cards(hand))] == pytest.approx(2 / 3, abs=0.01)

    # 3 kings earn 166.67, and 6 twos lose 100 whether they bluff or give up.
    assert result.value[0] == pytest.approx((3 * 500 / 3 - 6 * 100) / 9, abs=0.2)
    assert result.value[1] == pytest.approx(-result.value[0], abs=1e-3)


def _showdown_ev(hero, villain, board, stake):
    """Exact value for hero, by the engine's evaluator over every river."""
    used = set(hero + villain + board)
    total, count = 0, 0
    for river in full_deck():
        if str(river) in used:
            continue
        table = [Card.parse(c) for c in board] + [river]
        ours = evaluate([Card.parse(c) for c in hero] + table)
        theirs = evaluate([Card.parse(c) for c in villain] + table)
        total += stake * ((ours > theirs) - (ours < theirs))
        count += 1
    assert count == 44
    return total / count


def test_turn_equity_is_exact_and_blockers_remove_impossible_pairs(solver):
    # The button is all-in; we can only call or fold. Our AS KH shares a card with
    # one of its hands, so that hand must not count.
    board = "2C 7D 9H JS"
    hero = ["AS", "KH"]
    possible = [["QS", "QH"], ["TD", "8C"]]
    blocked = "AS QD"
    theirs = weights(
        (" ".join(possible[0]), 1.0), (" ".join(possible[1]), 1.0), (blocked, 5.0)
    )
    ours = weights((" ".join(hero), 1.0))
    result = solver.solve(
        cards(board), [500, 100], [0, 400], [400, 0], [True, False], 400, 20, 1,
        False, theirs, ours, budget_ms=30, prune=0.0,
    )
    call_ev = sum(_showdown_ev(hero, v, board.split(), 500) for v in possible) / 2
    assert result.value[1] == pytest.approx(max(call_ev, -100), abs=0.05)
    assert result.exploitability < 0.01


def test_a_real_river_range_converges_and_stays_zero_sum(solver):
    board = cards("AS KD 7H 3C 9D")
    full = [1.0] * HOLES
    short = solver.solve(board, [100, 100], [400, 400], [0, 0], [False, False], 20, 20,
                         1, True, full, full, budget_ms=5)
    long = solver.solve(board, [100, 100], [400, 400], [0, 0], [False, False], 20, 20,
                        1, True, full, full, budget_ms=200)
    assert long.iterations > short.iterations
    assert long.exploitability < short.exploitability
    assert long.exploitability < 0.002 * 200, "under 0.2% of the pot"
    assert long.value[0] + long.value[1] == pytest.approx(0, abs=0.01)


@pytest.mark.parametrize("board", ["AS KD 7H 3C", "AS KD 7H 3C 9D"])
def test_the_loop_allocates_nothing_and_keeps_the_budget(library, board):
    subgame = Subgame(library, threads=2)
    full = [1.0] * HOLES
    for _ in range(3):
        result = subgame.solve(cards(board), [100, 100], [400, 400], [0, 0],
                               [False, False], 20, 20, 1, True, full, full, budget_ms=200)
        assert result.loop_news == 0
        assert result.iterations > 1
        # Generous: a test machine may be busy. The target is 200.
        assert result.elapsed_ms < 400


def test_strategy_sums_to_one_in_range_and_zero_outside(solver):
    board = cards("AS KD 7H 3C 9D")
    in_range = weights(("QS QH", 1.0), ("JS JH", 1.0))
    solver.solve(board, [100, 100], [400, 400], [0, 0], [False, False], 20, 20, 1,
                 True, in_range, in_range, budget_ms=5)
    _, _, to = solver.node()
    strategy = solver.strategy(len(to))
    for h in range(HOLES):
        total = sum(strategy[a * HOLES + h] for a in range(len(to)))
        if in_range[h]:
            assert total == pytest.approx(1)
        else:
            assert total == 0


@pytest.mark.parametrize("bad", [
    dict(board=cards("AS AS 7H 3C")),
    dict(board=cards("AS KD 7H")),
    dict(board=[99, 1, 2, 3]),
    dict(to_act=2),
    dict(range0=[math.nan] * HOLES),
    dict(range1=[-1.0] * HOLES),
    dict(range0=[0.0] * HOLES),
    dict(budget_ms=0),
])
def test_bad_input_is_refused_not_crashed_on(solver, bad):
    arguments = dict(
        board=cards("AS KD 7H 3C"), invested=[100, 100], stacks=[400, 400], bets=[0, 0],
        acted=[False, False], last_raise=20, big_blind=20, to_act=1, can_raise=True,
        range0=[1.0] * HOLES, range1=[1.0] * HOLES, budget_ms=5,
    )
    arguments.update(bad)
    with pytest.raises(SubgameError):
        solver.solve(**arguments)


def _random_real_state(rng):
    """A real state on the turn or river, reached by legal play from random stacks."""
    deck = full_deck()
    rng.shuffle(deck)
    stacks = [rng.randint(100, 800), rng.randint(100, 800)]
    if rng.random() < 0.5:
        stacks[1] = stacks[0]
    state = GameState(list(stacks))
    state.setup_hand(list(stacks), 0, 10, 20)
    target = rng.choice([4, 5])
    cursor = iter(deck)
    for _ in range(200):
        if state.is_hand_over:
            return None
        if state.is_street_closed:
            state.deal_card(next(cursor))
            continue
        if len(state.board) == target and rng.random() < 0.6:
            return state, stacks, deck
        p = state.to_act
        if state.can_raise and rng.random() < 0.35:
            low, high = state.raise_range
            action = PlayerAction.raise_to(p, rng.randint(low, high))
        else:
            action = PlayerAction.call(p)
        state.process_action(action)
    return None


def _walk(subgame, state, rng):
    """Follows one random line down the solved tree on the real GameState."""
    deck = [c for c in full_deck() if c not in state.board]
    rng.shuffle(deck)
    for _ in range(100):
        kind, player, to = subgame.node()
        if kind == CHANCE:
            assert state.is_street_closed
            assert state.street == "turn"
            card = deck.pop()
            state.deal_card(card)
            subgame.deal(card_number(card))
        elif kind == PLAYER:
            while state.is_street_closed:
                state.deal_card(deck.pop())
            assert player == state.to_act
            action = rng.randrange(len(to))
            total = to[action]
            if total == -1:
                assert state.call_amount > 0
                move = PlayerAction.fold(player)
            elif total == 0:
                move = PlayerAction.call(player)
            else:
                move = PlayerAction.raise_to(player, total)
            state.process_action(move)  # raises InvalidAction if illegal
            subgame.advance(action)
        else:
            assert kind in (FOLD, SHOWDOWN)
            if kind == FOLD:
                assert state.folded[player]
                return
            while not state.is_hand_over:
                assert state.is_street_closed, "the engine still expects a decision"
                state.deal_card(deck.pop())
            return
    pytest.fail("the line never ended")


def test_every_line_of_the_tree_is_legal_from_random_real_states(library):
    subgame = Subgame(library, threads=2, reserve_mb=8)
    rng = random.Random(11)
    full = [1.0] * HOLES
    states = 0
    while states < 150:
        made = _random_real_state(rng)
        if made is None:
            continue
        state, stacks, _ = made
        states += 1
        board = [card_number(c) for c in state.board]
        subgame.solve(
            board, [stacks[p] - state.stacks[p] for p in (0, 1)], state.stacks, state.bets,
            state.acted, state.last_raise, state.big_blind, state.to_act, state.can_raise,
            full, full, budget_ms=1, max_table_mb=rng.choice([1, 4, 64]),
        )
        _walk(subgame, state, rng)
