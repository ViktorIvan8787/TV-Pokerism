"""
The pure-Python blueprint against solver/reader.py and the engine's evaluator, and
the real build played from its pickle as in the arena. Needs the files in
solver/out, which are not in the repo, so it is skipped without them.
"""

import pickle
import random
from pathlib import Path

import pytest

from arena.dummies import DUMMIES
from arena.match import MatchConfig, play_match
from arena.players import player
from blueprint import rank7, river_score
import model as model_module
from card_utils import evaluate, full_deck
from engine import play_hand
from evals.score import score
from game_state import GameState, GameStateView
from model import Model
from player_action import PlayerAction
from solver.reader import Strategy

OUT = Path(__file__).resolve().parent.parent / "solver" / "out"
pytestmark = pytest.mark.skipif(
    not (OUT / "river_table.bin").exists(), reason="solver/out is not built"
)
DECK = full_deck()  # card number = its index here


def test_rank7_orders_hands_like_the_engine():
    rng = random.Random(1)
    hands = []
    for _ in range(4000):
        # Few ranks and suits reach the rare categories.
        pool = rng.sample(range(52), rng.choice([52, 16, 12]))
        hands.append(rng.sample(pool, 7))
    # Straight flushes and wheels, which random hands rarely make.
    hands += [[48, 0, 4, 8, 12, 25, 30], [32, 28, 24, 20, 16, 1, 2], [48, 44, 40, 36, 32, 1, 5]]
    ranks =[(rank7(h), evaluate([DECK[c] for c in h])) for h in hands]
    ranks.sort(key=lambda pair: pair[0])
    for (mine_a, theirs_a), (mine_b, theirs_b) in zip(ranks, ranks[1:]):
        assert (mine_a < mine_b) == (theirs_a < theirs_b)
        assert (mine_a == mine_b) == (theirs_a == theirs_b)


@pytest.fixture(scope="module")
def built():
    import os

    from make_model import build

    here = os.getcwd()
    os.chdir(OUT.parent.parent)
    try:
        return build()
    finally:
        os.chdir(here)


def test_lookups_match_the_files_on_every_street(built):
    book = built.book
    real = Strategy(str(OUT / "strategy.bin"), str(OUT / "abstraction.bin"))
    assert (book.stack, book.small_blind, book.big_blind, book.buckets) == (
        real.stack, real.small_blind, real.big_blind, real.buckets
    )
    rng = random.Random(2)
    for _ in range(3000):
        index = rng.randrange(real.node_count)
        node = real.node(index)
        assert tuple(book.node(index)) == tuple(node)
        if node.kind == 0:
            bucket = rng.randrange(real.buckets[node.street])
            assert book.probabilities(node, bucket) == real.probabilities(node, bucket)
    for street_cards in (0, 3, 4):
        for _ in range(3000):
            cards = rng.sample(range(52), 2 + street_cards)
            hole, board = tuple(cards[:2]), tuple(cards[2:])
            assert book.bucket(hole, board) == real.bucket(hole, board)
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        hole, board = tuple(cards[:2]), tuple(cards[2:])
        assert book.bucket(hole, board) == real.bucket(hole, board), (hole, board)


def test_river_score_counts_against_990_hands():
    nuts = (48, 44)  # AC KC with the broadway board below
    assert river_score(nuts, (40, 36, 32, 1, 2)) == 1980


class _BrokenBook:
    """Every lookup fails, so the model cannot get past its first line of logic."""

    def __getattr__(self, name):
        raise RuntimeError("forced error")


def _facing(*moves):
    state = GameState([500, 500])
    state.setup_hand([500, 500], 0, 10, 20)
    for move in moves:
        state.process_action(move)
    return GameStateView(state)


@pytest.mark.parametrize(
    "moves, owed",
    [([], True), ([PlayerAction.raise_to(0, 60)], True), ([PlayerAction.call(0)], False)],
    ids=["blind", "raise", "check"],
)
def test_a_forced_error_returns_a_legal_action(moves, owed):
    view = _facing(*moves)
    model = Model(_BrokenBook())
    model.seed([DECK[0], DECK[5]], view.to_act)
    assert (view.call_amount > 0) == owed
    move = model.run([], view)
    view.validate_action(move)  # raises if illegal
    assert move.kind == "call" and move.player_id == view.to_act


def test_a_forced_error_folds_if_a_call_is_not_allowed():
    class FoldOnly:
        legal_actions = ("fold",)

    model = Model(_BrokenBook())
    model.player_id = 1
    move = model.run([], FoldOnly())
    assert (move.kind, move.player_id) == ("fold", 1)


def test_an_illegal_decision_is_replaced_by_a_legal_action(monkeypatch):
    class Illegal:
        def __init__(self, *args):
            pass

        def decide(self, view, hole, board):
            return PlayerAction.raise_to(1, 10_000)

    monkeypatch.setattr(model_module, "Translator", Illegal)
    view = _facing()
    model = Model(_BrokenBook())
    model.seed([DECK[0], DECK[5]], view.to_act)
    move = model.run([], view)
    view.validate_action(move)
    assert move.kind == "call"


def test_forced_errors_never_fault_a_whole_hand():
    deck = full_deck()
    random.Random(4).shuffle(deck)
    for button in (0, 1):
        result = play_hand([Model(_BrokenBook()), Model()], deck, [500, 500], button=button)
        assert result.faults == {} and sum(result.net) == 0


@pytest.fixture(scope="module")
def you(built, tmp_path_factory):
    path = tmp_path_factory.mktemp("you") / "you.pkl"
    path.write_bytes(pickle.dumps(built, protocol=4))
    assert path.stat().st_size < 50 << 20
    return player(str(path))


def _check(players, **config):
    result = score([play_match(players, MatchConfig(**config))])
    assert result.faults[0] == 0, result.first_faults[0]


@pytest.mark.parametrize("opponent", list(DUMMIES))
def test_the_real_build_plays_legally_from_its_pickle(you, opponent):
    _check([you, player(opponent)], deals=10)


def test_the_real_build_at_a_full_table_with_short_stacks(you):
    others = [player(name) for name in ("shove", "random", "call")]
    _check([you] + others, deals=10, stack=60)
