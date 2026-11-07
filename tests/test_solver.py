"""
The MCCFR pipeline in solver/: its evaluator and betting tree against the real
game code, and a small end-to-end run. Skipped without g++. About 1 minute:
the abstraction has to sweep every board even with few buckets.
"""

import copy
import random
import shutil
import subprocess
from pathlib import Path

import pytest

from card_utils import Card, evaluate, full_deck
from engine import play_hand
from game_state import GameState, GameStateView
from player_action import PlayerAction
from solver.reader import DECISION, FOLD, SHOWDOWN, Strategy, hole_index
from solver.translation import (
    DeviationMatrix,
    SubgameError,
    Translator,
    threshold_pseudo_harmonic,
)

SOLVER = Path(__file__).resolve().parent.parent / "solver"


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is not installed")
    path = tmp_path_factory.mktemp("build") / "cfr.exe"
    subprocess.run(
        ["g++", "-std=c++20", "-O3", "-march=native", "-fopenmp-simd",
         "-static", "-pthread", str(SOLVER / "main.cpp"), "-o", str(path)],
        check=True,
    )
    return path


@pytest.fixture(scope="module")
def trained(binary, tmp_path_factory):
    folder = tmp_path_factory.mktemp("out")
    abstraction = folder / "abstraction.bin"
    strategy = folder / "strategy.bin"
    subprocess.run(
        [binary, "abstract", "--flop", "8", "--turn", "8", "--river", "8",
         "--out", abstraction],
        check=True, capture_output=True,
    )
    subprocess.run(
        [binary, "train", "--abs", abstraction, "--out", strategy,
         "--iters", "20000", "--epoch", "5000"],
        check=True, capture_output=True,
    )
    return Strategy(str(strategy), str(abstraction)), strategy, abstraction


def test_kuhn_solution_and_equity_sweep(binary):
    done = subprocess.run([binary, "selfcheck"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def _stress_hands(rng):
    """Random hands, plus hands crowded into few suits and ranks, which reach
    the rare categories (straight flushes, quads, full houses, wheels)."""
    deck = full_deck()
    for _ in range(1500):
        yield rng.sample(deck, 7)
    for _ in range(1500):
        suits = rng.sample("CDHS", rng.choice([1, 2]))
        pool = [c for c in deck if c.suit in suits]
        yield rng.sample(pool, 7)
    for _ in range(1500):
        ranks = rng.sample("23456789TJQKA", 4)
        pool = [c for c in deck if c.rank in ranks]
        yield rng.sample(pool, 7)
    wheel = [c for c in deck if c.rank in "A2345" or c.rank == "K"]
    for _ in range(500):
        yield rng.sample(wheel, 7)


def test_evaluator_orders_hands_like_the_engine(binary):
    deck = full_deck()
    index = {card: i for i, card in enumerate(deck)}
    hands = list(_stress_hands(random.Random(3)))
    lines = "\n".join(" ".join(str(index[c]) for c in h) for h in hands)
    out = subprocess.run(
        [binary, "evalcheck"], input=lines, capture_output=True, text=True,
        check=True,
    ).stdout.split()
    ours = [int(x) for x in out]
    theirs = [evaluate(h) for h in hands]
    assert len(ours) == len(hands)
    order = random.Random(4).sample(range(len(hands)), len(hands))
    for i, j in zip(order, order[1:]):
        assert (ours[i] > ours[j]) == (theirs[i] > theirs[j])
        assert (ours[i] == ours[j]) == (theirs[i] == theirs[j])


def _deal_until_betting_or_end(state):
    for card in full_deck():
        if state.is_hand_over or not state.is_street_closed:
            return
        if card not in state.board:
            state.deal_card(card)


def _walk(book, index, state):
    """Replays one tree node on a real GameState, then each of its children."""
    node = book.node(index)
    invested = [book.stack - chips for chips in state.stacks]
    if node.kind == FOLD:
        assert state.folded[node.player]
        assert node.to[0] == invested[node.player]
        return
    _deal_until_betting_or_end(state)
    if node.kind == SHOWDOWN:
        assert state.is_hand_over, "the engine still expects a decision"
        assert invested == [node.to[0], node.to[0]]
        return
    assert node.kind == DECISION
    assert ["preflop", "flop", "turn", "river"][node.street] == state.street
    assert node.player == state.to_act
    for a in range(node.nact):
        total = node.to[a]
        if total == -1:
            assert state.call_amount > 0, "fold is only offered when chips are owed"
            action = PlayerAction.fold(state.to_act)
        elif total == 0:
            action = PlayerAction.call(state.to_act)
        else:
            action = PlayerAction.raise_to(state.to_act, total)
        following = copy.deepcopy(state)
        following.process_action(action)  # raises InvalidAction if illegal
        _walk(book, node.child[a], following)


def test_every_tree_node_is_legal_in_the_real_game(trained):
    book, _, _ = trained
    state = GameState([book.stack, book.stack])
    state.setup_hand([book.stack, book.stack], 0, book.small_blind, book.big_blind)
    _walk(book, 0, state)


def test_probabilities_sum_to_one_at_the_root(trained):
    book, _, _ = trained
    root = book.node(0)
    for bucket in range(book.buckets[0]):
        assert sum(book.probabilities(root, bucket)) == pytest.approx(1)


def test_buckets_ignore_suit_names(trained):
    book, _, _ = trained
    rng = random.Random(5)
    swaps = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 1, 0, 2]]
    for _ in range(300):
        cards = rng.sample(range(52), 7)
        hole, board = cards[:2], cards[2:7]
        for size in (0, 3, 4, 5):
            buckets = {
                book.bucket(
                    [c // 4 * 4 + swap[c % 4] for c in hole],
                    [c // 4 * 4 + swap[c % 4] for c in board[:size]],
                )
                for swap in swaps
            }
            assert len(buckets) == 1


def test_river_buckets_rank_by_strength(trained):
    book, _, _ = trained
    number = {str(c): i for i, c in enumerate(full_deck())}
    board = [number[c] for c in ("2C", "7D", "9H", "JS", "KC")]
    nuts_ish = [number["KD"], number["KH"]]
    trash = [number["3D"], number["4H"]]
    assert book.bucket(nuts_ish, board) > book.bucket(trash, board)


def test_rejects_a_strategy_from_another_abstraction(trained, tmp_path):
    _, strategy, abstraction = trained
    forged = bytearray(strategy.read_bytes())
    forged[8] ^= 0xFF  # the abstraction id
    other = tmp_path / "other.bin"
    other.write_bytes(forged)
    with pytest.raises(ValueError, match="another abstraction"):
        Strategy(str(other), str(abstraction))


def test_training_resumes_from_its_checkpoint(trained, binary, tmp_path):
    _, strategy, abstraction = trained
    # A new name: Windows cannot replace the strategy the fixture has mapped.
    out = tmp_path / "resumed.bin"
    shutil.copyfile(f"{strategy}.ckpt", f"{out}.ckpt")
    done = subprocess.run(
        [binary, "train", "--abs", abstraction, "--out", out,
         "--iters", "25000", "--epoch", "5000", "--resume", "1"],
        check=True, capture_output=True, text=True,
    )
    assert "resumed at 20000" in done.stdout
    assert Strategy(str(out), str(abstraction)).iterations == 25000


class _Sizer:
    """Raises to a chosen pot fraction, off the tree, whenever it can."""

    def __init__(self, pick, rate=0.5):
        self.pick, self.rate = pick, rate
        self.rng = random.Random(8)

    def seed(self, hand, player_id):
        self.player_id = player_id

    def run(self, actions, game_state):
        me = self.player_id
        if game_state.can_raise and self.rng.random() < self.rate:
            low, high = game_state.raise_range
            total = game_state.current_bet + int(
                self.pick(self.rng) * (game_state.pot + game_state.call_amount)
            )
            return PlayerAction.raise_to(me, max(low, min(high, total)))
        return PlayerAction.call(me)


@pytest.mark.parametrize("mapping", ["threshold", "geometric"])
@pytest.mark.parametrize(
    "pick",
    [lambda r: 0.0, lambda r: 0.37, lambda r: 0.75, lambda r: 1.6,
     lambda r: r.uniform(0, 3), lambda r: 99.0],
    ids=["min", "0.37", "0.75", "1.6", "uniform", "shove"],
)
def test_off_tree_bets_never_fault_or_lose_track(trained, monkeypatch, mapping, pick):
    import solver.bot as bot

    book, _, _ = trained
    monkeypatch.setattr(bot, "_book", book)

    class Checked(bot.SolverBot):
        """Also checks, every turn, that the replayed state is the engine's. A
        failed assert here is an exception in run, so the engine logs a fault."""

        def run(self, actions, game_state):
            move = super().run(actions, game_state)
            real = self.translator.real
            assert real is not None
            assert (real.stacks, real.bets, real.pot, real.street) == (
                list(game_state.stacks), list(game_state.bets),
                game_state.pot, game_state.street,
            )
            return move

    deck = full_deck()
    rng = random.Random(6)
    sizer = _Sizer(pick)
    for hand in range(150):
        rng.shuffle(deck)
        result = play_hand(
            [Checked(mapping, resolve=False), sizer], deck, [500, 500], button=hand % 2
        )
        assert result.faults == {}
        assert sum(result.net) == 0


def test_our_raise_is_rebuilt_on_the_real_pot(trained):
    book, _, _ = trained

    class AlwaysPotRaise:
        """A blueprint that always takes the 1x pot raise: second largest."""

        def __getattr__(self, name):
            return getattr(book, name)

        def probabilities(self, node, bucket):
            target = [a for a in range(node.nact) if node.to[a] > 0][-2]
            return [float(a == target) for a in range(node.nact)]

    # We are the big blind. The button raises to 75, off the tree (40, 60, 500);
    # that is read as 60, so the abstract pot (80) differs from the real one (95).
    state = GameState([500, 500])
    state.setup_hand([500, 500], 0, 10, 20)
    translator = Translator(
        AlwaysPotRaise(), GameStateView(state), 1, threshold_pseudo_harmonic,
        DeviationMatrix(), random.Random(1),
    )
    action = PlayerAction.raise_to(0, 75)
    translator.observe(action)
    state.process_action(action)
    move = translator.decide(GameStateView(state), (0, 5), ())
    assert move.kind == "raise"
    state.validate_action(move)
    # 1x the real pot after calling: 75 + (95 + 55). Not the tree's own 180.
    assert move.amount == 225


def test_tiny_bet_read_as_a_check_is_paid_and_stays_in_step(trained):
    book, _, _ = trained

    class AlwaysPassive:
        """A blueprint that always checks or calls."""

        def __getattr__(self, name):
            return getattr(book, name)

        def probabilities(self, node, bucket):
            return [float(node.to[a] == 0) for a in range(node.nact)]

    class ReadsSmall(random.Random):
        """Every randomized mapping picks its smaller size."""

        def random(self):
            return 0.0

    state = GameState([500, 500])
    state.setup_hand([500, 500], 0, 10, 20)
    view = GameStateView(state)
    matrix = DeviationMatrix()
    translator = Translator(
        AlwaysPassive(), view, 1, threshold_pseudo_harmonic, matrix, ReadsSmall()
    )

    def both(action):
        translator.observe(action)
        state.process_action(action)

    def deal(*cards):
        for text in cards:
            both(PlayerAction.reveal_card(Card.parse(text)))

    hole = (0, 5)  # 2C 3D
    both(PlayerAction.raise_to(0, 60))
    both(translator.decide(view, hole, ()))
    deal("9H", "TH", "JH")
    board = (30, 34, 38)
    both(translator.decide(view, hole, board))  # we check first, as the big blind

    # The button closes the flop in the tree by checking, but really bets 20 into
    # 120. Read as a check, the abstract hand is already on the turn.
    both(PlayerAction.raise_to(0, 20))
    assert translator.catch_up
    assert book.node(translator.node).street == 2
    assert matrix.counts["seat 0"][1][DeviationMatrix.size_column(20 / 120)] == 1

    move = translator.decide(view, hole, board)
    assert move.kind == "call"
    assert book.node(translator.node).street == 2, "paying does not move the tree"
    both(move)
    deal("2S")
    assert state.street == "turn"
    node = book.node(translator.node)
    assert (node.kind, node.street, node.player) == (DECISION, 2, 1)
    assert translator.decide(view, hole, board + (3,)).kind == "call"


@pytest.fixture(scope="module")
def live_library(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is not installed")
    path = tmp_path_factory.mktemp("live") / "subgame.dll"
    subprocess.run(
        ["g++", "-std=c++20", "-O3", "-march=native", "-fopenmp-simd",
         "-fno-trapping-math", "-shared", "-static", "-pthread",
         str(SOLVER / "subgame.cpp"), "-o", str(path)],
        check=True,
    )
    return str(path)


def _live_bot(bot, monkeypatch, book, library_path):
    from solver.subgame import load_library

    monkeypatch.setattr(bot, "_book", book)
    monkeypatch.setattr(bot, "load_library", lambda: load_library(library_path))


@pytest.mark.parametrize(
    "pick", [lambda r: 0.37, lambda r: r.uniform(0, 3), lambda r: 99.0],
    ids=["0.37", "uniform", "shove"],
)
def test_live_solving_plays_legally_and_replaces_the_blueprint(
    trained, monkeypatch, live_library, pick
):
    import solver.bot as bot

    book, _, _ = trained
    _live_bot(bot, monkeypatch, book, live_library)
    ours = bot.SolverBot(budget_ms=3)
    solves = []
    original = ours.run

    def run(actions, game_state):
        move = original(actions, game_state)
        resolver = ours.translator.resolver
        if len(game_state.board) >= 4:
            assert resolver is not None, "the blueprint still decided on the turn"
            solves.append(resolver.last.iterations)
        return move

    ours.run = run
    deck = full_deck()
    rng = random.Random(9)
    sizer = _Sizer(pick)  # one for all hands: a new one would replay its seed
    for hand in range(150):
        rng.shuffle(deck)
        result = play_hand([ours, sizer], deck, [500, 500], button=hand % 2)
        assert result.faults == {}
    if pick(rng) != 99.0:
        assert solves, "no hand reached the turn"


def test_a_failed_solve_falls_back_without_a_fault(trained, monkeypatch, live_library):
    import solver.bot as bot
    from solver import subgame

    book, _, _ = trained
    _live_bot(bot, monkeypatch, book, live_library)

    def broken(self, *args, **kwargs):
        raise SubgameError("no")

    monkeypatch.setattr(subgame.Subgame, "solve", broken)
    deck = full_deck()
    rng = random.Random(10)
    sizer = _Sizer(lambda r: 0.5)
    for hand in range(60):
        rng.shuffle(deck)
        result = play_hand(
            [bot.SolverBot(budget_ms=3), sizer], deck, [500, 500],
            button=hand % 2,
        )
        assert result.faults == {}


def test_a_missing_library_means_blueprint_only(trained, monkeypatch, tmp_path):
    import solver.bot as bot
    from solver.subgame import load_library

    book, _, _ = trained
    monkeypatch.setattr(bot, "_book", book)
    missing = str(tmp_path / "nothing.dll")
    monkeypatch.setattr(bot, "load_library", lambda: load_library(missing))
    deck = full_deck()
    ours = bot.SolverBot()
    result = play_hand([ours, _Sizer(lambda r: 0.5)], deck, [500, 500])
    assert result.faults == {}
    assert ours.resolve is False


def test_a_shared_range_comes_from_the_blueprint_path(trained):
    from solver.subgame import blueprint_ranges

    book, _, _ = trained
    root = book.node(0)
    board = (0, 5, 10, 15)
    fold_or_not = root.to.index(0, 0, root.nact)
    reach = blueprint_ranges(book, [(0, fold_or_not)], board)
    # Hands on the board are impossible for both seats; only seat 0 acted.
    assert reach[0][hole_index(5, 0)] == 0
    assert reach[1][hole_index(5, 0)] == 0
    free = hole_index(51, 50)
    assert reach[1][free] == 1
    assert reach[0][free] == pytest.approx(
        book.probabilities(root, book.bucket((51, 50)))[fold_or_not]
    )
