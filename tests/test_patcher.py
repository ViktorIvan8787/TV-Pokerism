"""
The background patcher in solver/patcher.py: its tree against the real GameState,
its deals against the blueprint, the regret gate, the swap under concurrent
readers, and that a solve does not stall the game loop. Skipped without g++.
"""

import ctypes
import functools
import pickle
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from card_utils import full_deck
from engine import play_hand
from game_state import GameState
from player_action import PlayerAction
from solver.patcher import Patcher, _map_for_pointer, _Root, _Step, load_library, make_spot, size_bin
from solver.reader import DECISION
from solver.subgame import blueprint_ranges
from test_solver import _live_bot, _Sizer, _walk, binary, live_library, trained  # noqa: F401

SOLVER = Path(__file__).resolve().parent.parent / "solver"


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("g++ is not installed")
    path = tmp_path_factory.mktemp("patch") / "patch.dll"
    subprocess.run(
        ["g++", "-std=c++20", "-O3", "-march=native", "-fopenmp-simd", "-shared",
         "-static", "-pthread", str(SOLVER / "patch.cpp"), "-o", str(path)],
        check=True,
    )
    return str(path)


def _wait_for(condition, seconds=30):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _open_raise(patcher, x=0.8, times=1, node_index=0):
    """The button raises x of the pot at the root, off the tree, `times` times."""
    node = patcher.book.node(node_index)
    pot = patcher.book.small_blind + patcher.book.big_blind
    owed = patcher.book.big_blind - patcher.book.small_blind
    sizes = [
        (a, (node.to[a] - patcher.book.big_blind) / (pot + owed))
        for a in range(node.nact) if node.to[a] > 0
    ]
    bets = [patcher.book.small_blind, patcher.book.big_blind]
    for _ in range(times):
        found = patcher.hit(
            patcher.snapshot(), node_index, node, sizes, x, [], bets, 0,
            patcher.book.big_blind,
        )
    return found, (node_index, size_bin(x))


def _solved(book, library, **options):
    """A patcher that has published one patch, of the root raise."""
    options = {"min_hits": 1, "regret_fraction": 1e9, **options}
    patcher = Patcher(book, library_path=library, **options)
    _, key = _open_raise(patcher)
    _wait_for(lambda: key in patcher.snapshot())
    return patcher, key


def test_hotspot_needs_repeats_and_is_queued_once(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=library, min_hits=5, start=False)
    _open_raise(patcher, times=4)
    assert patcher._queue.empty()
    _open_raise(patcher, times=10)
    assert patcher._queue.qsize() == 1
    spot = patcher._queue.get()
    assert (spot.responder, spot.street, spot.node) == (1, 0, 0)
    assert spot.bets == (20 + round(0.8 * 40), 20)
    # 0.8 lies between the tree's two raises, so the range blends them.
    assert 0 < spot.weight_a < 1 and spot.raise_a != spot.raise_b


def test_a_size_on_the_tree_needs_more_repeats(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=library, min_hits=2, on_tree_hits=6, start=False)
    # The tree's half-pot raise at the root: to 40, a raise of 20 into 40.
    _open_raise(patcher, x=0.5, times=5)
    assert patcher._queue.empty()
    _open_raise(patcher, x=0.5)
    assert patcher._queue.qsize() == 1


def test_an_all_in_has_no_spot(trained):  # noqa: F811
    book, _, _ = trained
    node = book.node(0)
    assert make_spot((0, 0), 99.0, [], 0, node, [(2, 0.5), (4, 12.0)], [10, 20], 0, 20, book.stack) is None


def test_patch_tree_is_legal_in_the_real_game(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher, key = _solved(book, library)
    patch = patcher.snapshot()[key]
    patcher.stop()
    state = GameState([500, 500])
    state.setup_hand([500, 500], 0, 10, 20)
    state.process_action(PlayerAction.raise_to(0, patch.root_bets[0]))
    _walk(patch, 0, state)
    root = patch.node(0)
    assert abs(sum(patch.probabilities(root, 0)) - 1) < 1e-9
    # Finer than the blueprint: fold, call, two sizes and all-in.
    assert (root.player, root.nact, root.to[0], root.to[1]) == (1, 5, -1, 0)


def _job(lib, book, steps, root_bets, seed=1):
    _, abstraction = _map_for_pointer(book.abstraction_path)
    _, strategy = _map_for_pointer(book.strategy_path)
    path = (_Step * len(steps))(*steps)
    root = _Root()
    root.bets[0], root.bets[1] = root_bets
    root.behind, root.last_raise, root.responder, root.street = 0, 40, 1, 0
    job = lib.ps_job_create(
        ctypes.addressof(abstraction), ctypes.addressof(strategy), path, len(steps),
        ctypes.byref(root), seed,
    )
    return job, (abstraction, strategy)


def test_deals_follow_the_line_the_blueprint_would_take(trained, library):  # noqa: F811
    book, _, _ = trained
    lib = load_library(library)
    node = book.node(0)
    action = next(a for a in range(node.nact) if node.to[a] > 0)
    job, keep = _job(lib, book, [_Step(0, action, -1, 1.0)], (60, 20))
    assert job
    deals = 300_000
    seat0, seat1 = (ctypes.c_int * 256)(), (ctypes.c_int * 256)()
    assert lib.ps_job_sample(job, deals, seat0, seat1) == 0
    lib.ps_job_destroy(job)

    combos = [0] * book.buckets[0]
    for a in range(1, 52):
        for b in range(a):
            combos[book.bucket((a, b))] += 1
    weights = [c * max(book.probabilities(node, k)[action], 0.01) for k, c in enumerate(combos)]
    total = sum(weights)
    distance = sum(abs(seat0[k] / deals - w / total) for k, w in enumerate(weights)) / 2
    assert distance < 0.02, "the raiser's hands must follow the blueprint"
    # Nearly untouched: the other seat did nothing on this line. Only blockers shift it.
    unbiased = sum(abs(seat1[k] / deals - c / 1326) for k, c in enumerate(combos)) / 2
    assert unbiased < 0.03


def test_regret_falls_as_it_runs(trained, library):  # noqa: F811
    book, _, _ = trained
    lib = load_library(library)
    node = book.node(0)
    action = next(a for a in range(node.nact) if node.to[a] > 0)
    job, keep = _job(lib, book, [_Step(0, action, -1, 1.0)], (60, 20))
    lib.ps_job_run(job, 5_000)
    early = lib.ps_job_regret(job)
    lib.ps_job_run(job, 195_000)
    late = lib.ps_job_regret(job)
    lib.ps_job_destroy(job)
    assert late < early / 3  # theory: 1 / sqrt(iterations) gives about 6.3 times


def test_nothing_is_published_above_the_regret_threshold(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=library, min_hits=1, regret_fraction=0, max_iterations=4000)
    _, key = _open_raise(patcher)
    events = []
    _wait_for(lambda: events.extend(patcher.status()) or events)
    patcher.stop()
    assert events[0][:2] == ("not converged", key)
    assert key not in patcher.snapshot()


def test_a_bad_spot_does_not_end_the_worker(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=library, min_hits=1, regret_fraction=1e9)
    _, key = _open_raise(patcher)
    _wait_for(lambda: key in patcher.snapshot())
    good = make_spot((1, 1), 0.8, [], 0, book.node(0), [(2, 0.5), (3, 1.0)], [10, 20], 0, 20, book.stack)
    patcher._queue.put(good._replace(key=(2, 2), node=10**8))  # no such node
    patcher._queue.put(good._replace(bets=None))  # blows up in Python
    patcher._queue.put(good)
    _wait_for(lambda: (1, 1) in patcher.snapshot())
    events = [e[0] for e in patcher.status()]
    patcher.stop()
    assert "too big" in events and "failed" in events and (2, 2) not in patcher.snapshot()


def test_a_missing_library_only_disables_patching(trained, tmp_path):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=str(tmp_path / "none.dll"), min_hits=1)
    _open_raise(patcher, times=3)
    events = []
    _wait_for(lambda: events.extend(patcher.status()) or events)
    assert events[0][0] == "disabled"
    assert patcher.snapshot() == {}


def test_published_patches_and_tables_cannot_change(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher, key = _solved(book, library)
    patcher.stop()
    table = patcher.snapshot()
    with pytest.raises(TypeError):
        table[(9, 9)] = None
    with pytest.raises(AttributeError):
        table[key].regret = 0


def test_readers_never_see_a_torn_swap(trained, library):  # noqa: F811
    """Readers race the real publish path, which copies from a live native job."""
    book, _, _ = trained
    lib = load_library(library)
    patcher, key = _solved(book, library)
    patcher.stop()
    node = book.node(0)
    action = next(a for a in range(node.nact) if node.to[a] > 0)
    job, keep = _job(lib, book, [_Step(0, action, -1, 1.0)], (60, 20))
    lib.ps_job_run(job, 1000)
    spot = make_spot(key, 0.8, [], 0, node, [(2, 0.5), (3, 1.0)], [10, 20], 0, 20, book.stack)
    swaps = 400
    done = threading.Event()
    problems = []
    reads = [0, 0, 0, 0]

    def reader(slot):
        try:
            last = 0
            while not done.is_set():
                table = patcher.snapshot()
                held = dict(table)
                assert len(table) >= last, "a table went backwards"
                last = len(table)
                for patch in held.values():
                    root = patch.node(0)
                    assert root.kind == DECISION and root.nact == 5
                    assert abs(sum(patch.probabilities(root, 3)) - 1) < 1e-9
                assert dict(table) == held, "a published table changed under its reader"
                reads[slot] += 1
        except BaseException as error:
            problems.append(error)

    readers = [threading.Thread(target=reader, args=(i,)) for i in range(4)]
    for thread in readers:
        thread.start()
    for i in range(swaps):
        patcher._publish(lib, job, spot._replace(key=(i, i)), 1.0, 1000)
    done.set()
    for thread in readers:
        thread.join()
    lib.ps_job_destroy(job)
    assert not problems, problems
    assert min(reads) > 0
    assert len(patcher.snapshot()) == swaps + 1


def test_a_native_chunk_does_not_stall_the_game_loop(trained, library):  # noqa: F811
    book, _, _ = trained
    # One chunk takes far longer than the allowed stall, so holding the GIL for it
    # would fail this test.
    patcher = Patcher(book, library_path=library, min_hits=1, regret_fraction=0,
                      chunk=2_000_000, max_iterations=2_000_000)
    _, key = _open_raise(patcher)
    worst = 0.0
    last = start = time.perf_counter()
    events = []
    while not events:
        now = time.perf_counter()
        worst = max(worst, now - last)
        last = now
        sum(range(200))  # some Python work, which needs the GIL
        patcher.snapshot()
        events.extend(patcher.status())
    elapsed = time.perf_counter() - start
    patcher.stop()
    assert elapsed > 0.5, "the chunk was too short to prove anything"
    assert worst < 0.1, f"stalled {worst * 1000:.0f} ms in a {elapsed:.1f} s solve"


def test_stop_ends_the_worker_after_the_current_chunk(trained, library):  # noqa: F811
    book, _, _ = trained
    patcher = Patcher(book, library_path=library, min_hits=1, regret_fraction=0,
                      max_iterations=10**12, max_seconds=10**6)
    _open_raise(patcher)
    time.sleep(0.2)
    start = time.perf_counter()
    patcher.stop(timeout=10)
    assert not patcher._thread.is_alive()
    assert time.perf_counter() - start < 2


def _patching_bot(monkeypatch, book, library, **options):
    import solver.bot as bot

    monkeypatch.setattr(bot, "_book", book)
    monkeypatch.setattr(
        bot, "Patcher",
        functools.partial(Patcher, library_path=library, min_hits=4, regret_fraction=1e9, **options),
    )
    return bot


def test_a_swap_never_changes_a_hand_in_progress(trained, library, monkeypatch):  # noqa: F811
    from solver.translation import MAPPINGS, DeviationMatrix, Translator
    from game_state import GameStateView

    book, _, _ = trained
    patcher, key = _solved(book, library)
    patcher.stop()
    published = patcher.snapshot()

    def new_hand(table):
        state = GameState([500, 500])
        state.setup_hand([500, 500], 0, 10, 20)
        patcher._published = table
        translator = Translator(
            book, GameStateView(state), 1, MAPPINGS["threshold"], DeviationMatrix(),
            random.Random(1), None, patcher,
        )
        return state, translator

    raise_to = published[key].root_bets[0]
    state, translator = new_hand({})  # the hand starts before the patch exists
    patcher._published = published  # the swap, before the bet arrives
    action = PlayerAction.raise_to(0, raise_to)
    translator.observe(action)
    assert translator.book is book

    state, translator = new_hand(published)  # the next hand sees it
    translator.observe(action)
    state.process_action(action)
    assert translator.book is published[key]
    assert translator.node == 0
    assert translator.history[-1][0] == 0, "the bet itself stays on the blueprint path"


@pytest.mark.parametrize("resolve", [False, True], ids=["blueprint", "live solver"])
def test_a_bot_plays_on_patches_without_faults(trained, library, monkeypatch, resolve, request):  # noqa: F811
    book, _, _ = trained
    bot = _patching_bot(monkeypatch, book, library)
    if resolve:
        _live_bot(bot, monkeypatch, book, request.getfixturevalue("live_library"))
    ours = bot.SolverBot(patch=True, resolve=resolve, budget_ms=3)
    used = []
    ranges = []
    original = ours.run

    def run(actions, game_state):
        move = original(actions, game_state)
        translator = ours.translator
        if translator.book is not translator.blueprint:
            used.append(translator.book.key)
            board = tuple(range(40, 40 + len(game_state.board)))
            ranges.append(max(blueprint_ranges(book, translator.history, board)[0]))
        return move

    ours.run = run
    deck = full_deck()
    rng = random.Random(3)
    sizer = _Sizer(lambda r: 0.8, rate=0.6)
    for hand in range(400):
        rng.shuffle(deck)
        result = play_hand([ours, sizer], deck, [500, 500], button=hand % 2)
        assert result.faults == {}, result.faults
        assert sum(result.net) == 0
        if len(used) > 100:
            break
        time.sleep(0.002)  # hands take real time, which lets the solves finish
    ours.patcher.stop()
    assert used, f"no patch was ever played: {ours.patcher.status()}"
    assert all(r > 0 for r in ranges)


def test_a_new_bot_can_still_be_pickled(trained, library, monkeypatch):  # noqa: F811
    import solver.bot as bot

    book, _, _ = trained
    monkeypatch.setattr(bot, "_book", book)
    copy = pickle.loads(pickle.dumps(bot.PatchingBot()))
    assert copy.patch and copy.patcher is None
