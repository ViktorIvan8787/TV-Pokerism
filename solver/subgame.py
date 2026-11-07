"""
Python side of the real-time subgame solver (solver/subgame.cpp), through ctypes.

    Subgame:  the library: solve one exact state, then read and walk the solution
    Resolver: one hand from the turn on: tracks both ranges and re-solves at each
              of our decisions, replacing the blueprint for the rest of the hand

Seat 0 is the button. A range is 1326 weights indexed by reader.hole_index.
"""

import ctypes
import os

from player_action import PlayerAction
from solver.reader import hole_index
from solver.translation import SubgameError, choose_anchor

HOLES = 1326
PLAYER, FOLD, SHOWDOWN, CHANCE = 0, 1, 2, 3
_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "subgame.dll")
_STREET_CARDS = {0: 0, 1: 3, 2: 4, 3: 5}


class _Input(ctypes.Structure):
    _fields_ = [
        ("board", ctypes.c_int * 5), ("board_count", ctypes.c_int),
        ("invested", ctypes.c_int * 2), ("stacks", ctypes.c_int * 2),
        ("bets", ctypes.c_int * 2), ("acted", ctypes.c_int * 2),
        ("last_raise", ctypes.c_int), ("big_blind", ctypes.c_int),
        ("to_act", ctypes.c_int), ("can_raise", ctypes.c_int),
        ("budget_us", ctypes.c_int), ("max_table_mb", ctypes.c_int),
        ("prune", ctypes.c_float),
    ]


class Result(ctypes.Structure):
    _fields_ = [
        ("iterations", ctypes.c_int), ("ladder_level", ctypes.c_int),
        ("nodes", ctypes.c_int), ("hands", ctypes.c_int * 2),
        ("exploitability", ctypes.c_double), ("value", ctypes.c_double * 2),
        ("elapsed_ms", ctypes.c_double), ("table_mb", ctypes.c_double),
        ("setup_ms", ctypes.c_double),
        ("loop_news", ctypes.c_longlong),
    ]


_ERRORS = {
    -1: "bad state", -2: "bad range", -3: "no possible pair of hands",
    -4: "no tree fits", -5: "out of memory", -6: "C++ exception",
}
_FloatRange = ctypes.c_float * HOLES


def load_library(path=_LIBRARY):
    """The compiled library, or None if it is missing or will not load."""
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    lib.ss_create.restype = ctypes.c_void_p
    lib.ss_create.argtypes = [ctypes.c_int]
    lib.ss_destroy.argtypes = [ctypes.c_void_p]
    lib.ss_reserve.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ss_solve.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Input), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Result),
    ]
    int_pointer = ctypes.POINTER(ctypes.c_int)
    lib.ss_node.argtypes = [ctypes.c_void_p] + [int_pointer] * 4
    lib.ss_advance.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ss_deal.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.ss_strategy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    return lib


class Subgame:
    """One solver and its reusable memory: keep it across solves."""

    def __init__(self, lib, threads=None, reserve_mb=48):
        self.lib = lib
        # Half the logical CPUs: on this 4P+4E laptop, 4 spinning threads run
        # about 1.7x as many iterations as 8. Measured, so re-measure elsewhere.
        self.handle = lib.ss_create(threads or max(1, (os.cpu_count() or 2) // 2))
        if not self.handle or lib.ss_reserve(self.handle, reserve_mb) != 0:
            raise SubgameError("could not create the solver")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.ss_destroy(self.handle)
            self.handle = None

    def solve(self, board, invested, stacks, bets, acted, last_raise, big_blind,
              to_act, can_raise, range0, range1, budget_ms=200, max_table_mb=None,
              prune=1e-3) -> Result:
        # Each MB of tables costs about 0.35 ms per iteration on 8 threads, so
        # this keeps roughly 50 iterations in the budget.
        if max_table_mb is None:
            max_table_mb = max(1, int(budget_ms / 16))
        state = _Input()
        state.board_count = len(board)
        for i, card in enumerate(board):
            state.board[i] = card
        for field, values in (("invested", invested), ("stacks", stacks),
                              ("bets", bets), ("acted", acted)):
            target = getattr(state, field)
            target[0], target[1] = int(values[0]), int(values[1])
        state.last_raise = last_raise
        state.big_blind = big_blind
        state.to_act = to_act
        state.can_raise = int(can_raise)
        state.budget_us = int(budget_ms * 1000)
        state.max_table_mb = max_table_mb
        state.prune = prune
        result = Result()
        status = self.lib.ss_solve(
            self.handle, ctypes.byref(state), _FloatRange(*range0),
            _FloatRange(*range1), ctypes.byref(result),
        )
        if status != 0:
            raise SubgameError(_ERRORS.get(status, f"status {status}"))
        return result

    def node(self):
        """(kind, player, to) at the cursor; to has one entry per action."""
        kind, player, nact = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        to = (ctypes.c_int * 5)()
        self.lib.ss_node(self.handle, ctypes.byref(kind), ctypes.byref(player),
                         ctypes.byref(nact), to)
        return kind.value, player.value, tuple(to[: nact.value])

    def advance(self, action):
        if self.lib.ss_advance(self.handle, action) != 0:
            raise SubgameError("no such action here")

    def deal(self, card):
        if self.lib.ss_deal(self.handle, card) != 0:
            raise SubgameError("not a chance node, or that card cannot come")

    def strategy(self, nact):
        """The solved strategy at the cursor: [action * 1326 + hole]."""
        out = (ctypes.c_float * (nact * HOLES))()
        if self.lib.ss_strategy(self.handle, out) != 0:
            raise SubgameError("not a decision node")
        return out


def blueprint_ranges(book, history, board):
    """Both seats' ranges from what the blueprint would have done on this path.

    history: (node index, action) for each decision, or (node index, action,
    patch) for one made in a patch's tree. A hand's reach is the product of the
    probability of each action its seat took.
    """
    buckets = {}  # street -> bucket per hole
    probabilities = {}  # (tree, node index, bucket) -> probability of the action
    blocked = set(board)
    usable = [
        h for h in _ALL_HOLES
        if h[0] not in blocked and h[1] not in blocked
    ]
    reach = [[0.0] * HOLES, [0.0] * HOLES]
    for hole in usable:
        index = hole_index(*hole)
        reach[0][index] = reach[1][index] = 1.0
    for step in history:
        node_index, action = step[:2]
        tree = step[2] if len(step) == 3 else book
        node = tree.node(node_index)
        cards = _STREET_CARDS[node.street]
        row = buckets.get(node.street)
        if row is None:
            row = {h: book.bucket(h, board[:cards]) for h in usable}
            buckets[node.street] = row
        seat = reach[node.player]
        for hole in usable:
            bucket = row[hole]
            key = (id(tree), node_index, bucket)
            p = probabilities.get(key)
            if p is None:
                p = tree.probabilities(node, bucket)[action]
                probabilities[key] = p
            seat[hole_index(*hole)] *= p
    return reach


_ALL_HOLES = [(a, b) for a in range(1, 52) for b in range(a)]


class Resolver:
    """Replaces the blueprint from the first turn or river decision on."""

    def __init__(self, subgame, book, history, board, hole, seat, start, mapping,
                 rng, budget_ms=200):
        self.subgame = subgame
        self.hole = hole
        self.start = start  # chips per player id when the hand began
        self.seat = seat
        self.mapping = mapping
        self.rng = rng
        self.budget_ms = budget_ms
        self.reach = blueprint_ranges(book, history, tuple(board))
        self.solved = False  # whether the cursor is on the current state
        self.last = None

    def _multiply(self, seat, strategy, action):
        reach = self.reach[seat]
        start = action * HOLES
        for h in range(HOLES):
            reach[h] *= strategy[start + h]

    def _ready_ranges(self):
        """Ranges safe to solve: our real hand is present, nobody's is empty."""
        ours = self.reach[self.seat]
        theirs = self.reach[1 - self.seat]
        # The real cards can be a hand the blueprint said we would never play.
        ours[hole_index(*self.hole)] = max(ours[hole_index(*self.hole)], max(ours) * 1e-2, 1e-6)
        if max(theirs) <= 1e-9:
            # They did something our model called impossible: assume nothing.
            theirs[:] = [1.0] * HOLES

    def observe_opponent(self, action, real, x_of):
        """Narrows their range by what the solution says they do; a cursor that
        cannot follow leaves it unchanged. Returns True if a raise was on-tree."""
        if not self.solved:
            return False
        kind, player, to = self.subgame.node()
        if kind != PLAYER or player == self.seat:
            self.solved = False
            return False
        if action.kind == "call":
            choice = to.index(0)
            on_tree = False
        else:
            choice, on_tree = self._map_raise(to, action.amount, real, x_of)
            if choice is None:
                self.solved = False
                return False
        self._multiply(player, self.subgame.strategy(len(to)), choice)
        self.subgame.advance(choice)
        return on_tree

    def _map_raise(self, to, amount, real, x_of):
        raises = [a for a, total in enumerate(to) if total > 0]
        if not raises:
            return None, False
        if amount in (to[a] for a in raises):
            return to.index(amount), True
        if amount == real.max_raise_to:
            return raises[-1], True
        passive = to.index(0)
        actions = [passive] + raises
        anchors = [0.0] + [x_of(to[a]) for a in raises]
        return actions[choose_anchor(x_of(amount), anchors, self.mapping, self.rng)], False

    def observe_card(self, card):
        if self.solved and self.subgame.node()[0] == CHANCE:
            self.subgame.deal(card)
        elif self.solved:
            self.solved = False

    def decide(self, real, me, board, button):
        """Solve this exact state and sample our move. Raises SubgameError if it
        cannot be solved; the caller falls back."""
        self._ready_ranges()
        seats = [button, 1 - button]
        result = self.subgame.solve(
            board,
            [self.start[p] - real.stacks[p] for p in seats],
            [real.stacks[p] for p in seats],
            [real.bets[p] for p in seats],
            [real.acted[p] for p in seats],
            real.last_raise, real.big_blind, self.seat, real.can_raise,
            self.reach[0], self.reach[1], self.budget_ms,
        )
        self.last = result
        self.solved = True
        _, _, to = self.subgame.node()
        strategy = self.subgame.strategy(len(to))
        index = hole_index(*self.hole)
        probs = [strategy[a * HOLES + index] for a in range(len(to))]
        action = self.rng.choices(range(len(to)), probs)[0]
        total = to[action]
        self._multiply(self.seat, strategy, action)
        self.subgame.advance(action)
        if total == -1:
            return PlayerAction.fold(me)
        if total == 0:
            return PlayerAction.call(me)
        return PlayerAction.raise_to(me, total)
