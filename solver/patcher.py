"""
Refines the blueprint in the background while the bot plays, without blocking it.

    game loop                         worker thread
    ---------                         -------------
    hit(): count a bet, O(1); if hot,
    SimpleQueue.put a frozen spot ->  get a spot; solve it in native chunks with
                                      the GIL released; check regret between them
    snapshot(): read one reference <- build an immutable Patch, then publish a
    at the start of each hand         new table by rebinding one reference

Why there are no data races:
    * Only the game loop touches the hit counters and the "submitted" set.
    * Only the worker touches a native job, and it copies the result out of it.
    * Spots, patches and published tables are never changed after they are shared.
    * Rebinding `_published` is one atomic reference store. A hand that took the
      old table keeps it alive by reference, so nothing is freed under a reader.
    * A hand reads the table once, so no hand changes strategy partway through.
"""

import ctypes
import math
import mmap
import os
import queue
import struct
import threading
import time
from types import MappingProxyType
from typing import NamedTuple

from solver.reader import _NODE, Node

_LIBRARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build", "patch.dll")
_BIN_RATIO = 1.15  # a size bin spans 15%: close sizes share one patch


class _Step(ctypes.Structure):
    _fields_ = [
        ("node", ctypes.c_int), ("action", ctypes.c_int),
        ("action2", ctypes.c_int), ("weight", ctypes.c_float),
    ]


class _Root(ctypes.Structure):
    _fields_ = [
        ("bets", ctypes.c_int * 2), ("behind", ctypes.c_int),
        ("last_raise", ctypes.c_int), ("responder", ctypes.c_int),
        ("street", ctypes.c_int),
    ]


def load_library(path=_LIBRARY):
    """The compiled library, or None if it is missing or will not load."""
    try:
        lib = ctypes.CDLL(path)
    except OSError:
        return None
    job = ctypes.c_void_p
    lib.ps_job_create.restype = job
    lib.ps_job_create.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(_Step), ctypes.c_int,
        ctypes.POINTER(_Root), ctypes.c_ulonglong,
    ]
    lib.ps_job_destroy.argtypes = [job]
    lib.ps_job_run.argtypes = [job, ctypes.c_ulonglong]
    lib.ps_job_regret.restype = ctypes.c_double
    lib.ps_job_regret.argtypes = [job]
    for name in ("ps_job_level", "ps_job_nodes", "ps_job_entries"):
        getattr(lib, name).argtypes = [job]
    lib.ps_job_export.argtypes = [job, ctypes.c_void_p]
    int_pointer = ctypes.POINTER(ctypes.c_int)
    lib.ps_job_sample.argtypes = [job, ctypes.c_int, int_pointer, int_pointer]
    return lib


def size_bin(x):
    """Pot fractions on a logarithmic scale: small sizes get narrower bins."""
    return round(math.log1p(max(x, 0.0)) / math.log(_BIN_RATIO))


class Spot(NamedTuple):
    """One coarse spot, frozen when submitted: what the worker needs to solve it."""

    key: tuple  # (blueprint node index, size bin)
    x: float  # the mean size seen, as a fraction of the pot after calling
    path: tuple  # ((node index, action), ...): the line to reach the node
    node: int  # the node where the opponent bet
    raise_a: int  # the blueprint raises nearest the bet, for its estimated range
    raise_b: int  # -1 if there is only one
    weight_a: float
    bets: tuple  # the abstract state after the bet
    behind: int
    last_raise: int
    responder: int
    street: int


def make_spot(key, x, history, node_index, node, sizes, bets, behind, last_raise, stack):
    """The spot after the opponent bets x at this abstract node, or None if that
    is really an all-in, which has no finer answer than call or fold."""
    current = max(bets)
    owed = current - bets[node.player]
    total = max(current + round(x * (2 * behind + sum(bets) + owed)), current + last_raise)
    if total >= stack - behind:
        return None
    after = list(bets)
    after[node.player] = total
    # Their range for this size: a blend of the blueprint's two nearest raises.
    below = [s for s in sizes if s[1] <= x]
    above = [s for s in sizes if s[1] >= x]
    if below and above and below[-1][0] != above[0][0]:
        (a, low), (b, high) = below[-1], above[0]
        weight = (high - x) / (high - low)
    else:
        a, b, weight = (below[-1] if below else above[0])[0], -1, 1.0
    return Spot(
        key, x, tuple(history), node_index, a, b, weight, tuple(after), behind,
        max(total - current, last_raise), 1 - node.player, node.street,
    )


class Patch:
    """A solved sub-tree. Immutable; quacks like reader.Strategy for Translator."""

    __slots__ = ("base", "key", "regret", "iterations", "root_bets", "root_behind",
                 "_data", "_probs_at")

    def __init__(self, base, spot, data, node_count, regret, iterations):
        set_field = object.__setattr__
        set_field(self, "base", base)
        set_field(self, "key", spot.key)
        set_field(self, "regret", regret)
        set_field(self, "iterations", iterations)
        set_field(self, "root_bets", spot.bets)
        set_field(self, "root_behind", spot.behind)
        set_field(self, "_data", bytes(data))
        set_field(self, "_probs_at", node_count * _NODE.size)

    def __setattr__(self, name, value):
        raise AttributeError("a published patch cannot change")

    @property
    def stack(self):
        return self.base.stack

    @property
    def buckets(self):
        return self.base.buckets

    def bucket(self, hole, board=()):
        return self.base.bucket(hole, board)

    def node(self, index) -> Node:
        v = _NODE.unpack_from(self._data, index * _NODE.size)
        return Node(v[0:5], v[5:10], v[10], v[11], v[12], v[13], v[14])

    def probabilities(self, node, bucket):
        at = self._probs_at + 2 * (node.offset + bucket * node.nact)
        raw = struct.unpack_from(f"<{node.nact}H", self._data, at)
        total = sum(raw)
        return [x / total for x in raw]


def _map_for_pointer(path):
    """A read mapping ctypes can point at. ACCESS_COPY only copies a page on a
    write, and nothing writes, so this shares the page cache with the reader."""
    with open(path, "rb") as file:
        mapped = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY)
    return mapped, (ctypes.c_char * len(mapped)).from_buffer(mapped)


class Patcher:
    def __init__(self, book, library_path=_LIBRARY, min_hits=8, on_tree_hits=40,
                 min_gap=0.05, regret_fraction=0.15, chunk=2000,
                 max_seconds=300.0, max_iterations=20_000_000, max_patches=64,
                 start=True):
        """min_hits: times an off-tree size must recur before it is solved, at
        least min_gap of the pot away from every blueprint size.
        on_tree_hits: the same for sizes the blueprint has, which are less coarse.
        regret_fraction: publish when regret is below this share of the pot.
        chunk: iterations between checks of regret and of stop().
        A spot that misses its limits is dropped, and never retried."""
        self.book = book
        self.min_hits, self.on_tree_hits, self.min_gap = min_hits, on_tree_hits, min_gap
        self.regret_fraction = regret_fraction
        self.chunk = chunk
        self.max_seconds, self.max_iterations = max_seconds, max_iterations
        self.max_patches = max_patches
        self._library_path = library_path
        self._hits = {}  # key -> [count, total size]; game loop only
        self._submitted = set()  # game loop only
        self._queue = queue.SimpleQueue()
        self._published = MappingProxyType({})  # read-only: replaced whole by the worker
        self._stop = threading.Event()
        self.log = queue.SimpleQueue()  # (event, key, detail), for status()
        self._thread = threading.Thread(target=self._work, name="patcher", daemon=True)
        if start:
            self._thread.start()

    # -- game loop: never blocks ---------------------------------------------

    def snapshot(self):
        """The current patches. Keep it for a whole hand."""
        return self._published

    def hit(self, table, node_index, node, sizes, x, history, bets, behind, last_raise):
        """The opponent bet x at a blueprint node: the patch for it in this hand's
        table, or None after counting it and queueing a solve if it is now hot."""
        key = (node_index, size_bin(x))
        patch = table.get(key)
        if patch is not None:
            return patch
        if key in self._submitted:
            return None
        hit = self._hits.get(key)
        if hit is None:
            hit = self._hits[key] = [0, 0.0]
        hit[0] += 1
        hit[1] += x
        gap = min(abs(x - size) for _, size in sizes)
        needed = self.min_hits if gap >= self.min_gap else self.on_tree_hits
        if hit[0] >= needed and len(self._submitted) < self.max_patches:
            self._submitted.add(key)
            spot = make_spot(key, hit[1] / hit[0], history, node_index, node, sizes,
                             bets, behind, last_raise, self.book.stack)
            if spot is not None:
                self._queue.put(spot)  # SimpleQueue: unbounded, never waits
        return None

    # -- worker --------------------------------------------------------------

    def _work(self):
        lib = load_library(self._library_path)
        if lib is None:
            self.log.put(("disabled", None, "the patch library did not load"))
            return
        try:
            _, abstraction = _map_for_pointer(self.book.abstraction_path)
            _, strategy = _map_for_pointer(self.book.strategy_path)
        except (OSError, ValueError) as error:
            self.log.put(("disabled", None, str(error)))
            return
        while True:
            spot = self._queue.get()
            if spot is None or self._stop.is_set():
                return
            try:
                self._solve(lib, abstraction, strategy, spot)
            except Exception as error:  # one bad spot must not end the worker
                self.log.put(("failed", spot.key, repr(error)))

    def _solve(self, lib, abstraction, strategy, spot):
        path = [_Step(node, action, -1, 1.0) for node, action in spot.path]
        path.append(_Step(spot.node, spot.raise_a, spot.raise_b, spot.weight_a))
        steps = (_Step * len(path))(*path)
        root = _Root()
        root.bets[0], root.bets[1] = spot.bets
        root.behind = spot.behind
        root.last_raise = spot.last_raise
        root.responder = spot.responder
        root.street = spot.street
        job = lib.ps_job_create(
            ctypes.addressof(abstraction), ctypes.addressof(strategy), steps,
            len(path), ctypes.byref(root), hash(spot.key) & 0xFFFFFFFF,
        )
        if not job:
            self.log.put(("too big", spot.key, ""))
            return
        try:
            pot = 2 * spot.behind + sum(spot.bets)
            target = self.regret_fraction * pot
            deadline = time.monotonic() + self.max_seconds
            iterations = 0
            next_check = 0
            while not self._stop.is_set():
                if lib.ps_job_run(job, self.chunk) != 0:
                    self.log.put(("unreachable", spot.key, ""))
                    return
                iterations += self.chunk
                over_limit = iterations >= self.max_iterations or time.monotonic() > deadline
                if iterations < next_check and not over_limit:
                    continue
                # Scanning every table is slow, so the checks thin out as it runs.
                next_check = iterations * 5 // 4
                regret = lib.ps_job_regret(job)
                if regret <= target:
                    self._publish(lib, job, spot, regret, iterations)
                    return
                if over_limit:
                    self.log.put(("not converged", spot.key, f"{regret:.1f} > {target:.1f}"))
                    return
        finally:
            lib.ps_job_destroy(job)

    def _publish(self, lib, job, spot, regret, iterations):
        nodes = lib.ps_job_nodes(job)
        size = nodes * _NODE.size + 2 * lib.ps_job_entries(job)
        buffer = ctypes.create_string_buffer(size)
        lib.ps_job_export(job, buffer)
        patch = Patch(self.book, spot, buffer.raw, nodes, regret, iterations)
        table = dict(self._published)
        table[spot.key] = patch
        self._published = MappingProxyType(table)  # the swap: one atomic reference store
        self.log.put(("published", spot.key, f"regret {regret:.2f}, {iterations} iterations"))

    def stop(self, timeout=5.0):
        """Ends the worker. The current native chunk finishes first."""
        self._stop.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout)

    def status(self):
        events = []
        while True:
            try:
                events.append(self.log.get_nowait())
            except queue.Empty:
                return events
