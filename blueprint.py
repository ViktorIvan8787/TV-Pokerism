"""
The trained strategy held as plain bytes, so it can live in the pickle: the same
lookups as solver/reader.py, with no files, mmap or struct. Layouts are written by
solver/abstraction.hpp and solver/mccfr.hpp.

The river's 170 MB bucket table does not fit, so river hands are scored here and
looked up in a small table (solver/river_table.cpp). Built by make_model.py.
"""

from math import comb
from typing import NamedTuple

_BINOMIAL = [[comb(n, k) for k in range(6)] for n in range(53)]
_HOLE_COUNT = 1326
_NODE_SIZE = 48


class Node(NamedTuple):
    child: tuple  # node index per action
    to: tuple  # per action: -1 fold, 0 call or check, else the raise-to total
    offset: int
    street: int  # 0 preflop .. 3 river
    player: int  # 0 is the button
    nact: int
    kind: int  # 0 decision, 1 fold, 2 showdown


def hole_index(a, b):
    if a < b:
        a, b = b, a
    return a * (a - 1) // 2 + b


def _straight_high(mask):
    """The top rank (0-12) of the best straight in a bitmask of ranks, or -1."""
    for high in range(12, 3, -1):
        if mask >> (high - 4) & 31 == 31:
            return high
    return 3 if mask & 0b1000000001111 == 0b1000000001111 else -1  # the wheel


def rank7(cards):
    """A number that orders 7-card hands as card_utils.evaluate does, far faster.
    A card is rank * 4 + suit."""
    counts = [0] * 13
    suits = [0, 0, 0, 0]
    for card in cards:
        counts[card >> 2] += 1
        suits[card & 3] |= 1 << (card >> 2)
    flush = 0
    for mask in suits:
        if mask.bit_count() >= 5:
            flush = mask
    if flush:
        high = _straight_high(flush)
        if high >= 0:
            return 8 << 20 | high
    ranks = [r for r in range(12, -1, -1) if counts[r]]
    quads = [r for r in ranks if counts[r] == 4]
    trips = [r for r in ranks if counts[r] == 3]
    pairs = [r for r in ranks if counts[r] == 2]
    if quads:
        kicker = max(r for r in ranks if r != quads[0])
        return 7 << 20 | quads[0] << 4 | kicker
    if trips and (len(trips) > 1 or pairs):
        low = max(trips[1:] + pairs)
        return 6 << 20 | trips[0] << 4 | low
    if flush:
        top = [r for r in range(12, -1, -1) if flush >> r & 1][:5]
    else:
        high = _straight_high(sum(1 << r for r in ranks))
        if high >= 0:
            return 4 << 20 | high
        if trips:
            top, category = [trips[0]] + [r for r in ranks if r != trips[0]][:2], 3
        elif len(pairs) >= 2:
            kicker = max(r for r in ranks if r not in pairs[:2])
            top, category = pairs[:2] + [kicker], 2
        elif pairs:
            top, category = [pairs[0]] + [r for r in ranks if r != pairs[0]][:3], 1
        else:
            top, category = ranks[:5], 0
        value = category
        for r in top:
            value = value << 4 | r
        return value << 4 * (5 - len(top))
    value = 5
    for r in top:
        value = value << 4 | r
    return value


_ALL_HOLES = [(a, b) for a in range(1, 52) for b in range(a)]


def river_score(hole, board):
    """2 * wins + ties against all 990 opposing hands: EHS * 1980."""
    board = list(board)
    mine = rank7(list(hole) + board)
    blocked = set(hole) | set(board)
    score = 0
    for a, b in _ALL_HOLES:
        if a in blocked or b in blocked:
            continue
        theirs = rank7([a, b] + board)
        score += 2 if mine > theirs else 1 if mine == theirs else 0
    return score


class Blueprint:
    def __init__(self, stack, small_blind, big_blind, buckets, nodes, probabilities,
                 preflop, perms, canon, tables, river):
        """nodes, probabilities: the strategy file's sections. canon, tables: the
        abstraction file's sections for the flop and turn. All bytes."""
        self.stack, self.small_blind, self.big_blind = stack, small_blind, big_blind
        self.buckets = tuple(buckets)
        self.nodes, self.probs = nodes, probabilities
        self.preflop, self.perms = preflop, perms
        self.canon, self.tables = canon, tables
        self.river = river
        self.river_memo = (None, 0)  # a hand asks about one river more than once

    def node(self, index):
        at = index * _NODE_SIZE
        data = self.nodes
        v = [int.from_bytes(data[at + 4 * i:at + 4 * i + 4], "little", signed=True) for i in range(10)]
        offset = int.from_bytes(data[at + 40:at + 44], "little")
        return Node(tuple(v[:5]), tuple(v[5:]), offset, data[at + 44], data[at + 45], data[at + 46], data[at + 47])

    def bucket(self, hole, board=()):
        a, b = hole
        if not board:
            return self.preflop[hole_index(a, b)]
        if len(board) == 5:
            key = (hole, tuple(board))
            if self.river_memo[0] != key:
                self.river_memo = (key, self.river[river_score(hole, board)])
            return self.river_memo[1]
        if len(board) not in (3, 4):
            raise ValueError(f"a board has 0, 3, 4 or 5 cards, not {len(board)}")
        rank = sum(_BINOMIAL[c][k] for k, c in enumerate(sorted(board), 1))
        street = len(board) - 3
        entry = int.from_bytes(self.canon[street][4 * rank:4 * rank + 4], "little")
        perm = (entry & 31) * 52
        hole_at = hole_index(self.perms[perm + a], self.perms[perm + b])
        return self.tables[street][(entry >> 5) * _HOLE_COUNT + hole_at]

    def probabilities(self, node, bucket):
        at = 2 * (node.offset + bucket * node.nact)
        raw = [int.from_bytes(self.probs[at + 2 * a:at + 2 * a + 2], "little") for a in range(node.nact)]
        total = sum(raw)
        return [x / total for x in raw]
