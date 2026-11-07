"""
Reads a trained strategy from its memory-mapped files. Standard library only.

    book = Strategy("solver/out/strategy.bin", "solver/out/abstraction.bin")
    bucket = book.bucket(hole, board)    # hole, board: card numbers
    node = book.node(0)                  # 0 is the root of the betting tree
    probs = book.probabilities(node, bucket)

A card number is rank * 4 + suit, the order of card_utils.full_deck(). Nothing
is loaded up front: the operating system pages in what each lookup touches.
Layouts are written by solver/abstraction.hpp and solver/mccfr.hpp.
"""

import mmap
import struct
from math import comb
from typing import NamedTuple

DECISION, FOLD, SHOWDOWN = 0, 1, 2
_NODE = struct.Struct("<5i5iI4B")
_STRATEGY_HEADER = struct.Struct("<8sQQQ5I4I")
_ABSTRACTION_HEADER = struct.Struct("<8sQ4I3I")
_BINOMIAL = [[comb(n, k) for k in range(6)] for n in range(53)]


class Node(NamedTuple):
    child: tuple  # node index per action
    to: tuple  # per action: -1 fold, 0 call or check, else the raise-to total
    offset: int
    street: int  # 0 preflop .. 3 river
    player: int  # 0 is the button
    nact: int
    kind: int  # DECISION, FOLD or SHOWDOWN


def _map(path):
    with open(path, "rb") as file:
        return mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)


def hole_index(a, b):
    if a < b:
        a, b = b, a
    return a * (a - 1) // 2 + b


class Strategy:
    def __init__(self, strategy_path, abstraction_path):
        self.strategy_path, self.abstraction_path = strategy_path, abstraction_path
        self._strategy = _map(strategy_path)
        self._abstraction = _map(abstraction_path)
        s = _STRATEGY_HEADER.unpack_from(self._strategy)
        a = _ABSTRACTION_HEADER.unpack_from(self._abstraction)
        if s[0] != b"CFRSTR1\0" or a[0] != b"CFRABS1\0":
            raise ValueError("not a strategy file and an abstraction file")
        if s[1] != a[1]:
            raise ValueError("this strategy was trained on another abstraction")
        self.iterations = s[2]
        self.stack, self.small_blind, self.big_blind = s[4:7]
        self.node_count, self.entries = s[7:9]
        self.buckets = s[9:13]
        self._nodes_at = _STRATEGY_HEADER.size
        self._probs_at = self._nodes_at + self.node_count * _NODE.size

        self._preflop_at = _ABSTRACTION_HEADER.size
        self._perms_at = self._preflop_at + 1326
        at = self._perms_at + 24 * 52
        self._street_at = [None] * 4  # (canon offset, table offset), by street
        for street, boards in enumerate(a[6:9], 1):
            sets = comb(52, street + 2)
            self._street_at[street] = (at, at + 4 * sets)
            at += 4 * sets + boards * 1326

    def node(self, index) -> Node:
        v = _NODE.unpack_from(self._strategy, self._nodes_at + index * _NODE.size)
        return Node(v[0:5], v[5:10], v[10], v[11], v[12], v[13], v[14])

    def bucket(self, hole, board=()) -> int:
        """The bucket of two hole cards on a board of 0, 3, 4 or 5 cards."""
        a, b = hole
        data = self._abstraction
        if not board:
            return data[self._preflop_at + hole_index(a, b)]
        if len(board) not in (3, 4, 5):
            raise ValueError(f"a board has 0, 3, 4 or 5 cards, not {len(board)}")
        rank = sum(_BINOMIAL[c][k] for k, c in enumerate(sorted(board), 1))
        canon_at, table_at = self._street_at[len(board) - 2]
        entry = struct.unpack_from("<I", data, canon_at + 4 * rank)[0]
        perm = self._perms_at + (entry & 31) * 52
        return data[
            table_at + (entry >> 5) * 1326
            + hole_index(data[perm + a], data[perm + b])
        ]

    def probabilities(self, node: Node, bucket) -> list:
        """The average strategy at a decision node: one probability per action."""
        at = self._probs_at + 2 * (node.offset + bucket * node.nact)
        raw = struct.unpack_from(f"<{node.nact}H", self._strategy, at)
        total = sum(raw)
        return [x / total for x in raw]
