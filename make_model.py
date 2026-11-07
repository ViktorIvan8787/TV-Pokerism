"""
Builds the submission with the blueprint inside the pickle. Not shipped: nothing
that plays imports it, so its file reads are not part of the submission.

    python -m submit --team NAME --model make_model:build

Plain `python -m submit --team NAME` pickles an EMPTY Model that only calls.
"""

import struct
from math import comb

from blueprint import Blueprint
from model import Model

OUT = "solver/out/"


def build():
    with open(OUT + "strategy.bin", "rb") as file:
        strategy = file.read()
    with open(OUT + "abstraction.bin", "rb") as file:
        header = file.read(44)
        preflop = file.read(1326)
        perms = file.read(24 * 52)
        _, _, *counts = struct.unpack("<8sQ4I3I", header)
        boards = counts[4:]
        canon, tables = [], []
        for street in range(2):  # flop and turn; the river is scored during play
            canon.append(file.read(4 * comb(52, street + 3)))
            tables.append(file.read(boards[street] * 1326))
    with open(OUT + "river_table.bin", "rb") as file:
        river = file.read()
    layout = struct.Struct("<8sQQQ5I4I")
    fields = layout.unpack_from(strategy)
    stack, small_blind, big_blind, node_count = fields[4:8]
    nodes_end = layout.size + node_count * 48
    book = Blueprint(
        stack, small_blind, big_blind, fields[9:13], strategy[layout.size:nodes_end],
        strategy[nodes_end:], preflop, perms, canon, tables, river,
    )
    return Model(book)
