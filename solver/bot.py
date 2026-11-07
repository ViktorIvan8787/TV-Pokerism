"""
Plays the trained strategy, for measuring it locally:

    .venv/Scripts/python -m evals --model solver.bot:SolverBot

Not a submission as it stands: it needs the files in solver/out, and the arena
accepts only a pickle of at most 50 MB plus .py files (see solver/README.md).
"""

import os
import random

from solver.patcher import Patcher
from solver.reader import Strategy
from solver.subgame import Resolver, Subgame, load_library
from solver.translation import MAPPINGS, DeviationMatrix, Translator, card_number

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
_book = None  # one open mapping per process; mmaps cannot be pickled


def _strategy():
    global _book
    if _book is None:
        _book = Strategy(
            os.path.join(_OUT, "strategy.bin"),
            os.path.join(_OUT, "abstraction.bin"),
        )
    return _book


class SolverBot:
    def __init__(self, mapping="threshold", budget_ms=200, resolve=True, patch=False):
        self.mapping = mapping  # a name in translation.MAPPINGS
        self.budget_ms = budget_ms
        self.resolve = resolve  # solve the turn and river live, if the library loads
        self.subgame = None  # built on first use: holds native memory
        self.patch = patch  # refine repeated spots in the background
        self.patcher = None  # built on first use: holds a thread
        self.hole = ()
        self.player_id = None
        self.translator = None
        self.deviations = DeviationMatrix()  # kept across hands, per team
        self.rng = random.Random()

    def seed(self, hand, player_id):
        self.hole = tuple(card_number(c) for c in hand)
        self.player_id = player_id
        self.translator = None

    def run(self, actions, game_state):
        if self.translator is None:
            self.translator = Translator(
                _strategy(), game_state, self.player_id,
                MAPPINGS[self.mapping], self.deviations, self.rng,
                self._make_resolver if self._subgame_ready() else None,
                self._patcher_ready(),
            )
        for action in actions:
            self.translator.observe(action)
        board = tuple(card_number(c) for c in game_state.board)
        return self.translator.decide(game_state, self.hole, board)

    def _subgame_ready(self):
        if self.resolve and self.subgame is None:
            library = load_library()
            if library is None:
                self.resolve = False  # no compiled library: blueprint only
            else:
                self.subgame = Subgame(library)
        return self.resolve

    def _patcher_ready(self):
        if self.patch and self.patcher is None:
            self.patcher = Patcher(_strategy())
        return self.patcher

    def _make_resolver(self, history, board, hole, seat, start):
        return Resolver(
            self.subgame, _strategy(), history, board, hole, seat, start,
            MAPPINGS[self.mapping], self.rng, self.budget_ms,
        )


class PatchingBot(SolverBot):
    """For evals --model, which builds a model with no arguments."""

    def __init__(self):
        super().__init__(patch=True)
