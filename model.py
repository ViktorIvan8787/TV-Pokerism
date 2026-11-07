"""
Plays the trained blueprint with off-tree bet mapping. Pure Python: the strategy
is plain bytes in this object's pickle, built by make_model.py. See solver/README.md.
"""

import random

from player_action import PlayerAction
from translation import MAPPINGS, DeviationMatrix, Translator, card_number


class Model:
    def __init__(self, book=None, mapping="threshold"):
        self.book = book  # a blueprint.Blueprint; without one the bot only calls
        self.mapping = mapping  # a name in translation.MAPPINGS
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
        try:
            if self.book is None:
                return PlayerAction.call(self.player_id)
            if self.translator is None:
                self.translator = Translator(
                    self.book, game_state, self.player_id,
                    MAPPINGS[self.mapping], self.deviations, self.rng,
                )
            for action in actions:
                self.translator.observe(action)
            board = tuple(card_number(c) for c in game_state.board)
            move = self.translator.decide(game_state, self.hole, board)
            game_state.validate_action(move)  # an illegal move raises here too
            return move
        except Exception:
            if "call" in game_state.legal_actions:  # also checks when nothing is owed
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def end_game(self, actions, game_state):
        pass
