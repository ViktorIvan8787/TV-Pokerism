"""
Sits between the live engine and the fixed blueprint: interprets opponents' bets
that the abstract tree does not have, and keeps two states in step.

    real state      a GameState replayed from the actions, so the rules and
                    chip counts are the engine's own
    abstract state  a node of the blueprint's tree, with its own bets and pot

Sizes are fractions of the pot after calling, the way tree.hpp builds them:
    x = (raise_to - current_bet) / (pot + call_amount)

A mapping function returns P(interpret x as the smaller size a), for a <= x <= b
(Ganzfried & Sandholm 2013). The check/call counts as size 0, so a tiny bet can
be read as no bet instead of as half the pot.

The two states can drift after a mapped bet, in three ways, all handled here:
    * Pot size: our raises are stored as fractions and rebuilt on the real pot.
    * Timing: a raise read as a check or call can end the abstract street while
      we still owe chips. We pay them to catch up (_catch_up).
    * Leaving the tree: a bet read as all-in can end the abstract hand while the
      real one goes on. We then play by hand strength and price (_off_tree).
"""

import math

from game_state import GameState
from player_action import DEALER, PlayerAction

DECISION = 0  # a node's kind: see tree.hpp
STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
ON_TREE_TOLERANCE = 0.02  # fraction of the pot; covers chip rounding


class SubgameError(RuntimeError):
    """The live solver could not solve this spot; the blueprint takes over."""


def card_number(card):
    return (card.value - 2) * 4 + "CDHS".index(card.suit)


# -- mapping functions ----------------------------------------------------------


def pseudo_harmonic(x, a, b):
    return (b - x) * (1 + a) / ((b - a) * (1 + x))


def randomized_geometric(x, a, b):
    """Even odds at the geometric mean of a and b, sqrt(a * b).

    This keeps the ratios between sizes, not the average chips: only a
    randomized arithmetic mapping does that, and it is far more exploitable.
    """
    if a <= 0:
        # The formula is 0 at a = 0, which would read a 1-chip bet as size b.
        return pseudo_harmonic(x, a, b)
    return a * (b - x) / (a * (b - x) + x * (x - a))


def threshold_pseudo_harmonic(x, a, b, snap=0.8):
    """Pseudo-harmonic, but sizes clearly nearer one neighbour go to it outright;
    only those close to the boundary between them stay randomized."""
    p = pseudo_harmonic(x, a, b)
    return 1.0 if p >= snap else 0.0 if p <= 1 - snap else p


MAPPINGS = {
    "geometric": randomized_geometric,
    "threshold": threshold_pseudo_harmonic,
}


def choose_anchor(x, anchors, mapping, rng):
    """An index into anchors (ascending sizes): the two around x, picked by
    mapping, or the end one when x is outside them."""
    if x <= anchors[0]:
        return 0
    for i in range(len(anchors) - 1):
        if x <= anchors[i + 1]:
            p = mapping(x, anchors[i], anchors[i + 1])
            return i if rng.random() < p else i + 1
    return len(anchors) - 1


# -- deviation matrix -----------------------------------------------------------


def wilson_lower(hits, total, z=1.96):
    """A cautious estimate of a rate: stops a few hands raising a flag."""
    if total == 0:
        return 0.0
    p = hits / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (centre - margin) / (1 + z * z / total)


class DeviationMatrix:
    """Per team: raises counted by street (rows) and how they fit the tree
    (columns). Only raises are counted; folds and calls cannot leave the tree."""

    STREET_NAMES = ("preflop", "flop", "turn", "river")
    COLUMNS = ("on_tree", "<0.25", "0.25-0.5", "0.5-1", "1-2", ">2", "no_node")
    ON_TREE, NO_NODE = 0, 6
    _EDGES = (0.25, 0.5, 1, 2)

    def __init__(self):
        self.counts = {}  # team -> [street][column], plain data so it pickles

    def record(self, team, street, column):
        rows = self.counts.setdefault(
            team, [[0] * len(self.COLUMNS) for _ in self.STREET_NAMES]
        )
        rows[street][column] += 1

    @classmethod
    def size_column(cls, x):
        return 1 + sum(x >= edge for edge in cls._EDGES)

    def flags(self, team, min_raises=30, off_tree_rate=0.2, erratic_entropy=0.7):
        """One dict per suspicious street:
        frequent: the off-tree share of raises is confidently over off_tree_rate;
        erratic: sizes are spread so evenly that they are hard to predict
                 (Shannon entropy over the columns, 0 fixed to 1 uniform)."""
        found = []
        for street, row in enumerate(self.counts.get(team, ())):
            total = sum(row)
            if total < min_raises:
                continue
            off = total - row[self.ON_TREE]
            entropy = -sum(n / total * math.log(n / total) for n in row if n)
            entropy /= math.log(len(row))
            kinds = []
            if wilson_lower(off, total) > off_tree_rate:
                kinds.append("frequent")
            if entropy >= erratic_entropy:
                kinds.append("erratic")
            if kinds:
                found.append({
                    "street": self.STREET_NAMES[street],
                    "kinds": kinds,
                    "off_tree": off / total,
                    "entropy": entropy,
                    "raises": total,
                })
        return found

    def report(self):
        lines = []
        for team in sorted(self.counts):
            lines.append(f"{team}: " + " ".join(f"{c:>8}" for c in self.COLUMNS))
            for name, row in zip(self.STREET_NAMES, self.counts[team]):
                lines.append(f"  {name:<7} " + " ".join(f"{n:>8}" for n in row))
            for flag in self.flags(team):
                lines.append(
                    f"  FLAG {flag['street']}: {'+'.join(flag['kinds'])} "
                    f"(off-tree {flag['off_tree']:.0%}, entropy "
                    f"{flag['entropy']:.2f}, {flag['raises']} raises)"
                )
        return "\n".join(lines)


# -- one hand -------------------------------------------------------------------


class Translator:
    """Tracks one hand. observe() every action in order, then decide() our move.
    Neither raises on any legal hand: a drift ends in _off_tree, not an error."""

    def __init__(self, book, view, me, mapping, matrix, rng, make_resolver=None, patcher=None):
        self.book = book  # the blueprint, or the patch this hand has moved to
        self.blueprint = book
        self.patcher = patcher
        # One read per hand: a swap can never change the strategy mid-hand.
        self.patches = patcher.snapshot() if patcher is not None else {}
        self.me = me
        self.mapping = mapping
        self.matrix = matrix
        self.rng = rng
        self.team = view.teams[1 - me] if view.num_players == 2 else None
        self.seat = 0 if view.button == me else 1
        self.catch_up = False
        self.history = []  # (node index, action): the blueprint path, for ranges
        self.make_resolver = make_resolver  # builds a subgame.Resolver, or None
        self.resolver = None
        self.node = None  # abstract node index; None once off the tree
        self.real = None
        # The first turn of a heads-up hand is preflop, so stack + bet is what
        # each player started with. The tree only describes heads-up play.
        if view.num_players == 2 and view.street == "preflop":
            start = [s + b for s, b in zip(view.stacks, view.bets)]
            self.start = start
            self.real = GameState(list(start))
            self.real.setup_hand(
                list(start), view.button, view.small_blind, view.big_blind
            )
            self.node = 0
            self.abstract_bets = [book.small_blind, book.big_blind]
            self.abstract_behind = 0
            self.abstract_last = book.big_blind

    def _sizes(self, node):
        """Each abstract raise at node: (action index, fraction of the pot)."""
        bets = self.abstract_bets
        current = max(bets)
        pot = 2 * self.abstract_behind + sum(bets)
        owed = current - bets[node.player]
        return [
            (a, (node.to[a] - current) / (pot + owed))
            for a in range(node.nact)
            if node.to[a] > 0
        ]

    def _advance(self, node, action):
        """Moves the abstract state down one action."""
        on_patch = self.book is not self.blueprint
        self.history.append((self.node, action, self.book) if on_patch else (self.node, action))
        total = node.to[action]
        if total > 0:
            self.abstract_last = max(total - max(self.abstract_bets), self.abstract_last)
            self.abstract_bets[node.player] = total
        elif total == 0:
            self.abstract_bets[node.player] = max(self.abstract_bets)
        child = self.book.node(node.child[action])
        if child.kind == DECISION and child.street > node.street:
            self.abstract_behind += max(self.abstract_bets)
            self.abstract_bets = [0, 0]
            self.abstract_last = self.blueprint.big_blind
        self.node = node.child[action]

    def _opponent_raise(self, node, amount):
        """Interprets a raise, read before the real state applies it."""
        real, street = self.real, STREETS[self.real.street]
        if node is None or node.kind != DECISION or node.player == self.seat:
            self.matrix.record(self.team, street, DeviationMatrix.NO_NODE)
            self.node = None
            return
        sizes = self._sizes(node)
        patch = None
        passive = node.to.index(0, 0, node.nact)
        x = (amount - real.current_bet) / (real.pot + real.call_amount)
        column = DeviationMatrix.size_column(x)
        # Their all-in is our all-in, whatever the pots say: it is the last size.
        if sizes and amount == real.max_raise_to:
            column = DeviationMatrix.ON_TREE
            action = sizes[-1][0]
        else:
            if sizes and self.patcher is not None and self.book is self.blueprint:
                patch = self.patcher.hit(
                    self.patches, self.node, node, sizes, x, self.history,
                    self.abstract_bets, self.abstract_behind, self.abstract_last,
                )
            action = None
            for a, size in sizes:
                if abs(x - size) <= ON_TREE_TOLERANCE:
                    column = DeviationMatrix.ON_TREE
                    action = a
            if action is None:
                actions = [passive] + [a for a, _ in sizes]
                anchors = [0.0] + [size for _, size in sizes]
                action = actions[choose_anchor(x, anchors, self.mapping, self.rng)]
        self.matrix.record(self.team, street, column)
        self._advance(node, action)
        if action == passive:
            next_node = self.book.node(self.node)
            self.catch_up = not (
                next_node.kind == DECISION
                and next_node.street == street
                and next_node.player == self.seat
            )
        if patch is not None:
            self.book = patch
            self.node = 0  # the patch's root: we respond to this very bet
            self.abstract_bets = list(patch.root_bets)
            self.abstract_behind = patch.root_behind
            self.catch_up = False

    def observe(self, action):
        if self.real is None or action.kind not in ("fold", "call", "raise", "reveal_card"):
            return
        node = None if self.node is None else self.book.node(self.node)
        # Our own moves were applied to the abstract state in decide().
        if self.resolver is not None:
            self._observe_resolved(action)
        elif action.player_id == 1 - self.me:
            if action.kind == "raise":
                self._opponent_raise(node, action.amount)
            elif action.kind == "call":
                if node is None or node.kind != DECISION or node.player == self.seat:
                    self.node = None
                else:
                    self._advance(node, node.to.index(0, 0, node.nact))
        try:
            self.real.process_action(action)
        except ValueError:
            self.real = None  # should not happen: fall back to the view alone
            self.node = None

    def _observe_resolved(self, action):
        """After the hand has moved to the subgame solver: the abstract tree is
        no longer followed, but the real state and the matrix still are."""
        real = self.real
        if action.player_id == DEALER:
            self.resolver.observe_card(card_number(action.card))
        elif action.player_id == 1 - self.me and action.kind in ("call", "raise"):
            pot_after_call = real.pot + real.call_amount
            on_tree = self.resolver.observe_opponent(
                action, real, lambda total: (total - real.current_bet) / pot_after_call
            )
            if action.kind == "raise":
                x = (action.amount - real.current_bet) / pot_after_call
                self.matrix.record(
                    self.team, STREETS[real.street],
                    DeviationMatrix.ON_TREE if on_tree else DeviationMatrix.size_column(x),
                )

    def _resolve(self, view, hole, board):
        """Our move from an exact solve of this spot, or None to use the blueprint."""
        if self.resolver is None:
            self.resolver = self.make_resolver(self.history, board, hole, self.seat, self.start)
        self.catch_up = False
        try:
            move = self.resolver.decide(self.real, self.me, board, view.button)
            view.validate_action(move)
            return move
        except (ValueError, SubgameError):  # ValueError includes InvalidAction
            pass
        # The abstract tree stopped being followed once a solve was played.
        if self.resolver.last is not None:
            self.node = None
        self.resolver = None
        self.make_resolver = None
        return None

    def _off_tree(self, view, board):
        """No node describes this spot: take free cards, otherwise compare hand
        strength with the price. Buckets are ordered weakest first postflop.
        shortcut: strength is against a random hand, not a betting range;
        upgrade by re-solving the subgame."""
        self.node = None
        me = self.me
        if view.call_amount == 0 or not board:
            return PlayerAction.call(me)
        k = self.book.buckets[len(board) - 2]
        strength = (self.book.bucket(self.hole, board) + 0.5) / k
        return PlayerAction.call(me) if strength >= view.pot_odds else PlayerAction.fold(me)

    def decide(self, view, hole, board):
        self.hole = hole
        me = self.me
        if self.make_resolver is not None and self.real is not None and len(board) >= 4:
            move = self._resolve(view, hole, board)
            if move is not None:
                return move
        if self.catch_up:
            self.catch_up = False
            return PlayerAction.call(me)
        node = None if self.node is None else self.book.node(self.node)
        if (
            node is None
            or node.kind != DECISION
            or node.street != STREETS[view.street]
            or node.player != self.seat
        ):
            return self._off_tree(view, board)
        probs = self.book.probabilities(node, self.book.bucket(hole, board))
        action = self.rng.choices(range(node.nact), probs)[0]
        total = node.to[action]
        passive = node.to.index(0, 0, node.nact)
        if total == -1 and view.call_amount > 0:
            self._advance(node, action)
            return PlayerAction.fold(me)
        if total <= 0 or not view.can_raise:
            self._advance(node, passive)
            return PlayerAction.call(me)
        size = dict(self._sizes(node))[action]
        all_in = total == self.book.stack - self.abstract_behind
        low, high = view.raise_range
        if all_in:
            target = high
        else:
            target = view.current_bet + round(size * (view.pot + view.call_amount))
        self._advance(node, action)
        return PlayerAction.raise_to(me, max(low, min(high, target)))
