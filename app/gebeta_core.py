"""Gebeta game rules, a numpy version of the trained network, and MCTS.

Must match the rules and network in notebook/gebeta_alphazero.py. The weights come from
gebeta_model.json, exported by the notebook's "Export the model" cell.
"""
import json
import math

import numpy as np

PITS = 6
N_PITS = 12

RULES = dict(
    seeds_per_pit=4,
    relay=True,
    capture_opposite=True,
    capture_fours=False,
    skip_origin=False,
    max_plies=300,
    max_sow_steps=1000,
)


class GebetaState:
    """Pits 0-5 belong to player 0 (bottom row), pits 6-11 to player 1 (top row).
    Sowing is counter-clockwise (increasing index). Pit i is opposite pit 11 - i."""
    __slots__ = ("board", "stores", "player", "ply")

    def __init__(self, board=None, stores=None, player=0, ply=0):
        self.board = board if board is not None else [RULES["seeds_per_pit"]] * N_PITS
        self.stores = stores if stores is not None else [0, 0]
        self.player = player
        self.ply = ply

    def copy(self):
        return GebetaState(self.board[:], self.stores[:], self.player, self.ply)

    @staticmethod
    def owner(pit):
        return 0 if pit < PITS else 1

    def legal_moves(self):
        off = PITS * self.player
        return [a for a in range(PITS) if self.board[off + a] > 0]

    def legal_mask(self):
        off = PITS * self.player
        return np.array([self.board[off + a] > 0 for a in range(PITS)], dtype=bool)

    def is_terminal(self):
        half = (sum(self.board) + sum(self.stores)) // 2
        if self.stores[0] > half or self.stores[1] > half:
            return True
        if self.ply >= RULES["max_plies"]:
            return True
        return not self.legal_moves()

    def final_scores(self):
        s = self.stores[:]
        s[0] += sum(self.board[:PITS])
        s[1] += sum(self.board[PITS:])
        return s

    def winner(self):
        s = self.final_scores()
        if s[0] == s[1]:
            return -1
        return 0 if s[0] > s[1] else 1

    def terminal_value(self):
        w = self.winner()
        return 0.0 if w == -1 else (1.0 if w == self.player else -1.0)

    def play(self, action, trace=None):
        """Return the new state after sowing from relative pit `action` (0-5).
        If `trace` is a list, the pit index of every seed dropped is appended to it."""
        s = self.copy()
        b, me = s.board, s.player
        origin = PITS * me + action
        seeds = b[origin]
        assert seeds > 0, "illegal move: empty pit"
        b[origin] = 0
        pos, steps = origin, 0
        while True:
            while seeds > 0:
                pos = (pos + 1) % N_PITS
                if RULES["skip_origin"] and pos == origin:
                    continue
                b[pos] += 1
                seeds -= 1
                steps += 1
                if trace is not None:
                    trace.append(pos)
                if RULES["capture_fours"] and b[pos] == 4 and seeds > 0:
                    s.stores[self.owner(pos)] += 4
                    b[pos] = 0
            if steps > RULES["max_sow_steps"]:
                break
            if RULES["capture_fours"] and b[pos] == 4:
                s.stores[me] += 4
                b[pos] = 0
                break
            if RULES["relay"] and b[pos] > 1:
                seeds, b[pos] = b[pos], 0
                continue
            break
        if RULES["capture_opposite"] and b[pos] == 1 and self.owner(pos) == me:
            opp = N_PITS - 1 - pos
            if b[opp] > 0:
                s.stores[me] += b[opp] + 1
                b[opp] = 0
                b[pos] = 0
        s.player = 1 - me
        s.ply += 1
        return s

    def encode(self):
        me = self.player
        own = self.board[PITS * me: PITS * me + PITS]
        opp = self.board[PITS * (1 - me): PITS * (1 - me) + PITS]
        total = 12 * RULES["seeds_per_pit"]
        x = np.array(own + opp, dtype=np.float32) / 8.0
        st = np.array([self.stores[me], self.stores[1 - me]], dtype=np.float32) / total
        nonempty = (np.array(own + opp) > 0).astype(np.float32)
        ply = np.array([self.ply / RULES["max_plies"]], dtype=np.float32)
        return np.concatenate([x, st, nonempty, ply])


class NumpyNet:
    """Inference-only copy of GebetaNet from the notebook (residual MLP with LayerNorm)."""

    def __init__(self, path):
        with open(path) as f:
            data = json.load(f)
        RULES.update(data["meta"]["rules"])
        self.meta = data["meta"]
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in data["weights"].items()}
        self.blocks = data["meta"]["blocks"]

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T + self.w[name + ".bias"]

    def _ln(self, x, name, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + eps) * self.w[name + ".weight"] + self.w[name + ".bias"]

    def forward(self, x):
        relu = lambda t: np.maximum(t, 0)
        h = relu(self._lin(x, "inp.0"))
        for i in range(self.blocks):
            y = relu(self._ln(self._lin(h, f"body.{i}.l1"), f"body.{i}.n1"))
            y = self._ln(self._lin(y, f"body.{i}.l2"), f"body.{i}.n2")
            h = relu(h + y)
        logits = self._lin(relu(self._lin(h, "policy.0")), "policy.2")
        value = np.tanh(self._lin(relu(self._lin(h, "value.0")), "value.2"))[..., 0]
        return logits, value

    def evaluate(self, state):
        logits, v = self.forward(state.encode()[None, :])
        logits = np.where(state.legal_mask(), logits[0], -1e9)
        p = np.exp(logits - logits.max())
        return p / p.sum(), float(v[0])


class Node:
    __slots__ = ("state", "P", "N", "W", "children", "expanded", "terminal", "tval")

    def __init__(self, state):
        self.state = state
        self.P = np.zeros(PITS, dtype=np.float32)
        self.N = np.zeros(PITS, dtype=np.float32)
        self.W = np.zeros(PITS, dtype=np.float32)
        self.children = {}
        self.expanded = False
        self.terminal = state.is_terminal()
        self.tval = state.terminal_value() if self.terminal else 0.0

    def select(self, c_puct):
        legal = self.state.legal_mask()
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)
        u = c_puct * self.P * math.sqrt(self.N.sum() + 1) / (1 + self.N)
        return int(np.argmax(np.where(legal, q + u, -np.inf)))


def mcts(net, state, sims=200, c_puct=1.5):
    """Return (visit counts, root value estimate for the player to move)."""
    root = Node(state)
    root.P, _ = net.evaluate(state)
    root.expanded = True
    for _ in range(sims):
        node, path = root, []
        while node.expanded and not node.terminal:
            a = node.select(c_puct)
            path.append((node, a))
            if a not in node.children:
                node.children[a] = Node(node.state.play(a))
            node = node.children[a]
        if node.terminal:
            v = node.tval
        else:
            node.P, v = net.evaluate(node.state)
            node.expanded = True
        for n, a in reversed(path):
            v = -v
            n.N[a] += 1
            n.W[a] += v
    q = float((root.W.sum() / root.N.sum()) if root.N.sum() else 0.0)
    return root.N, q


def best_move(net, state, sims=200):
    visits, q = mcts(net, state, sims)
    return int(np.argmax(visits)), q
