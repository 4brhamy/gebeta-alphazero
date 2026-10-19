# %% [markdown]
# # Gebeta × AlphaZero
#
# An AlphaZero-style agent (the self-play successor of AlphaGo) for **Gebeta**, the Ethiopian mancala game.
#
# **How it learns:** a neural network looks at the board and outputs (1) a *policy*: how good each of the 6 pits
# looks, and (2) a *value*: who is likely to win. Monte Carlo Tree Search (MCTS) uses the network to look ahead,
# and the improved move choices from the search become the training targets. The agent only ever plays itself.
#
# **Colab:** `Runtime → Change runtime type → GPU` (T4 is plenty). CPU also works, just slower.
# Run the cells top to bottom. Training saves checkpoints, and the last cells let you play against the agent.

# %% [markdown]
# ## 1. Rules of Gebeta used here
#
# "Gebeta" (ገበጣ) is the name for a family of mancala games in Ethiopia and Eritrea; rules vary by region.
# This notebook uses the common two-row version, with the regional differences as options in `RULES`:
#
# * **Board:** 2 rows × 6 pits. Each player owns the row nearest them. 4 seeds in every pit (48 total).
# * **Move:** pick up all seeds from one of your non-empty pits and sow them one by one, counter-clockwise.
# * **Relay sowing** (`relay=True`): if the last seed lands in a pit that already had seeds, pick up
#   that whole pit and keep sowing. The turn continues until the last seed lands in an empty pit.
# * **Capture** (`capture_opposite=True`): if the turn ends in an empty pit on **your own side** and the opposite
#   pit has seeds, you take the opposite pit's seeds (plus your landing seed).
# * **Fours** (`capture_fours=False`, optional variant): whenever sowing makes a pit hold exactly 4,
#   the owner of that pit's side takes those 4 seeds.
# * **End:** when the player to move has no seeds, the opponent keeps all seeds left on their side.
#   Most seeds wins. Reaching more than half (25+) ends the game early.

# %%
import math, random, time, copy, os
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PITS = 6                 # pits per row
N_PITS = 2 * PITS        # 12 pits on the board

RULES = dict(
    seeds_per_pit=4,
    relay=True,             # keep sowing from the landing pit if it was occupied
    capture_opposite=True,  # ending in an empty own pit captures the opposite pit
    capture_fours=False,    # any pit that reaches exactly 4 is taken by that side's owner
    skip_origin=False,      # Oware-style: never drop a seed back into the starting pit
    max_plies=300,          # safety cap; the game is scored as-is after this
    max_sow_steps=1000,     # safety cap for very long relay chains
)

# %%
class GebetaState:
    """Immutable-style game state. Pits 0-5 belong to player 0, pits 6-11 to player 1.
    Sowing goes counter-clockwise = increasing index. Pit i is opposite pit 11 - i."""
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
        total = sum(self.board) + sum(self.stores)
        half = total // 2
        if self.stores[0] > half or self.stores[1] > half:
            return True
        if self.ply >= RULES["max_plies"]:
            return True
        return not self.legal_moves()

    def final_scores(self):
        """Scores if the game ended now: each side also keeps the seeds on its own row."""
        s = self.stores[:]
        s[0] += sum(self.board[:PITS])
        s[1] += sum(self.board[PITS:])
        return s

    def winner(self):
        """0 or 1 for the winner, -1 for a draw. Only meaningful when terminal."""
        s = self.final_scores()
        if s[0] == s[1]:
            return -1
        return 0 if s[0] > s[1] else 1

    def terminal_value(self):
        """+1 / -1 / 0 from the point of view of the player to move."""
        w = self.winner()
        return 0.0 if w == -1 else (1.0 if w == self.player else -1.0)

    def play(self, action):
        """Return the new state after the current player sows from relative pit `action` (0-5)."""
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
                if RULES["capture_fours"] and b[pos] == 4 and seeds > 0:
                    s.stores[self.owner(pos)] += 4
                    b[pos] = 0
            if steps > RULES["max_sow_steps"]:
                break
            if RULES["capture_fours"] and b[pos] == 4:
                s.stores[me] += 4          # the last seed's 4 goes to the mover
                b[pos] = 0
                break
            if RULES["relay"] and b[pos] > 1:
                seeds, b[pos] = b[pos], 0  # pick up the pit and keep going
                continue
            break
        # capture: ended in a pit that was empty, on my side
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
        """Features from the mover's point of view: own row first, then opponent row, then stores."""
        me = self.player
        own = self.board[PITS * me: PITS * me + PITS]
        opp = self.board[PITS * (1 - me): PITS * (1 - me) + PITS]
        total = 12 * RULES["seeds_per_pit"]
        x = np.array(own + opp, dtype=np.float32) / 8.0
        st = np.array([self.stores[me], self.stores[1 - me]], dtype=np.float32) / total
        nonempty = (np.array(own + opp) > 0).astype(np.float32)
        ply = np.array([self.ply / RULES["max_plies"]], dtype=np.float32)
        return np.concatenate([x, st, nonempty, ply])

    def __str__(self):
        b = self.board
        top = " ".join(f"{b[i]:2d}" for i in range(N_PITS - 1, PITS - 1, -1))
        bot = " ".join(f"{b[i]:2d}" for i in range(PITS))
        return (f"      P1 pits:  6  5  4  3  2  1\n"
                f"P1 [{self.stores[1]:2d}]    {top}\n"
                f"P0 [{self.stores[0]:2d}]    {bot}\n"
                f"      P0 pits:  1  2  3  4  5  6\n"
                f"  to move: P{self.player}   ply {self.ply}")

FEATURES = len(GebetaState().encode())

# quick sanity check
_s = GebetaState()
print(_s)
print("features:", FEATURES)

# %% [markdown]
# ## 2. Policy-value network
# A small residual MLP. Gebeta's state is tiny (14 numbers), so no convolution is needed.

# %%
class ResBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.l1, self.l2 = nn.Linear(h, h), nn.Linear(h, h)
        self.n1, self.n2 = nn.LayerNorm(h), nn.LayerNorm(h)

    def forward(self, x):
        y = F.relu(self.n1(self.l1(x)))
        y = self.n2(self.l2(y))
        return F.relu(x + y)

class GebetaNet(nn.Module):
    def __init__(self, hidden=256, blocks=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(FEATURES, hidden), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(hidden) for _ in range(blocks)])
        self.policy = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, PITS))
        self.value = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1), nn.Tanh())

    def forward(self, x):
        h = self.body(self.inp(x))
        return self.policy(h), self.value(h).squeeze(-1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

@torch.no_grad()
def evaluate_batch(net, states):
    """Return (priors masked to legal moves, values) for a list of states."""
    x = torch.from_numpy(np.stack([s.encode() for s in states])).to(DEVICE)
    logits, v = net(x)
    masks = torch.from_numpy(np.stack([s.legal_mask() for s in states])).to(DEVICE)
    logits = logits.masked_fill(~masks, -1e9)
    p = torch.softmax(logits, dim=-1)
    return p.cpu().numpy(), v.cpu().numpy()

# %% [markdown]
# ## 3. Monte Carlo Tree Search (PUCT, as in AlphaGo Zero / AlphaZero)
# Searches for many games at once so the network is called on a batch (fast on GPU).
# Players strictly alternate in this rule set, so a child's value is the negative of its parent's.

# %%
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
        total = self.N.sum()
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)
        u = c_puct * self.P * math.sqrt(total + 1) / (1 + self.N)
        score = np.where(legal, q + u, -np.inf)
        return int(np.argmax(score))

def mcts_search(net, states, sims, c_puct=1.5, dirichlet_alpha=0.6, noise_frac=0.25, add_noise=True):
    """Run MCTS from each (non-terminal) state. Returns a list of visit-count vectors."""
    roots = [Node(s) for s in states]
    priors, _ = evaluate_batch(net, [r.state for r in roots])
    for r, p in zip(roots, priors):
        if add_noise:
            legal = r.state.legal_mask()
            noise = np.zeros(PITS, dtype=np.float32)
            noise[legal] = np.random.dirichlet([dirichlet_alpha] * legal.sum())
            p = (1 - noise_frac) * p + noise_frac * noise
        r.P, r.expanded = p, True

    for _ in range(sims):
        leaves, paths = [], []
        for root in roots:
            node, path = root, []
            while node.expanded and not node.terminal:
                a = node.select(c_puct)
                path.append((node, a))
                if a not in node.children:
                    node.children[a] = Node(node.state.play(a))
                node = node.children[a]
            if node.terminal:
                backup(path, node.tval)
            else:
                leaves.append(node)
                paths.append(path)
        if leaves:
            priors, values = evaluate_batch(net, [n.state for n in leaves])
            for node, path, p, v in zip(leaves, paths, priors, values):
                node.P, node.expanded = p, True
                backup(path, float(v))
    return [r.N.copy() for r in roots]

def backup(path, v):
    """v is the value for the player to move at the leaf; flip sign at each step up."""
    for node, a in reversed(path):
        v = -v
        node.N[a] += 1
        node.W[a] += v

# %% [markdown]
# ## 4. Opponents for measuring progress

# %%
def random_agent(state):
    return random.choice(state.legal_moves())

def greedy_agent(state):
    """Picks the move that gains the most seeds right now (ties broken randomly)."""
    me = state.player
    best, best_gain = [], -1
    for a in state.legal_moves():
        gain = state.play(a).stores[me] - state.stores[me]
        if gain > best_gain:
            best, best_gain = [a], gain
        elif gain == best_gain:
            best.append(a)
    return random.choice(best)

def play_match(net, opponent, games=40, sims=50):
    """AlphaZero agent (no noise, greedy) vs a simple agent; half the games each side.
    Returns (wins, draws, losses) for the AlphaZero agent."""
    net.eval()
    states = [GebetaState() for _ in range(games)]
    az_side = [i % 2 for i in range(games)]
    # random opening move for variety
    states = [s.play(random_agent(s)) for s in states]
    while True:
        live = [i for i, s in enumerate(states) if not s.is_terminal()]
        if not live:
            break
        az_turn = [i for i in live if states[i].player == az_side[i]]
        if az_turn:
            visits = mcts_search(net, [states[i] for i in az_turn], sims, add_noise=False)
            for i, n in zip(az_turn, visits):
                states[i] = states[i].play(int(np.argmax(n)))
        for i in live:
            if i not in az_turn and not states[i].is_terminal():
                states[i] = states[i].play(opponent(states[i]))
    w = d = l = 0
    for s, side in zip(states, az_side):
        r = s.winner()
        if r == -1: d += 1
        elif r == side: w += 1
        else: l += 1
    return w, d, l

def make_net_agent(net, sims=200):
    def agent(state):
        visits = mcts_search(net, [state], sims, add_noise=False)[0]
        return int(np.argmax(visits))
    return agent

# %% [markdown]
# ## 5. Self-play and training settings
# Defaults fit a free Colab GPU in roughly 1-2 hours. For a quick smoke test, set `iterations=3`.

# %%
@dataclass
class Config:
    iterations: int = 60          # self-play → train cycles
    games_per_iter: int = 128     # games played in parallel each iteration
    sims: int = 100               # MCTS simulations per move during self-play
    temp_moves: int = 12          # sample moves by visit count for the first N plies, then play the best
    buffer_size: int = 100_000    # training positions kept
    batch_size: int = 512
    train_steps: int = 300        # gradient steps per iteration
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 5
    eval_games: int = 40
    eval_sims: int = 50
    hidden: int = 256
    blocks: int = 4
    save_dir: str = "checkpoints"

cfg = Config()

# Optional: save checkpoints to Google Drive so they survive a Colab disconnect.
USE_DRIVE = False
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    cfg.save_dir = "/content/drive/MyDrive/gebeta_alphazero"
os.makedirs(cfg.save_dir, exist_ok=True)

# %%
def self_play(net, cfg):
    """Play cfg.games_per_iter games in parallel. Returns training samples (x, pi, z)."""
    net.eval()
    states = [GebetaState() for _ in range(cfg.games_per_iter)]
    history = [[] for _ in states]  # (features, pi, player)
    while True:
        live = [i for i, s in enumerate(states) if not s.is_terminal()]
        if not live:
            break
        visits = mcts_search(net, [states[i] for i in live], cfg.sims, add_noise=True)
        for i, n in zip(live, visits):
            s = states[i]
            pi = n / n.sum()
            history[i].append((s.encode(), pi.astype(np.float32), s.player))
            if s.ply < cfg.temp_moves:
                a = int(np.random.choice(PITS, p=pi))
            else:
                a = int(np.argmax(n))
            states[i] = s.play(a)
    samples, results = [], [0, 0, 0]
    for s, h in zip(states, history):
        w = s.winner()
        results[w if w >= 0 else 2] += 1
        for x, pi, p in h:
            z = 0.0 if w == -1 else (1.0 if w == p else -1.0)
            samples.append((x, pi, np.float32(z)))
    return samples, results

def train(net, opt, buffer, cfg):
    net.train()
    tot_p = tot_v = 0.0
    for _ in range(cfg.train_steps):
        batch = random.sample(buffer, min(cfg.batch_size, len(buffer)))
        x = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(DEVICE)
        z = torch.tensor([b[2] for b in batch], device=DEVICE)
        logits, v = net(x)
        loss_p = -(pi * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        loss_v = F.mse_loss(v, z)
        loss = loss_p + loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot_p += loss_p.item()
        tot_v += loss_v.item()
    return tot_p / cfg.train_steps, tot_v / cfg.train_steps

def save(net, cfg, name):
    path = os.path.join(cfg.save_dir, name)
    torch.save({"model": net.state_dict(), "rules": RULES, "cfg": cfg.__dict__}, path)
    return path

# %% [markdown]
# ## 6. Train
# Watch `vs random` and `vs greedy` climb. A good agent should beat random ~100% and greedy 85%+.

# %%
net = GebetaNet(cfg.hidden, cfg.blocks).to(DEVICE)
opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iterations, eta_min=cfg.lr * 0.1)
buffer = deque(maxlen=cfg.buffer_size)
log = []

w, d, l = play_match(net, random_agent, cfg.eval_games, cfg.eval_sims)
print(f"untrained net + MCTS vs random: {w}W {d}D {l}L")

for it in range(1, cfg.iterations + 1):
    t0 = time.time()
    samples, res = self_play(net, cfg)
    buffer.extend(samples)
    lp, lv = train(net, opt, buffer, cfg)
    sched.step()
    row = dict(iter=it, positions=len(samples), buffer=len(buffer), loss_p=lp, loss_v=lv,
               p0_wins=res[0], p1_wins=res[1], draws=res[2], secs=time.time() - t0)
    msg = (f"iter {it:3d} | +{len(samples):5d} pos (buf {len(buffer):6d}) | "
           f"policy {lp:.3f} value {lv:.3f} | P0/P1/draw {res[0]}/{res[1]}/{res[2]} | {row['secs']:.0f}s")
    if it % cfg.eval_every == 0 or it == cfg.iterations:
        wr = play_match(net, random_agent, cfg.eval_games, cfg.eval_sims)
        wg = play_match(net, greedy_agent, cfg.eval_games, cfg.eval_sims)
        row.update(vs_random=wr, vs_greedy=wg)
        msg += f"\n           vs random {wr[0]}W {wr[1]}D {wr[2]}L | vs greedy {wg[0]}W {wg[1]}D {wg[2]}L"
        save(net, cfg, f"gebeta_iter{it:03d}.pt")
    save(net, cfg, "gebeta_latest.pt")
    log.append(row)
    print(msg)

# %% [markdown]
# ## 7. Training curves

# %%
import matplotlib.pyplot as plt
its = [r["iter"] for r in log]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot(its, [r["loss_p"] for r in log], label="policy loss")
ax[0].plot(its, [r["loss_v"] for r in log], label="value loss")
ax[0].set_xlabel("iteration"); ax[0].legend(); ax[0].set_title("Losses")
ev = [r for r in log if "vs_greedy" in r]
ax[1].plot([r["iter"] for r in ev], [r["vs_random"][0] / cfg.eval_games for r in ev], "o-", label="win rate vs random")
ax[1].plot([r["iter"] for r in ev], [r["vs_greedy"][0] / cfg.eval_games for r in ev], "o-", label="win rate vs greedy")
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("iteration"); ax[1].legend(); ax[1].set_title("Strength")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 8. Export the model for the web app
# Writes the weights and rules to `gebeta_model.npz`, which the app (`app/gebeta_core.py`) loads with
# numpy only. Download it from the Files panel (or run `files.download(path)`) and put it in `app/`.

# %%
import json

def export_web(net, path):
    weights = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    meta = {"rules": RULES, "hidden": cfg.hidden, "blocks": cfg.blocks, "features": FEATURES,
            "iterations": cfg.iterations}
    np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **weights)
    print(path, os.path.getsize(path) // 1024, "KB")

export_web(net, os.path.join(cfg.save_dir, "gebeta_model.npz"))

# %% [markdown]
# ## 9. Play against the agent
# Enter a pit number 1-6 (counted left to right on your row). More `sims` = stronger, slower.

# %%
def load(path):
    ck = torch.load(path, map_location=DEVICE)
    RULES.update(ck["rules"])
    c = ck["cfg"]
    m = GebetaNet(c["hidden"], c["blocks"]).to(DEVICE)
    m.load_state_dict(ck["model"])
    m.eval()
    return m

def play_human(net, human_side=0, sims=400):
    ai = make_net_agent(net, sims)
    s = GebetaState()
    while not s.is_terminal():
        print("\n" + str(s))
        if s.player == human_side:
            legal = [a + 1 for a in s.legal_moves()]
            a = None
            while a not in legal:
                try:
                    a = int(input(f"your pit {legal}: "))
                except ValueError:
                    pass
            s = s.play(a - 1)
        else:
            a = ai(s)
            print(f"AI sows pit {a + 1}")
            s = s.play(a)
    print("\n" + str(s))
    sc = s.final_scores()
    w = s.winner()
    print(f"final score P0 {sc[0]} - P1 {sc[1]}  →",
          "draw" if w == -1 else ("you win!" if w == human_side else "AI wins"))

# net = load(os.path.join(cfg.save_dir, "gebeta_latest.pt"))   # to reload later
play_human(net, human_side=0)
