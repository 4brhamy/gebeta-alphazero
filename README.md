# Gebeta × AlphaZero

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4brhamy/gebeta-alphazero/blob/main/notebook/gebeta_alphazero.ipynb)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Play-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/4brham/gebeta-alphazero)

An AlphaZero-style AI for **Gebeta** (ገበጣ), the Ethiopian mancala game. It learns only by playing
against itself, with no human games or hand-written strategy. A web app lets you play against it.

- **Play it:** https://huggingface.co/spaces/4brham/gebeta-alphazero
- **Train it:** [`notebook/gebeta_alphazero.ipynb`](notebook/gebeta_alphazero.ipynb). Open it in Google Colab with a GPU runtime.

## About the project

Gebeta is one of the oldest board games in the Horn of Africa. It is played in homes, schools and
cafés across Ethiopia and Eritrea, often on a carved wooden board or on pits dug in the ground. Like other
mancala games, it is easy to learn: on each turn you only pick one pit. It is hard to master, because one
choice can set off a long chain of sowing that changes the whole board.

This project applies the method behind DeepMind's AlphaGo Zero and AlphaZero to Gebeta. The agent starts
knowing nothing but the rules. Each training round, it plays many games against itself, using a neural
network and Monte Carlo Tree Search to choose moves. It then trains the network on the results. After a
short run on a free Colab GPU, it beats simple strategies in nearly every game.

The project has three parts:

| Part | What it is |
|---|---|
| **Training notebook** | One Colab notebook with the game engine, the neural network, the search, the self-play training loop, evaluation, and model export. |
| **Trained model** | `app/gebeta_model.npz` (2 MB, 604,295 weights) from a 60-iteration training run. |
| **Web app** | A Gradio app on Hugging Face Spaces with a wooden-style board, three difficulty levels, a choice of who moves first, and the AI's own estimate of its win chance. It runs the network with plain numpy, so it needs no PyTorch or GPU. |

## Results

Training ran for 60 iterations (about 45 minutes on a Colab T4 GPU) and produced about 122,000 self-play positions.

| | Before training | After 60 iterations |
|---|---|---|
| vs random player (40 games) | 30 wins, 1 draw, 9 losses | 40 wins |
| vs greedy player (40 games) | not measured | 33 wins, 2 draws, 5 losses |

In a separate test at the app's Medium level (120 searches per move), the final model won all 20 games
against the random and greedy players, playing both first and second.

**Moving first is a big advantage.** By the end of training, the first player won about 94% of self-play
games, and the network rates the opening position at +0.91 for the player to move. Some losses to the
greedy player may come from games where the AI moved second, but the code doesn't record which side lost.

## Rules used

Gebeta is a family of games with regional variants. This project uses the common two-row version:

| | |
|---|---|
| Board | 2 rows × 6 pits, 4 seeds per pit (48 total). Each player owns the row nearest them. |
| Move | Take all seeds from one of your pits and sow them one per pit, counter-clockwise. |
| Relay sowing | If the last seed lands in an occupied pit, pick that pit up and keep sowing. |
| Capture | If the turn ends in an empty pit on your side, take the seeds in the opposite pit (plus your seed). |
| End | When the player to move has no seeds, each player keeps the seeds on their row. Most seeds wins. |

Variants (e.g. capturing pits that reach 4 seeds) can be switched on in the `RULES` dict in the notebook.

## Reinforcement learning

The agent is trained with model-based reinforcement learning, following AlphaGo Zero / AlphaZero.
It is never shown a human game or told what a good move is. It only gets the rules and the result of
each game it plays against itself.

### The RL problem

| RL term | In this project |
|---|---|
| Agent | The neural network together with Monte Carlo Tree Search (MCTS) |
| Environment | The Gebeta rules (`GebetaState` in the notebook) |
| State | The 12 pits, both stores and whose turn it is, encoded as 27 numbers from the mover's point of view |
| Action | One of the mover's 6 pits (empty pits are masked out) |
| Reward | Only at the end of the game: +1 win, −1 loss, 0 draw. Captures during the game give no reward. |
| Policy | Probability of playing each pit |
| Value | Expected final result for the player to move, from −1 to +1 |

Because the only reward comes at the end of the game, the agent has to work out which earlier moves led to the
win. The value network and the tree search solve that together.

### The training loop

```
        ┌──────────────┐  (state, π, z)  ┌────────────────┐  samples  ┌──────────────┐
        │  Self-play   │ ──────────────▶ │ Replay buffer  │ ────────▶ │   Training   │
        │ MCTS + net   │                 │ last 100k pos. │           │ policy+value │
        └──────▲───────┘                 └────────────────┘           └──────┬───────┘
               │                     improved network                        │
               └─────────────────────────────────────────────────────────────┘
```

Each iteration:

1. **Self-play.** 128 games run in parallel. For every move, MCTS runs 100 simulations guided by the
   network. The visit counts at the root, normalised, become the search policy **π**. For the first 12 moves
   a move is sampled from π, for variety. After that, the most-visited move is played.
2. **Labelling.** When a game ends, every position gets the final result **z** (+1/−1/0) from the
   point of view of the player who was to move there.
3. **Training.** 300 gradient steps on random batches of 512 positions from the replay buffer, which holds
   the last 100,000.
4. **Evaluation.** Every 5 iterations, the agent plays 40 games each against a random and a greedy player.

### The search (MCTS with PUCT)

Each simulation walks down the tree, picking at each node the move with the highest score:

```
score(a) = Q(a) + c · P(a) · √(N_total + 1) / (1 + N(a))        c = 1.5
```

`Q` is the average result of that move so far, `P` is the network's prior, and `N` counts visits. The
first term favours moves that have worked. The second favours promising moves that haven't been tried much.
At a new leaf, the network supplies `P` and a value `v`. That value is then backed up the path, flipping sign
at each level because the players alternate. During self-play, Dirichlet noise (α = 0.6, 25%) is mixed
into the root priors so the agent keeps trying moves it doesn't yet favour.

### What the network learns

The network has a shared residual body (4 blocks, 256 wide) and two heads. It is trained to match the
search:

```
loss = −Σ π(a) · log p(a)   +   (z − v)²
         policy: match search      value: predict the result
```

The optimizer is AdamW (learning rate 1e-3 with cosine decay to 1e-4, weight decay 1e-4).

### Why it improves

Search with the network plays better than the network alone, so π is a better target than the
network's own policy. Training on π makes the network stronger. A stronger network makes the next search
stronger still. This feedback loop is policy iteration: MCTS improves the policy and supervised training
brings the network up to it. Over the 60-iteration run, the policy loss fell from 1.30 to 0.99 and the
value loss from 0.41 to 0.09.

### Engineering notes

- Self-play batches network calls across all 128 games, so one GPU call evaluates 128 positions.
  One iteration takes about 30 seconds on a T4.
- The board is always encoded from the mover's point of view, so one network plays both sides.
- The web app re-implements the forward pass in numpy. It was checked against PyTorch (max difference 0.001).

## Repository layout

```
notebook/gebeta_alphazero.ipynb   Colab notebook: rules, network, MCTS, self-play training, export
notebook/gebeta_alphazero.py      Same code as a plain script (# %% cells)
app/app.py                        Gradio app (Hugging Face Space)
app/gebeta_core.py                Game rules + numpy inference + MCTS (no PyTorch needed)
app/gebeta_model.npz              Trained weights exported by the notebook
```

## Run the app locally

```bash
cd app
pip install gradio numpy
python app.py
```

## Retrain

Open the notebook in Colab, choose a GPU runtime and run all cells. The "Export the model" cell writes
`gebeta_model.npz`; copy it into `app/`. If you change the rules, change them in `app/gebeta_core.py` too.

## License

MIT
