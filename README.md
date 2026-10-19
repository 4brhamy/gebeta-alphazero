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

## How the AI works

Same recipe as AlphaGo Zero / AlphaZero:

1. **Network**: a residual MLP takes the board and outputs a *policy* (which pit to play) and a *value* (who is winning).
2. **Search**: Monte Carlo Tree Search (PUCT) uses the network to look ahead. The search's visit counts are a stronger policy than the raw network.
3. **Self-play**: the agent plays itself. Every position is saved with the search policy and the final result.
4. **Training**: the network learns to predict the search policy and the game result. Repeat.

Self-play runs 128 games in parallel and batches network calls on the GPU, so one training iteration
takes about 30 seconds.

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
