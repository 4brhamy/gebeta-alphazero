# Gebeta × AlphaZero

An AlphaZero-style AI for **Gebeta** (ገበጣ), the Ethiopian mancala game — trained purely by self-play,
plus a web app to play against it.

- **Play it:** https://huggingface.co/spaces/4brham/gebeta-alphazero
- **Train it:** [`notebook/gebeta_alphazero.ipynb`](notebook/gebeta_alphazero.ipynb) — open in Google Colab with a GPU runtime

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

1. **Network** — a residual MLP takes the board and outputs a *policy* (which pit to play) and a *value* (who is winning).
2. **Search** — Monte Carlo Tree Search (PUCT) uses the network to look ahead; the search's visit counts are a stronger policy than the raw network.
3. **Self-play** — the agent plays itself; every position is saved with the search policy and the final result.
4. **Training** — the network learns to predict the search policy and the game result. Repeat.

Self-play runs many games in parallel and batches network calls on the GPU.

## Repository layout

```
notebook/gebeta_alphazero.ipynb   Colab notebook: rules, network, MCTS, self-play training, export
notebook/gebeta_alphazero.py      Same code as a plain script (# %% cells)
app/app.py                        Gradio app (Hugging Face Space)
app/gebeta_core.py                Game rules + numpy inference + MCTS (no PyTorch needed)
app/gebeta_model.json             Trained weights exported by the notebook
```

## Run the app locally

```bash
cd app
pip install gradio numpy
python app.py
```

## Retrain

Open the notebook in Colab, choose a GPU runtime and run all cells. The "Export the model" cell writes
`gebeta_model.json`; copy it into `app/`.
