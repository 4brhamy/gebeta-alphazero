---
title: Gebeta vs AlphaZero
emoji: 🌰
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Play Ethiopian mancala Gebeta against an AlphaZero AI
---

# Gebeta vs AlphaZero

Play **Gebeta** (ገበጣ), the Ethiopian mancala game, against an AI that learned the game only by playing
against itself. It saw no human games and was given no strategy.

Code and training notebook: https://github.com/4brhamy/gebeta-alphazero

## About Gebeta

Gebeta is one of the oldest board games in the Horn of Africa. It is played across Ethiopia and Eritrea,
often on a carved wooden board or on pits dug in the ground. It is easy to learn, since each turn you
only pick one pit. It is hard to master, because one choice can set off a long chain of sowing that
changes the whole board.

## How to play

- The board has 2 rows of 6 pits, with 4 seeds in each. **The bottom row is yours.**
- On your turn, **pick one of your pits**. That is your only decision; everything else happens automatically.
- Its seeds are sown one per pit, **counter-clockwise**: right along your row, then left along the AI's row.
- **Relay:** if the last seed lands in a pit that already had seeds, those seeds are picked up and sowing continues.
- **Capture:** if your turn ends in an empty pit on **your side**, you take the seeds in the pit opposite it.
- The game ends when the player to move has no seeds. Each player keeps the seeds on their own row.
  Most seeds wins (25 of 48 guarantees it).

**Tip:** moving first is a big advantage with these rules. Try "AI goes first" if you want a real challenge.

## Using the app

| Option | What it does |
|---|---|
| Who goes first | You or the AI. |
| Difficulty | How far the AI looks ahead: Easy (20 searches per move), Medium (120), Hard (400). |
| Pit 1–6 | Your move, counted left to right on your row. Empty pits are disabled. |
| Win chance | The AI's own estimate of its chances, shown after each of its moves. |

The AI's last move is highlighted in yellow.

## How the AI was trained

The agent uses the **AlphaZero** reinforcement learning method:

- **Neural network:** reads the board and outputs a *policy* (which pit looks best) and a *value*
  (who is likely to win). It is a residual network with 604,295 weights.
- **Monte Carlo Tree Search:** uses the network to look ahead before each move. The search makes much
  stronger choices than the network alone.
- **Self-play:** each training round, the AI plays 128 games against itself. The only reward is the final
  result: +1 for a win, −1 for a loss, 0 for a draw.
- **Training:** the network learns to match the search's move choices and to predict the final result.
  The improved network then guides the next round of self-play.

Training ran for 60 rounds, about 45 minutes on a free Google Colab GPU.

## Results

| Opponent (40 games) | Before training | After 60 rounds |
|---|---|---|
| Random player | 30 wins, 1 draw, 9 losses | 40 wins |
| Greedy player (grabs the most seeds each turn) | not measured | 33 wins, 2 draws, 5 losses |

At this app's Medium level, the final model won all 20 test games against these players, both as first
and second player.

## Notes

- Gebeta has many regional variants. This is the common two-row version with relay sowing.
- The app runs the network with plain numpy on CPU, so each move takes under a second.
