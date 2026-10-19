"""Play Gebeta against an AlphaZero-trained agent (Gradio app for Hugging Face Spaces)."""
import os

import gradio as gr

from gebeta_core import PITS, GebetaState, NumpyNet, best_move

try:
    # Hugging Face ZeroGPU hardware needs at least one @spaces.GPU function to start.
    # The game runs on CPU (the network is tiny), so this is never called.
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gebeta_model.json")
NET = NumpyNet(MODEL_PATH)
LEVELS = {"Easy": 20, "Medium": 120, "Hard": 400}

CSS = """
.board {background: linear-gradient(160deg,#8a5a2b,#6b4220); border-radius: 28px; padding: 18px;
        display: grid; grid-template-columns: 90px 1fr 90px; gap: 14px; align-items: center;
        box-shadow: inset 0 2px 8px rgba(0,0,0,.35), 0 6px 18px rgba(0,0,0,.25); max-width: 760px; margin: 0 auto;}
.rows {display: grid; grid-template-rows: 1fr 1fr; gap: 14px;}
.row {display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px;}
.pit, .store {background: radial-gradient(circle at 40% 35%, #4a2c12, #2e1a09); border-radius: 50%;
        aspect-ratio: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
        color: #f6e7c8; box-shadow: inset 0 4px 10px rgba(0,0,0,.6); position: relative;}
.store {border-radius: 44px; aspect-ratio: auto; height: 100%; min-height: 150px;}
.count {font-size: 22px; font-weight: 700; font-family: system-ui, sans-serif;}
.label {font-size: 11px; opacity: .7; font-family: system-ui, sans-serif;}
.seeds {display: flex; flex-wrap: wrap; justify-content: center; gap: 2px; max-width: 70%; margin-top: 3px;}
.seed {width: 7px; height: 7px; border-radius: 50%; background: #e9d3a3;}
.last {outline: 3px solid #ffcf4a; outline-offset: 2px;}
.side-label {grid-column: 1 / -1; text-align: center; color: #f6e7c8; font: 600 13px system-ui, sans-serif; opacity: .85;}
@media (max-width: 600px) { .board {grid-template-columns: 52px 1fr 52px; padding: 10px; gap: 8px;}
  .count {font-size: 16px;} .seeds {display: none;} .row {gap: 6px;} .rows {gap: 8px;} }
"""


def pit_html(n, label, last=False):
    dots = "".join('<span class="seed"></span>' for _ in range(min(n, 12)))
    cls = "pit last" if last else "pit"
    return f'<div class="{cls}"><span class="count">{n}</span><div class="seeds">{dots}</div><span class="label">{label}</span></div>'


def render(game):
    s, me = game["state"], game["human"]
    ai = 1 - me
    b = s.board
    last = game.get("last_ai_pit")
    mine = [pit_html(b[PITS * me + i], f"{i + 1}") for i in range(PITS)]
    # AI's pits shown right-to-left so sowing looks counter-clockwise from the human's seat
    theirs = [pit_html(b[PITS * ai + i], "", last == PITS * ai + i) for i in reversed(range(PITS))]
    return f"""
<div class="board">
  <div class="side-label">AI</div>
  <div class="store"><span class="label">AI</span><span class="count">{s.stores[ai]}</span></div>
  <div class="rows"><div class="row">{''.join(theirs)}</div><div class="row">{''.join(mine)}</div></div>
  <div class="store"><span class="label">You</span><span class="count">{s.stores[me]}</span></div>
  <div class="side-label">You</div>
</div>"""


def status_text(game):
    s, me = game["state"], game["human"]
    if s.is_terminal():
        sc = s.final_scores()
        w = s.winner()
        result = "Draw!" if w == -1 else ("**You win!** 🎉" if w == me else "**The AI wins.**")
        return f"{result}  Final score: you {sc[me]} – AI {sc[1 - me]} (seeds left on a row go to that row's owner)."
    msg = game.get("msg", "")
    q = game.get("ai_q")
    conf = f"  ·  AI's estimated win chance: {round(50 + 50 * q)}%" if q is not None else ""
    return f"{msg}  Your move: pick a pit (1–6, left to right on your row).{conf}"


def ai_turn(game, level):
    s = game["state"]
    if s.is_terminal() or s.player == game["human"]:
        return game
    a, q = best_move(NET, s, LEVELS[level])
    game["last_ai_pit"] = PITS * s.player + a
    game["ai_q"] = q
    game["state"] = s.play(a)
    game["msg"] = f"AI sowed from its pit {a + 1} (highlighted)."
    return game


def outputs(game):
    s = game["state"]
    can_play = not s.is_terminal() and s.player == game["human"]
    off = PITS * game["human"]
    buttons = [gr.update(interactive=can_play and s.board[off + i] > 0) for i in range(PITS)]
    return [game, render(game), status_text(game)] + buttons


def new_game(first, level):
    human = 0 if first == "You" else 1
    game = {"state": GebetaState(), "human": human, "msg": "New game."}
    game = ai_turn(game, level)
    return outputs(game)


def human_move(i, game, level):
    s = game["state"]
    if s.is_terminal() or s.player != game["human"] or not s.legal_mask()[i]:
        return outputs(game)
    game["state"] = s.play(i)
    game["last_ai_pit"] = None
    game = ai_turn(game, level)
    return outputs(game)


RULES_MD = """
### How to play Gebeta (ገበጣ)
* Two rows of 6 pits, 4 seeds in each. The bottom row is yours.
* On your turn, pick one of your pits. Its seeds are sown one per pit, **counter-clockwise**
  (left to right along your row, then right to left along the AI's row).
* **Relay:** if your last seed lands in a pit that already had seeds, you pick all of them up and keep sowing.
* **Capture:** if your turn ends in an empty pit on **your side**, you take the seeds in the pit opposite it.
* The game ends when the player to move has no seeds. Each player keeps the seeds left on their own row.
  Most seeds wins.

Gebeta has many regional variants; this is the common two-row version with relay sowing.
The AI was trained with AlphaZero-style self-play (neural network + Monte Carlo Tree Search).
"""

with gr.Blocks(css=CSS, title="Gebeta vs AlphaZero") as demo:
    gr.Markdown("# Gebeta vs AlphaZero\nPlay the Ethiopian mancala game against an AI that taught itself by self-play.")
    game_state = gr.State()
    with gr.Row():
        first = gr.Radio(["You", "AI"], value="You", label="Who goes first")
        level = gr.Radio(list(LEVELS), value="Medium", label="Difficulty")
        new_btn = gr.Button("New game", variant="primary")
    board = gr.HTML()
    with gr.Row():
        pit_btns = [gr.Button(f"Pit {i + 1}") for i in range(PITS)]
    status = gr.Markdown()
    with gr.Accordion("Rules", open=False):
        gr.Markdown(RULES_MD)

    outs = [game_state, board, status] + pit_btns
    new_btn.click(new_game, [first, level], outs)
    for i, btn in enumerate(pit_btns):
        btn.click(lambda g, lv, i=i: human_move(i, g, lv), [game_state, level], outs)
    demo.load(new_game, [first, level], outs)

if __name__ == "__main__":
    demo.launch()
