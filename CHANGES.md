Changes & How to run (enhancements)

This fork adds convenience, faster training options, model checkpointing, and small model improvements.

- Added `requirements.txt` with main dependencies.
- `game.SnakeGameAI` now accepts `render` and `speed` parameters so training can run without rendering for speed.
- `model.Linear_QNet` now supports multiple hidden layers and has a `load()` helper.
- `agent.py` now accepts CLI args to control training/eval.

Usage examples (PowerShell):

```powershell
cd 'c:\Users\Adil\Desktop\FAI\snake-ai-pytorch'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train fast (no render) with default settings
python agent.py --fast

# Train with render and save checkpoints every 100 games
python agent.py --render --save-every 100

# Run human playable game
python agent.py --mode human

# Load a saved model and evaluate (rendered)
python agent.py --mode eval --load model/model_best_game_100.pth --render

# Or use eval.py for more detailed evaluation
python eval.py --model model_best_game_663.pth --episodes 10 --render --speed 15
```

**Model Loading & Evaluation:**

Added `eval.py` for standalone model evaluation:

```powershell
# Run 10 headless evaluation episodes with a specific checkpoint
python eval.py --model model_best_game_663.pth --episodes 10

# Watch the AI play (rendered) at slow speed
python eval.py --model model_best_game_663.pth --episodes 3 --render --speed 10

# Available models in your ./model/ folder (pick any):
#   model_best_game_663.pth  (achieved score 29 ± 3.6)
#   model_checkpoint_game_2000.pth
#   best_at_10_score_1.pth
```

Key behavior changes:
- The trainer will save checkpoints periodically in `./model/` (files like `model_checkpoint_game_50.pth`, `model_best_game_20.pth`, `best_at_10_score_3.pth`).
- If the agent achieves a record score >= `--best-at-10-threshold` within the first 10 games, a special checkpoint is saved for quick inspection.

Tuning tips:
- Use `--fast` to disable rendering for much faster training iterations.
- Experiment with `--hidden 256 128` to use two hidden layers (256 then 128 units).
- Adjust `--speed` to control frames-per-second when rendering.

Next steps I can do for you:
- Add an `eval.py` script that runs multiple evaluation episodes and reports mean/std scores.
- Add a small web UI or tkinter overlay for dynamic interactive controls.
- Add unit tests and a GitHub Actions workflow for CI.
