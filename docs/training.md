# RL training notes

Notes on how I trained the agent locally, including the setup, how results are measured, every run with what changed, and what each run taught me.

## Setup

**Task:** place blocks (rows of identical machines) one at a time on a site of up to 96 x 96 m with 3 floors. A router then connects them and the layout gets a score.

```
score   = 1 / (1 + penalty)
penalty = 0.35 belt + 0.15 lifts + 0.15 crossings + 0.2 sprawl + 0.15 floors
```
Broken layouts (ran out of room, or flow doesn't reach the exit) get score <= 0.

**Baselines:**
- A: greedy heuristic, ~35 ms per task.
- B: A + 1000 tries of simulated annealing, 4-31 s per task.

**Policy:** 1.35M parameters
- CNN over the site grid (3 floors)
- Graph network over the recipe graph
- U-Net decoder that outputs one logit per position on a 2 m grid (27,648 positions)
- Each block is 2 moves: first the number of machines, then rotation + floor + position
- Illegal moves are masked out

## Performance Evaluation

- Fixed test set of 500 tasks across 5 stages (each increasing in factory size and difficulty), never trained on
- Only tasks Baseline A can solve are counted (some sites are too small for any method)
- **greedy:** always take the most likely move, one try per task
- **best-of-k:** sample k layouts per task and keep the best. This will be the method the final agent uses
- Models are compared to baselines task by task

## Training method

Call python -m layout.train from root. Many different hyperparameters that can be tweaked (all can be found in main() in layout/train) Example call:
```
python -m layout.train --run stage3_b --stage 3 --rounds 2000 --ent-coef 0.01 --workers 16 --solvable --copies 8 --groups 14 --fixed-scale --init runs/stage1/stage1_h/last.pt --note "stage2_e recipe on stage 3 from stage1_h weights"
```
**Quick description of the above call**: --run gives it a unique name, --stage picks the curriculum stage (training tasks only use that stage's target items, and the eval uses that stage's test tasks), --rounds picks how long it trains for (the learning rate decays linearly to 0 over these rounds), --ent-coef decides how much of a boost the model should get for having a higher entropy (useful to encourage the model to try new things early), --workers picks how many parallel CPU processes to run with, --solvable declares that it will only be trained on tasks that have a proven solution (Baseline A can build a working layout for them), --copies 8 plays every task 8 times and grades each game against the average of its other 7 copies (group-baseline training), --groups 14 is how many different tasks go into one round (14 tasks x 8 copies = 112 games per round), --fixed-scale divides all grades by one number for the whole run (measured on the first round) instead of rescaling every batch, --init starts the run from an existing model's weights (the optimizer and learning rate start fresh), and --note is a required short description of what the run changes, saved with the run.

**Other useful flags:**
- `--resume` continues a stopped run from its last checkpoint. It needs the exact same settings as the original run, without --init, --note or --scale.
- `--lr` sets the starting learning rate (default 2.5e-4).
- `--scale` sets the fixed scale by hand instead of measuring it (needs --fixed-scale).
- `--broken-margin` sets how much worse a broken game counts than the worst working copy (default 0.1).
- `--eval-every` sets how often the greedy eval runs on the test tasks (default every 10 rounds).

**Output**:
- `log.csv`: one row per round (scores, penalty parts, entropy, KL, timings).
- `eval.csv`: test-set results every eval.
- `last.pt` and `best.pt` checkpoints.
- `run.json`: the command, the note and the git commit.
- TensorBoard logs: `tensorboard --logdir runs`.

To score a checkpoint with greedy and best-of-k on the test set:
```
python -m layout.best_of runs/stage3/stage3_b/last.pt --stage 3 --k 64 --save plans.json
```

## Stage 1: (Items - iron plate, iron rod, wire)

Reference scores: A 0.726, B 0.757 (144 tasks).

| run | change | greedy | best-of-64 | notes |
|---|---|---|---|---|
| stage1_a / b | first runs: value head + GAE, constant vs decaying lr | ~A level | Not tested | First block on floor 2 in 26/28 tasks ("start upstairs" habit) |
| stage1_c | entropy bonus 0.03 | ~0.720 | 0.732 | Habit stayed |
| stage1_d | PPO from a clone of Baseline B | 0.745 | 0.758 | Side experiment only. Decided copying baseline is cheating |
| stage1_e | c + train only on solvable tasks | 0.738 | 0.751 | +0.013 vs c. Habit halved |
| stage1_f | e + reward shaping based on belt length + overlapped CPU & GPU for speed | | | -0.010 vs e. Habit moved to floor 1 |
| stage1_g | group baseline, grades normalised per batch | 0.737 | -0.011 vs e | Collapsed: entropy 0.23, 60% of groups all copies identical |
| smoke_j | group baseline + overlap | | | Policy jumped (KL 0.69). Overlap dropped |
| stage1_h | group baseline + fixed scale | 0.760 | 0.769 | Beats B. First block on floor 0 in 79/144 without any fix for it |

Notes:
- **Value head noise:** it was off by ~0.040, while the moves I wanted learned change the score by ~0.017. Most of the signal was noise. That's why I switched to the group baseline.
- **Why per-batch normalising collapsed:** as the copies agree more, their differences shrink, but normalising scales them back up to full size, so sharpening never slows down. The fixed scale fixes this.

## Stage 2: (Items - reinforced iron plate, rotor)

Reference scores: A 0.605, B 0.658 (83 tasks).

| run | init | change | greedy | best-of-64 | notes |
|---|---|---|---|---|---|
| stage2_b | stage1_e | old method, entropy 0.03 | | | Drifted (entropy up, layouts messier) |
| stage2_c | stage1_e | old method, entropy 0.01 | 0.560 | 0.597 | Below A |
| stage2_d | stage1_h | group method, entropy 0.03 | | | Drifted: entropy 1.9 -> 2.9 in 50 rounds |
| stage2_e | stage1_h | group method, entropy 0.01 | 0.607** | 0.654 | Above A, 0.004 below B |
| stage2_f | stage1_h | e's method over 3000 rounds | 0.580 | 0.650 | Not better than e |

Notes:
- **Longer training doesn't help:** plotted against the learning rate, the f and e eval curves line up. The gain at the end of a run comes from the lr decaying, not from extra rounds.
- **Remaining gap vs A:** belt length and lifts, mostly on bigger factories.


## Stage 3: (Items - modular frame, encased industrial beam)

Reference scores: A 0.599, B 0.653 (90 tasks).

| run | init | change | greedy | best-of-64 | notes |
|---|---|---|---|---|---|
| stage3_a | stage2_e | lr 1e-4 | | | Got worse early. Stopped at round 60 |
| **stage3_b** | stage1_h | e's recipe | **0.460** | **0.632** | Learned all run (greedy 0.296 -> 0.50). Above A, 0.020 below B |

Notes:
- **Building from stage 2s weights does not yet work:** current hypothesis is that the model is already sure of itself, and a sure model plays the same moves in all 8 copies during the group baseline. The few moves that differ get big grades, and the update jumps around instead of improving

### To be continued...