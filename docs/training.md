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
- `--epochs` sets how many passes the update makes over each page (default 4).
- `--scale` sets the fixed scale by hand instead of measuring it (needs --fixed-scale).
- `--broken-margin` sets how much worse a broken game counts than the worst working copy (default 0.1).
- `--eval-every` sets how often the greedy eval runs on the test tasks (default every 10 rounds).
- `--async-play` (needs --copies) lets each worker move on as soon as it answers, instead of every game waiting for the slowest worker on every tick (a game's last move, which routes and scores, or a reset can take 100+ ms while a normal move takes ~3 ms). Same jobs, same weights per page; only the order of the random draws changes.
- `--overlap` plays the next page in the background while the network learns from the current one. Each page is then played by weights one update old.
- `--lag` (with --overlap): `log` records how far apart the playing weights and the current ones are (lag_kl, lag_clip, clip_frac_played); `fix` also corrects for it with decoupled PPO (the clip is measured from the current weights, and each move is weighted by p_current / p_played, capped at 2).
- `--max-k K` (with --copies, 2 <= K <= copies) grades each game for the best of K copies instead of the average: a game's raw grade is how far it beats the best of K-1 random other copies (0 if it doesn't), which is its share of the gradient of E[best of K]; the raw grades are then compared with the other copies' like the normal grade. Tried on stage 5 (see below), not used since.
- `--stop-at N` stops after N rounds but keeps the lr schedule of --rounds (for short tests that match a full run's first N rounds).

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
| stage3_b | stage1_h | e's recipe | 0.460 | 0.632 | Learned all run (greedy 0.296 -> 0.50). Above A, 0.020 below B |
| **stage3_f** | stage3_d softened | softened start + cost-map input | **0.561** | **0.649** | Best-of-64 +0.050 vs A (helps), -0.004 vs B (neutral). Specialist |
| stage3_h | stage3_f softened + widened | n head widened to 16 machines | 0.530 | 0.633 | One best-of-64 layout broken (task 56030) |
| stage3_i | stage3_f softened + widened | h with seed 1 | 0.497 | 0.649 | The break in h was luck. Same best-of-64 as f, worse greedy |

Notes:
- **Building from a previous model's weights does work, once softened:** a sure model plays the same moves in all 8 copies, so the few moves that differ get big grades and the update jumps around. `python -m layout.soften` divides the output logits by a temperature, chosen so the entropy on the new stage's tasks is ~1.34, before the run starts.
- **Cost map:** `--cost-map` gives the network a map of what each position would cost. It went into stage3_f and has been on since.
- **Widening:** `python -m layout.widen` copies the n head's row for 8 machines into rows 9-16, so the model can place blocks of up to 16 machines. Stage 4 needs blocks of 9-10.
- **Run-to-run noise:** h and i differ only in seed. Their greedy differs by 0.033 and their best-of-64 by 0.016, so one run is not enough to judge small changes.

## Stage 4: (Items - heavy modular frame, motor)

Reference scores: A 0.532, B 0.569 (88 tasks). Games are about twice as long as stage 3.

| run | init | change | greedy | best-of-64 | notes |
|---|---|---|---|---|---|
| stage4_a | stage3_i softened on stage 4 | stage 3 recipe (lr 2.5e-4) | 0.349 | 0.541 | Drifted: entropy +0.75 by round 110, eval fell. 10 broken greedy layouts |
| stage4_b | stage3_f softened + widened | same | | | Same drift (entropy ~2.8 by round 200, eval 0.318). Stopped at round 229. ~14 s per round |
| **stage4_c** | stage4_b's start | lr 1e-4 + async play + overlap + lag fix | 0.458 | 0.594 | No drift. Best-of-64 beats A (+0.061) and B (+0.025) |
| **stage4_d** | stage4_b's start | c with 16 copies x 7 tasks (same 112 games) | **0.517** | **0.594** | Same best-of-64 as c. Only 2 broken greedy layouts (c: 7). Specialist |

220-round tests (from stage4_b's start, stopped with `--stop-at`):

| test | s / round | entropy (r190-219) | eval (r150-210) | result |
|---|---|---|---|---|
| sync (stage4_b) | 13.9 | 2.80 | 0.402 | baseline |
| `--async-play` | 9.7 | 3.24 | 0.298 | 30% faster, drift judged one-seed noise |
| async + `--overlap --lag fix` | 8.5 | 2.79 | 0.404 | 40% faster, same as baseline. Chosen |
| async + `--overlap` (no fix) | 8.6 | 2.27 | 0.453 | Stale pages shrink the step, so it drifted less |
| lr 1e-4 (with the speedups) | | -0.23 entropy rise | 0.464 | Stopped the drift. Chosen |
| 2 epochs | | | 0.424 | Held, but didn't improve |
| entropy bonus 0.003 | | ~0.5 | 0.35 | Collapsed |

Notes:
- **Why stage 4 drifted:** the updates were too big for it. At lr 2.5e-4, approx_kl was ~0.10-0.14 and clip_frac ~0.28 every round. lr 1e-4 fixed it.
- **Speed:** a sync round waited for the slowest game on every tick, and the last move of a game (route + score) takes 100+ ms. `--async-play` lets each worker move on when it answers (a page takes ~3.4 s instead of ~7.4 s). `--overlap` plays the next page in the background while the network learns.
- **Overlap with group training needs the lag fix:** the page is played by weights one update old. Plain PPO measures the clip from those old weights, so part of the clip room is used up before learning starts. `--lag fix` uses decoupled PPO (Hilton et al. 2021): the clip is measured from the current weights, and each move is weighted by p_current / p_played, capped at 2. This is the overlap that failed in smoke_j, now working.
- **Overlap slowdown (fixed):** the first overlap runs got slower over time within one process (eval 15 s -> 161 s by round 1600, RAM falling ~1 GB per 100 rounds). A new background thread per page leaked ~0.3 MB of native memory each time it ran the policy on CUDA, which fragmented the memory heap. One thread is now reused for every page. Eval stays at ~12-13 s.
- **16 copies vs 8:** the same number of games split into fewer tasks with more copies each. Each game is graded against 15 others, so the grade is less noisy. Greedy improved (0.517 vs 0.458, 1.9 SE, just short of the 2 SE rule), best-of-64 didn't change.
- **Stage 4 is cleared at the same level as stage 3:** best-of-64 beats A and B, greedy is level with A (-0.015, inconclusive). Greedy is still 0.052 below B.

Current recipe (stage4_d):
```
python -m layout.train --run stage4_d --stage 4 --rounds 2000 --workers 16 --init runs/other/f_s4soft_wide.pt --ent-coef 0.01 --solvable --copies 16 --groups 7 --fixed-scale --scale 0.056543 --cost-map --async-play --overlap --lag fix --seed 0 --lr 0.0001 --note "..."
```

## Stage 5: (Items - computer)

Reference scores: A 0.511, B 0.539 (45 tasks). Games have ~14 blocks (stage 4: ~12). The test set is half the size of stage 4's, so the SEs are ~1.4x bigger.

All runs start from stage4_d's last.pt, softened by T 2.5 (entropy 1.42 on stage 5 tasks), with stage4_d's recipe (16 copies x 7 tasks, async play, overlap, lag fix).

| run | change | greedy | best-of-64 | notes |
|---|---|---|---|---|
| stage5_a | stage4_d's recipe (lr 1e-4) | | | Drifted: entropy flat for 150 rounds, then +0.32 by round 220 (approx_kl ~0.1, clip_frac ~0.23). Killed |
| stage5_b | a + `--max-k 4` | | | Drifted faster: entropy +0.49 by round 50. Killed |
| **stage5_c** | a with lr 5e-5 | **0.528** | 0.597 | No drift. Best-of-64 beats A (+0.087, all 45 tasks) and B (+0.059, 44 of 45). Greedy level with B. Specialist |
| stage5_d | c + `--max-k 2` | 0.419 | **0.602** | Best-of-64 +0.005 vs c (1.9 SE, neutral). 6 broken greedy layouts (c: 1) |

Notes:
- **Each stage needed a smaller step:** lr 2.5e-4 worked to stage 3, stage 4 drifted at it and needed 1e-4, and stage 5 drifted at 1e-4 and needed 5e-5. At 5e-5, approx_kl was ~0.05-0.08 early and shrank all run.
- **Greedy looks flat for a long time:** stage5_c's greedy eval rose only ~0.02 in the first 800 rounds, then 0.44 -> 0.49 as the lr decayed. With 45 tasks, one eval swings +-0.04, so only 100+ round averages show a trend.
- **Max-k (training for best-of-k) didn't pay off:** at k 4 it drifted at once: grading only the top games pushes probability off all the middle moves, which spreads it out. At k 2 with the smaller lr it trained fine, but best-of-64 only moved +0.005 (neutral), and greedy broke on 6 tasks. On tasks where it didn't break, its greedy was +0.010 better than c's, so the problem is breakage, not layout quality. Training on the average is not what limits best-of-k here.
- **Stage 5 is the best result vs B so far:** best-of-64 beats B on 44 of 45 tasks (stage 4: 63 of 88), mostly with shorter belts (618 vs 743 m), fewer floors (2.2 vs 2.7) and fewer crossings (1.1 vs 2.0), at the cost of more lifts (22 vs 15). Greedy alone is -0.011 vs B (inconclusive); on stage 4 it was -0.052 (worse).

Final recipe (stage5_c):
```
python -m layout.train --run stage5_c --stage 5 --rounds 2000 --workers 16 --init runs/other/d4_s5soft.pt --ent-coef 0.01 --solvable --copies 16 --groups 7 --fixed-scale --cost-map --async-play --overlap --lag fix --seed 0 --lr 5e-05 --note "..."
```
`runs/other/d4_s5soft.pt` is `python -m layout.soften runs/stage4/stage4_d/last.pt --stage 5 --temps 1.0 1.5 2.0 2.5 3.0 3.5 --target 1.34 --save runs/other/d4_s5soft.pt`.