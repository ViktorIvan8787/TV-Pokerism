# MCCFR solver

Trains a heads-up strategy for this repo's game (10/20 blinds, 500 chip stacks)
and stores it in memory-mapped files. Run everything from the repo root.

```sh
# Build (g++ with C++20; OpenMP's runtime is not needed)
mkdir -p solver/build solver/out
g++ -std=c++20 -O3 -march=native -fopenmp-simd -static -pthread solver/main.cpp -o solver/build/cfr.exe

# 1. Card abstraction: about 1.5 minutes, 214 MB
solver/build/cfr.exe abstract --out solver/out/abstraction.bin

# 2. Train. Saves every 10 minutes and at the end; add --resume 1 to continue
solver/build/cfr.exe train --abs solver/out/abstraction.bin --out solver/out/strategy.bin --iters 500000000 --epoch 1000000

# Live turn and river solver (optional: without it the bot plays the blueprint)
g++ -std=c++20 -O3 -march=native -fopenmp-simd -fno-trapping-math -shared -static -pthread solver/subgame.cpp -o solver/build/subgame.dll

# Background patching (optional: SolverBot(patch=True), or solver.bot:PatchingBot)
g++ -std=c++20 -O3 -march=native -fopenmp-simd -shared -static -pthread solver/patch.cpp -o solver/build/patch.dll

# 3. Measure
solver/build/cfr.exe exploit --abs solver/out/abstraction.bin --ckpt solver/out/strategy.bin.ckpt --iters 30000000
.venv/Scripts/python -m evals --model solver.bot:SolverBot
```

Options: `abstract --flop --turn --river` (buckets, 2-256, default 200),
`train --stack --sb --bb --threads --seed --save-seconds`. Keep `--epoch` the
same across a resumed run: it sets the linear weighting's step.

Training replaces `strategy.bin`, which Windows refuses while a reader has it
open. It then warns and leaves the latest copy in `strategy.bin.tmp`.

## Files

| File | What it does |
|---|---|
| `cards.hpp` | 7-card evaluator, suit-isomorphic board indexing, RNG, threads |
| `abstraction.hpp` | equity features, EMD k-means, bucket tables |
| `tree.hpp` | betting tree: fold, call, 0.5x pot, 1x pot, all-in |
| `mccfr.hpp` | external-sampling MCCFR, linear weighting, best response, saving |
| `main.cpp` | command line and self-check |
| `reader.py` | looks up the strategy, standard library only |
| `translation.py` | off-tree bet mapping, state tracking, deviation matrix |
| `subgame.cpp` | real-time turn and river solver: exact equities, DCFR |
| `subgame.py` | ctypes bindings, blueprint ranges, re-solving during a hand |
| `patch.cpp` | solves one repeated spot with MCCFR, for the patcher |
| `patcher.py` | finds repeated spots, solves them in a thread, swaps patches in |
| `bot.py` | plays the strategy in the local engine |

Tests: `.venv/Scripts/python -m pytest tests/test_solver.py -q`

## How it works

- **Infosets** are direct array positions: `node.offset + bucket * node.nact`.
  The betting history is the tree node and the cards are the bucket. There is no
  hashing, and the tables for 200 buckets take 57 MB.
- **Buckets:** preflop is the lossless 169 classes. The river clusters EHS. The
  flop and turn cluster the distribution of final equity, using L1 distance
  between CDFs, which is the 1-D earth mover's distance. Recall is imperfect:
  each street's bucket ignores earlier streets' buckets.
- **Self-play** is lock-free: threads share the tables through relaxed atomics,
  and a racing update can be lost. Threads join after each epoch for the linear
  CFR discount.
- **Vectorisation** applies to k-means distances and the discount pass. A
  traversal is recursive over 2-5 actions per node, so it has no useful matrix
  form; its speed comes from the flat tables.

## What exploitability means here

`exploit` trains a best response against the average strategy and reports what
it wins. That is a lower bound, and only inside the abstraction: the responder
uses the same buckets and bet sizes and trains for a limited number of
iterations. It tracks convergence. It does not measure distance from true
equilibrium in the full game.

## Live solving on the turn and river

From our first turn or river decision, `SolverBot` stops using the blueprint and
solves the exact state again at each of its decisions: exact board, pot, stacks
and bets, with no card abstraction. Both ranges start as what the blueprint would
have played on this path. After that they narrow by the solved strategies, and an
opponent's off-tree raise narrows by its mapped action. A failed solve falls back
to the blueprint, or to `_off_tree` once a solve has been played.

- **Budget:** `SolverBot(budget_ms=200)`. The tree is chosen by the budget (about
  1 MB of tables per 16 ms), going down the `LADDER` in `subgame.cpp` as ranges
  widen. Check `Result.ladder_level` before reading much into exploitability: it
  is measured within that tree, and most turn solves use a lower rung.
- **Threads:** half the logical CPUs by default, measured on a 4P+4E laptop, where
  8 spinning threads did far worse than 4. A river solve uses one thread.
- **Memory:** 48 MB reserved and touched when the solver is created, and more only
  if a solve needs it. The iterations allocate nothing (`Result.loop_news`).

This is unsafe re-solving: it assumes the estimated ranges are right, and has no
gadget protecting against an opponent who deviated earlier in the hand. So the
result is an equilibrium of the subgame given those ranges, not a guarantee in the
whole game.

The library cannot run in the arena, which takes only `.py` files and a pickle.
Local play is the only place it runs, for now.

## Background patches

`SolverBot(patch=True)` counts each opponent raise by blueprint node and size bin
(bins span 15%). A size at least 5% of the pot from any tree size is queued after 8
hits, a size on the tree after 40. A worker thread solves one spot at a time, with
finer sizes on that street (0.33, 0.66, 1x pot to open, 0.5, 1x facing a bet, and
the opponent's real size at the root), and the patch replaces the blueprint from
that bet onward. The card buckets stay the same, so only the bet sizes get finer.

- **The gate:** a patch is published once the regret bound is below 15% of the pot
  (`regret_fraction`). It is the sum of positive regret over infosets per iteration,
  an upper bound that sampling noise inflates. A spot that misses it within 300 s
  or 20M iterations is dropped and not retried.
- **The swap:** `_published` is a read-only table of immutable patches, replaced
  whole by one reference store. A hand reads it once, at its start, so a swap only
  affects later hands, and a table a hand holds stays alive by reference. The
  native job belongs to the worker, and the result is copied out of it.
- **The game loop:** native calls release the GIL and run below normal priority.
- **Memory:** at most 64 MB of tables during a solve and 8 MB per patch, 64 patches.

Only raises trigger patches, not checks or calls. With the live solver on, turn and
river raises never reach the patcher, because the solver already answers them
exactly, so patches apply preflop and on the flop. The opponent's range comes from
the blueprint, so this is unsafe re-solving with the same limits as the live solver.

`evals` rebuilds the model for each seat rotation, so patches do not carry between
rotations. A worker from an earlier rotation still finishes its queue. The module
uses `threading` and `ctypes`, and a pickle cannot hold a started patcher: build the
pickle before the first hand. The library cannot ship to the arena.

## Limits

- The trained tree assumes equal 500-chip stacks. Other stacks need their own
  `train --stack` run, and `bot.py` only clamps sizes to what is legal.
- `bot.py` cannot be submitted as it is. The arena takes `.py` files and a pickle
  of at most 50 MB, with no folders, and the abstraction file is 214 MB
  (`arena/vet.py`).

## Off-tree bets

`SolverBot(mapping="threshold")` is the default; `"geometric"` is the other
option. Sizes are fractions of the pot after calling, and a check or call counts
as size 0, so a tiny bet can be read as no bet.

- **`geometric`:** randomized geometric mapping between the two nearest sizes,
  even odds at their geometric mean. Below the smallest size it uses
  pseudo-harmonic, because the geometric formula always picks the larger size
  when the smaller is 0.
- **`threshold`:** pseudo-harmonic with `snap=0.8`. A size at least 80% likely to
  map to one neighbour goes there outright; the rest stay randomized.

An opponent's all-in always maps to the abstract all-in. Our own raises are
stored as pot fractions and rebuilt on the real pot. If a mapped bet ends the
abstract hand early, the bot plays by bucket strength against the pot odds until
the hand ends.

`SolverBot.deviations` counts each team's raises by street and fit to the tree.
`flags(team)` reports `frequent` (off-tree share confidently above 20%) and
`erratic` (normalised entropy of 0.7 or more), once a street has 30 raises;
`report()` prints the table. It only reports: the bot's play does not change. It
also lives in the bot object, so `evals` and arena workers do not print it.
