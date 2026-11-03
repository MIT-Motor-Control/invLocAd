# Manuscript simulation engine

`engine/` contains the fall-detecting MATLAB model used for the manuscript simulations. `run_manuscript_trial` and `run_manuscript_sweep` add explicit seeds and saved trials around it. The engine differs from the baseline model in `matlab/src` (see below); the [model guide](../../docs/model.md) maps its parameters to the paper's notation.

## Run one trial

From the repository root in MATLAB:

```matlab
addpath('matlab/manuscript')
result = run_manuscript_trial('outputs/trial42.mat', seed=42, max_iterations=40);
```

Omit `max_iterations` for the full protocol: 1 minute of tied-belt walking, then 45 minutes of split-belt walking that begins with a 25-second ramp. The iteration budget, 2,161 iterations of two steps each, assumes 2 model time units per step. Simulated steps are typically shorter, so a complete trial usually ends before the protocol does; `result.time` is the cumulative model time after each iteration, in units of √(1 m / g) ≈ 0.32 s. A model fall, detected when the point mass passes below ground level, ends the trial; `result.fall_flag` is then `[1 iteration]`, otherwise `[0 0]`. `help run_manuscript_trial` lists the saved fields.

`energy_weight` (default 0.5) weights the energy term of the objective; the manuscript symmetry weight is ψ = 1 − energy_weight. `transition_seconds` (default 25) and `speed_divisor` (default 2) set the ramp duration and the scaling from experimental to model belt speeds.

Each saved trial stores its options, MATLAB version, engine source text and final random number generator state. Calling `run_manuscript_trial` with an existing file reloads it only when the options and the engine source text, including comments, match; otherwise it raises `invLocAd:CheckpointMismatch`. The engine source text stored in the DATA file `checkpoints/manuscript_seed42_trial.mat` differs from `engine/`, so read that file with `load`; a new run with `seed=42` and otherwise default options reproduces its numerical results in MATLAB R2025a. The simulations reported in the paper were not seeded, so no seeded run reproduces a particular trial from the paper.

## Run a sweep

```matlab
run_manuscript_sweep('outputs/small_sweep', ...
    speed_differences=[0.4 0.7], learning_rates=0.02, ...
    energy_weights=0.5, trials=2, seed=42, max_iterations=40);
```

Trial `i` uses seed `seed + i - 1`. Rerunning reloads completed trials; an interrupted trial restarts from its seed, because the engine does not save partial trials. For iteration-level checkpoints, use the [Python model](../../docs/model.md). The default grid (3 speed differences × 20 learning rates × 20 energy weights × 50 trials) has 60,000 trials and is not needed to reproduce the figures.

## Validate

```matlab
validate_manuscript_model('outputs/validation/manuscript')
```

This checks seed repeatability, saved-trial reuse, rejection of mismatched settings and sweep restart, and the step-pair rounding of the iteration budget in `engine/loadProtocolParameters.m` and `engine/loadHowLongParameters.m`, which `run_sim.m` does not call.

## Differences from `matlab/src`

| Setting | `matlab/src` | Manuscript engine |
| --- | --- | --- |
| Belt speeds | predefined profile in `makeTreadmillSpeed_Split.m` | experimental speeds / `speed_divisor` |
| Tied-to-split ramp | 15 s | `transition_seconds` (25 s) |
| Protocol | 6 min tied, 45 min split, 5 min tied | 1 min tied, then `split_minutes` (45) min split |
| Iteration budget | protocol time / 1.7 time units per step, rounded to 100 steps (3,200 iterations) | protocol time / 2 time units per step, capped by `max_iterations` (2,161 iterations) |
| Model time unit | √(0.95 m / g) | √(1 m / g) |
| Learning rate, exploration noise, energy weight | 0.00012, 0.002, 0.75 | per trial (defaults 0.02, 0.01, 0.5) |
| Energy measurement noise | 0.0001 | 0.01 |
| Trust region | 0.25 × exploration noise | 0.5 × exploration noise |
| Memory-based learning | yes | no |
| Nominal belt speed in velocity feedback | −0.35 | −0.5 |
| Stance ODE tolerances | 1e-10 | 1e-8 |
| Fall detection | no | yes |

`engine/run_sim.m` sets every model, learner and protocol parameter. It does not call the `load*.m`, `makeTreadmillSpeed_*.m`, `errorFromMemoryCompute.m` or `gradientOfErrorFromMemoryCompute.m` files in `engine/`. `engine/source_manifest.json` gives the SHA-256 of each original source file and lists the changes.
