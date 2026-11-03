# Model guide

The model has a low-level feedback controller that sets each step from the center-of-mass state and a high-level learner that updates the controller by gradient descent on an energy–symmetry objective (Seethapathi et al., 2024; paper Figure 2A). Three implementations are included.

| | Python model | MATLAB baseline | MATLAB manuscript engine |
|---|---|---|---|
| Location | `src/invLocAd/` | `matlab/src/` | `matlab/manuscript/` |
| Use | Run, pause and resume simulations; simulations of Supplementary Figures 2 and 4 | Reference for the Python model | Simulations of Figures 2–4 and Supplementary Figures 1 and 3 |
| Split-belt protocol | 6 min tied, 45 min split, 5 min tied; 15 s ramps | Same as Python | 1 min tied, 45 min split; 25 s ramp |
| Belt speeds | 1.0 m/s difference by default (`--split-speed-diff`) | 1.0 m/s difference, fixed in `makeTreadmillSpeed_Split.m` | Experimental speeds × 0.5 |
| Falls | Not detected | Not detected | Detected; a fall ends the trial |

Both MATLAB folders define functions with the same names, so add only one of them to the MATLAB path. The [manuscript engine guide](../matlab/manuscript/README.md) lists every setting that differs between the engine and the baseline. Parameter values are not interchangeable between them: for example, the manuscript sweeps use learning rates of 0.005–0.05, whereas the baseline default is 1.2 × 10⁻⁴. The MATLAB baseline runs with `addpath('matlab'); run_simulation`, which uses a clock-based random seed.

## Parameters

| Paper | Python (`param_fixed`) | Manuscript engine (`run_manuscript_trial`) |
|---|---|---|
| Learning rate α | `["Learner"]["LearningRate"]` | `learning_rate` |
| Symmetry weight ψ | `1 - lambdaEnergyVsSymmetry` | `1 - energy_weight` |
| Symmetry scale M = 10 | `symmetryMultiplier` | fixed at 10 |
| Fall penalty weight w | `fallPenaltyWeight` (default 0) | not available |
| Exploration noise | `["Learner"]["noiseSTDExploratory"]` | `exploration_noise` |
| Belt speed difference Δv | `--split-speed-diff` | `speed_difference` |

The objective is J₀ = (1 − ψ) e + ψ M s (Eq. 1), where e is the metabolic power over a learning iteration and s is the squared difference between the stance-end leg angles of its two steps. With w > 0 the Python model minimizes J₀ + w Φ (Eq. 2). Φ is the mean, over the stance phases of the iteration, of min(max((h_safe − h_min) / (h_safe − h_ground), 0), 1)², where h_min is the lowest center-of-mass height in the stance phase, h_safe = 0.95 and h_ground = 0.90 leg lengths; Φ is 0 while h_min ≥ h_safe and 1 when h_min ≤ h_ground. With w = 0 the penalty is not evaluated and the objective is J₀.

## Python model

```sh
invLocAd-simulate demo
invLocAd-simulate run --seed 42 --iterations 1000 --output-dir outputs/example
invLocAd-simulate run --split-or-tied tied --seed 1 --iterations 200 --output-dir outputs/tied
```

`run` writes `run.npz` (settings and controller, body-state, energy and learner histories), `metadata.json` (seed, condition, protocol, configuration checksum, software versions and iteration counts), a checkpoint and a summary figure. The full split-belt protocol has 3,200 iterations and the tied protocol 1,000. Other options include `--transition-time` (seconds) and `--speed-protocol`, which takes a quoted name: `classic split belt` (default) or `single speed` for split runs, and `four speed changes` (default) or another protocol listed in `parameter_loading/make_treadmill_speed_tied.py` for tied runs. See `invLocAd-simulate run --help` for all options. An omitted seed is generated and recorded.

### Pause and resume

```sh
invLocAd-simulate run --seed 42 --iterations 100 --stop-after 40 --output-dir outputs/paused
invLocAd-simulate resume outputs/paused/checkpoint.npz --output-dir outputs/resumed
```

`--iterations` is the total number of learning iterations and `--stop-after` the iteration at which to stop; both count completed iterations. A checkpoint is saved every 25 iterations (`--checkpoint-every`) and at the end. It stores the body state, treadmill time, controller, gradient estimate, memory, estimator history, all recorded histories and the random-number-generator state, so a resumed run is identical to an uninterrupted run. `resume` checks the model version, the NumPy and SciPy versions, the settings and the seed, and refuses a checkpoint that does not match. It writes to `--output-dir` (default `outputs/resumed`) and leaves the input checkpoint unchanged unless `--checkpoint` names that file. Results and checkpoints are NumPy archives read with `allow_pickle=False`.

The DATA download includes `checkpoints/example_seed42.npz`, a run paused after 10 of 20 iterations that resumes with NumPy 1.26.4 and SciPy 1.17.1, the versions pinned in `requirements-lock.txt`:

```sh
invLocAd-simulate resume data/invLocAd-DATA/checkpoints/example_seed42.npz --output-dir outputs/checkpoint-example
```

### Python API

```python
from invLocAd import load_simulation_results, plot_simulation_results, run_simulation

result = run_simulation(seed=42, num_iterations=100, make_plots=False, output_dir="outputs/run")
saved = load_simulation_results("outputs/run/run.npz")
print(saved.summary["average_energy_rate"])
plot_simulation_results(saved, output_dir="outputs/replotted")
```

`run_simulation` uses its own random generator and leaves NumPy's global random state unchanged. It uses the default values of α, ψ and w set in `parameter_loading/load_learner_parameters.py` and `load_biped_model_parameters.py`; edit those defaults to simulate other values.

## MATLAB manuscript engine

From the repository root in MATLAB (R2021a or later):

```matlab
addpath('matlab/manuscript')
result = run_manuscript_trial('outputs/trial42.mat', seed=42, speed_difference=0.7, ...
    learning_rate=0.02, energy_weight=0.5, max_iterations=40);
```

Omit `max_iterations` to run the full protocol. The [manuscript engine guide](../matlab/manuscript/README.md) describes the saved fields, parameter sweeps and validation. The DATA download includes `checkpoints/manuscript_seed42_trial.mat`, a trial with seed 42 and otherwise default options that ends with a fall at iteration 104; the guide explains how to load and regenerate it, and `tests/test_paper_values.py` checks the regeneration when MATLAB is available.

## Python and MATLAB agreement

With exploration and measurement noise set to zero, the Python model and the MATLAB baseline give the same objective for the initial controller, to within 6 × 10⁻¹², and the same controller history over 40 learning iterations:

```matlab
addpath('matlab/validation')
export_reference_run('outputs/invLocAd_reference.mat', 40)
```

```sh
python scripts/validate_matlab_python_parity.py outputs/invLocAd_reference.mat
```

Without noise the learner receives no gradient, so this comparison checks the stance dynamics, energetics and objective rather than learning; `tests/test_model_regression.py` pins the results of seeded runs with learning. Random seeds do not carry across languages, so noisy runs differ between Python and MATLAB.
