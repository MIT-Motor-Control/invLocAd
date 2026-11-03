# invLocAd

Code for **Safety prioritization shapes locomotor adaptation and performance** by Inseung Kang, Kanishka Mitra and Nidhi Seethapathi. The preprint has the earlier title *Fall risk-aware adaptation explains suboptimal locomotor performance*.

[Code and data (Zenodo)](https://doi.org/10.5281/zenodo.23904047) · [Preprint](https://doi.org/10.64898/2026.03.02.709033) · [Model guide](docs/model.md) · [Reproduction guide](docs/reproducibility.md)

## Contents

- **Python model** (`src/invLocAd/`, command `invLocAd-simulate`): the locomotor adaptation model of Seethapathi et al. (2024), in which a gradient-based learner updates a step-to-step feedback controller, with the optional fall penalty of the paper's Eq. 2. It runs, pauses and resumes simulations with the model's baseline settings.
- **MATLAB manuscript engine** (`matlab/manuscript/`): the model with fall detection and the protocol of the paper's Methods, used for the simulations of Figures 2–4 and Supplementary Figures 1 and 3; the Python model produced those of Supplementary Figures 2 and 4 and the fall-penalty comparison in the Results.
- **MATLAB baseline** (`matlab/src/`): the MATLAB version of the Python model, used to check it.
- **Figure workflow** (command `invLocAd-publication`): recomputes the paper's figures, tables and statistics from the saved measurements and simulations on Zenodo.

Requirements: [Miniforge](https://github.com/conda-forge/miniforge#install) (Conda), about 30 GB of disk space for the data, and optionally MATLAB R2021a or later for the MATLAB code. The release was tested on macOS (arm64) with MATLAB R2025a; the automated tests also run on Ubuntu.

## Install

Clone the repository and install its environment:

```sh
git clone https://github.com/MIT-Motor-Control/invLocAd.git
cd invLocAd
conda env create -f environment.yml
conda activate invLocAd
invLocAd-simulate demo
```

You can also download `invLocAd-CODE.zip` from Zenodo or use GitHub **Code → Download ZIP**. Extract it and open a terminal in the extracted repository before running the Conda commands above.

The demo runs 40 learning iterations in a few seconds and writes `outputs/demo/split_belt_summary.png`. Activate `invLocAd` in each new terminal before running the commands below.

## Run a simulation

```sh
invLocAd-simulate run --seed 42 --iterations 1000 --output-dir outputs/my-run
```

| Output | Contents |
|---|---|
| `run.npz` | Controller, body-state, energy and learner histories |
| `metadata.json` | Seed, condition, protocol, configuration checksum, software versions, iteration counts and summary values |
| `checkpoint.npz`, `checkpoint.metadata.json` | Learner state for resuming the run and its description |
| `split_belt_summary.png` | Summary figure (`tied_belt_summary.png` for tied walking) |

The full split-belt protocol has 3,200 iterations: 6 minutes of tied walking, 45 minutes of split-belt walking and 5 minutes of tied walking. To pause after 10 of 20 iterations and resume:

```sh
invLocAd-simulate run --seed 42 --iterations 20 --stop-after 10 --output-dir outputs/paused
invLocAd-simulate resume outputs/paused/checkpoint.npz --output-dir outputs/resumed
```

A resumed run is identical to an uninterrupted run with the same seed. These simulations use the model's baseline settings, not the participant fits of the paper. The [model guide](docs/model.md) maps the paper's parameters to the code and shows how to run the MATLAB manuscript engine with the paper's settings.

## Reproduce the paper

**1. Install the data** (~13 GB):

Download the matching DATA ZIP from Zenodo:

```sh
python scripts/download_data.py
```

To install a ZIP you already downloaded:

```sh
python scripts/download_data.py --archive /path/to/invLocAd-DATA.zip
```

Both commands verify the ZIP's checksum and extract it to `data/invLocAd-DATA/`. Rerun the download command to resume an interrupted transfer. The [data guide](data/README.md) describes the files.

**2. Prepare the inputs and reproduce the figures:**

```sh
invLocAd-publication prepare
invLocAd-publication reproduce --strict
```

`prepare` checks all 5,166 inputs against their SHA-256 checksums and writes them to `data/publication/`; for a DATA folder elsewhere, add `--source /path/to/invLocAd-DATA`. `reproduce` writes `outputs/publication/figures/` (PNG, PDF and SVG), `outputs/publication/tables/` (CSV source data) and `outputs/publication/report.md`. MATLAB is not needed.

**3. Compare the results with the paper:**

```sh
invLocAd-publication validate-manuscript
```

This checks 27 values reported in the paper, including sample sizes, test results, fit errors, fall probabilities and cross-validation results, and writes `outputs/manuscript-validation/report.md`.

| Paper | Analysis | Figures | Tables |
|---|---|---|---|
| Figure 1C–E | `figure1` | `figure1_cd`, `figure1_e` | `figure1_*` |
| Figure 2B–C | `figure2` | `figure2b_*`, `figure2c` | `figure2b_*`, `figure2c_*` |
| Figure 3B–D | `figure3` | `figure3_b`, `figure3_c`, `figure3_d` | `figure3_*` |
| Figure 4 | `figure4` | `figure4` | `figure4_*` |
| In-sample fit error (Results) and exponential-fit R² (Methods) | `r2` | `r2_expfit` | `inverse_fit_rmse_*`, `r2_*` |
| Fall-penalty comparison (Results) and Supplementary Figure 4 | `s4` | `s4_fall_penalty` | `s4_*` |
| Supplementary Figure 1 | `s1` | `s1_loso` | `s1_*` |
| Supplementary Figure 2 | `s2` | `s2_timeseries`, `s2_adapted` | `s2_*` |
| Supplementary Figure 3 | `s3` | `s3_fall_risk_slices` | `s3_*` |

Select analyses with `--figures`, for example `invLocAd-publication reproduce --figures figure3 s1 --output outputs/selected`. The workflow recomputes every analysis from saved measurements and simulations, except Supplementary Figure 1, which it replots from saved MATLAB cross-validation results (`invLocAd-publication recompute-s1` reruns them). It does not rerun the trajectory optimization, the stochastic parameter sweeps or the processing of raw motion-capture and metabolic recordings, and it does not draw schematic panels. The [reproduction guide](docs/reproducibility.md) describes the inputs and methods of each analysis.

## Repository map

```text
invLocAd/
├── src/invLocAd/
│   ├── learning/             Stance dynamics, objective and learner
│   ├── initializing/         Initial conditions and treadmill speeds
│   ├── parameter_loading/    Model, learner and protocol settings
│   ├── post_processing/      Replay, energetics and summary plots
│   ├── publication/          Figure analyses, data preparation and paper comparison
│   └── cli.py, workflow.py, artifacts.py   Commands, runs and checkpoints
├── matlab/
│   ├── manuscript/           Manuscript simulation engine
│   ├── src/                  MATLAB baseline model
│   ├── run_simulation.m      Run the MATLAB baseline model
│   └── validation/           Python/MATLAB comparison and fit checks
├── scripts/                  Data download, source-data workbook and Python/MATLAB comparison
├── tests/                    Unit, regression and paper-value tests
├── docs/                     Model and reproduction guides
├── reproducibility/          Verification record of this release
├── data/                     Downloaded and prepared data (not tracked)
└── outputs/                  Generated results (not tracked)
```

## Tests

```sh
python -m pytest -q
```

Tests that need the data skip when it is absent. With the data prepared, `tests/test_paper_values.py` reproduces all analyses and checks them against the paper; set the `MATLAB` environment variable to the MATLAB executable (with the Statistics and Machine Learning Toolbox) to include the MATLAB tests (manuscript-engine regression and the MATLAB versions of the Supplementary Figure 2 and 4 analyses).

## Citation and license

Please cite the paper and the Zenodo version of invLocAd used in your work ([CITATION.cff](CITATION.cff)). Code is released under the MIT License and data under CC BY 4.0. Contact: Kanishka Mitra, mitra819@mit.edu.
