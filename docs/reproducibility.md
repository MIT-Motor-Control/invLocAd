# Reproduction guide

`invLocAd-publication` recomputes the paper's numerical figures, tables and statistics from the saved experimental measurements and simulations in the Zenodo DATA archive. It does not rerun the stochastic simulation sweeps, so its results are those of the saved simulations. The [README](../README.md#reproduce-the-paper) lists the commands and maps each figure to its analysis.

## Commands

| Command | Purpose |
|---|---|
| `invLocAd-publication prepare` | Check the DATA folder and write the analysis inputs to `data/publication/` |
| `invLocAd-publication verify-data` | Recheck every prepared input against its SHA-256 checksum |
| `invLocAd-publication reproduce --strict` | Write figures, CSV tables and a report to `outputs/publication/` |
| `invLocAd-publication validate-manuscript` | Compare the reproduced tables with the values reported in the paper |
| `invLocAd-publication recompute-s1 --matlab /path/to/matlab` | Rerun the Supplementary Figure 1 cross-validation in MATLAB |

`reproduce` exits with 1 on an input, output-folder or analysis error and, with `--strict`, with 2 if an analysis is partial or blocked by missing inputs; otherwise it exits with 0. Options: `--figures` selects analyses, `--output` sets the directory and `--overwrite` replaces an earlier run in it. `report.json` records the checksums of the input manifest, code and outputs and the software versions; `report.md` lists the status and caveats of each analysis.

`validate-manuscript` needs a complete `reproduce` run made with the current code and inputs. It writes `report.md` to an empty directory (`--output`, default `outputs/manuscript-validation/`; choose a new directory for a second run) with each checked value, its location in the paper and the reproduced value, and exits with 2 if any value differs. `tests/test_paper_values.py` runs the same comparison.

`recompute-s1` requires MATLAB with the Statistics and Machine Learning Toolbox; `reproduce` plots the saved results of the same MATLAB analysis.

MATLAB versions of the Supplementary Figure 2 and 4 analyses are in the same folder as the Supplementary Figure 1 scripts, `src/invLocAd/publication/vendor/`. They also need the Statistics and Machine Learning Toolbox, read the prepared inputs and write the same CSV tables as `reproduce`, with PNG figures:

```matlab
addpath('src/invLocAd/publication/vendor')
publication_s2_asymmetry('data/publication', 'outputs/matlab/s2')
publication_s4_fall_penalty('data/publication', 'outputs/matlab/s4')
```

Their tables agree with the Python tables to within 3 × 10⁻⁹, with identical model selections and test results; `tests/test_paper_values.py` checks this when MATLAB is available.

## Inputs

`data/publication/manifest.json` lists every input with its origin and SHA-256 checksum. Two realizations of the inverse-adaptation sweep are included. The **original** realization holds the fits reported in the paper. The **regenerated** realization is a second run of the same sweep, with its own fits and a trial-averaged fit grid; of its individual trials, it keeps the 50 at each of the 23 grid cells of the fits reported in the paper.

| Folder | Contents | Used by |
|---|---|---|
| `original/` | Experimental energy and step-length asymmetry, the fits reported in the paper, the fall-risk landscape, the Sánchez et al. (2021) comparison and trajectory-optimization outputs | Figures 1, 2C, 4; Supplementary Figure 3; `r2` |
| `regenerated/` | Experimental energy, the regenerated fits, the trial-averaged fit grid, cross-validation results and 50 simulations at each of the 23 grid cells of the fits reported in the paper | Figure 2B, 2C (model); Supplementary Figure 1 |
| `fall_probability/` | 3,092 simulations; Figure 3 uses trials 1–50 at ψ = 0 (1,750: 7 speed differences × 5 learning rates × 50 trials) | Figure 3 |
| `extensions/s2/` | Asymmetry measurements and model simulations | Supplementary Figure 2 |
| `extensions/s5/` | 750 fall-penalty simulation cells and their fits | Supplementary Figure 4 |
| `extensions/experimental_csv/` | The six experimental CSV tables of the DATA folder | Supplementary Figure 4; checks against the full-precision arrays |
| `reference_tables/` | Reference tables from the original analysis; only `expfit_r2.csv` is read | Check of the `r2` R² values |

## Methods of each analysis

**Figure 1C–E.** Energy change is split-belt minus tied-belt metabolic cost over the final 3 minutes of each condition, as a percentage of tied-belt cost; short exposure uses minutes 3–6 of split-belt walking. Step-length asymmetry change compares the final 100 split-belt and tied-belt strides; the short window ends at stride ⌈6/45 · N⌉, where N is the number of split-belt strides. Figure 1E compares adjacent 3-minute means with two-sided paired t-tests (p < 0.05, uncorrected). Box plots use MATLAB's percentile definition. The energy-minimization points are the five reported trajectory-optimization values; the saved OpenSim outputs cover four conditions (tied, 0.4, 0.7 and 1.0 m/s) and are used only for a separate check (`figure1_optimization_source_subset.csv`).

**Figure 2B.** The plotted trajectories are the means of the nine 0.7 m/s participants and of their nine fitted simulations (regenerated realization); the paper's caption describes a representative participant. Each trajectory of n samples is smoothed with a 100-sample moving mean, started at sample max(30, round(0.03 n)) to remove the startup transient, expressed as percent change from the mean of its next 120 samples and resampled to 270 points; an exponential is fitted to each group mean.

**Figure 2C.** For each participant, the model value is the median energy change of the nonfalling simulations among the 50 at the participant's fitted parameters, comparing the final 100 strides with tied-belt strides 31–70; participant values are averaged by condition. The experimental values are those of Figure 1C at 45 minutes.

**Figure 3.** Fall probability is the fraction of 50 simulations per cell that fall. Figure 3B shows the lowest center-of-mass height between consecutive midstances, which is reached at the transition between stances. Figure 3D fits a single exponential to the mean strides until a fall at each speed by unweighted least squares; an independent MATLAB fit agrees within 3 × 10⁻⁶ strides.

**Figure 4 and Supplementary Figure 3.** The saved 20 × 20 fall-risk landscape of each condition. Ellipses are centered on the median fitted learning rate and symmetry weight, with semi-axes equal to their standard errors; each annotation is the mean fall probability of the grid cells inside the ellipse. `figure4_parameter_tests.csv` compares the fitted parameters across conditions. `figure4_fall_probability_tests.csv` tests the Results statement that keeping the same parameters at a larger belt-speed difference would increase fall probability: for each participant, the fall probability at their fitted grid cell on their own condition's landscape is paired with that on each larger belt-speed landscape, and the pairs are compared with a one-sided Wilcoxon signed-rank test (0.4 → 0.7 and 0.4 → 1.0 m/s: 8 of 8 participants higher, p = 0.004; 0.7 → 1.0 m/s: 8 of 9, p = 0.010). Supplementary Figure 3 plots rows and columns of the same landscapes.

**In-sample fit error and R² (`r2`).** For each participant, the reported fit selects, among 20 × 20 parameter cells with 50 simulations each, the single simulation whose exponential fit is closest to the participant's (lowest RMSE); the reported RMSE is that minimum. R² is that of the single-exponential fit to the 3-minute smoothed experimental energy. A double-exponential comparison is also exported; because that fit is poorly conditioned, its R² can vary between runs in the eighth decimal place.

**Supplementary Figure 1.** Leave-one-subject-out cross-validation of condition-specific versus size-matched common parameters on the regenerated trial-averaged grid, which scores each cell by the mean normalized trajectory of its nonfalling trials (at least five). The p-value shown is MATLAB's `signrank`; SciPy's exact value is exported alongside.

**Supplementary Figure 2.** Step-length and step-time asymmetry from the 20 participants with force-plate data (7, 7 and 6 by condition) and five Python-model simulations per condition with one parameter set for all conditions (learning rate 6 × 10⁻⁵ on the Python model's scale, symmetry weight 1, no fall penalty). Time-course bands are ±1.96 standard errors; adapted values have Student-t 95% confidence intervals.

**Supplementary Figure 4 and the fall-penalty comparison.** The simulations were run with the Python model (learning rates 3 × 10⁻⁵ to 4.8 × 10⁻⁴; these are not on the scale of the manuscript engine's learning rates). All 750 saved cell means are refitted. For each condition, the model with a free fall-penalty weight (`C` in the tables) and the model without it (`E`) are selected by RMSE, with both refitting the learning rate and symmetry weight, and compared by leave-one-subject-out cross-validation and paired Wilcoxon tests. The noise floor is the RMSE between the exponential fit to each participant's trajectory and the exponential fit to the mean trajectory of the other participants in the condition.

## Source data workbook

The source data of the figures with Zenodo data statements (Figures 1C–E, 2C, 3C–D and Supplementary Figure 1) can be collected into one Excel workbook:

```sh
python scripts/build_source_workbook.py outputs/publication outputs/source-data.xlsx
python scripts/check_source_workbook.py outputs/publication outputs/source-data.xlsx
```

The builder copies the CSV values and records their checksums; the checker compares every workbook cell with its CSV.

## Not reproduced

- The trajectory optimization (OpenSim Moco) of the energy-minimization predictions.
- The stochastic sweeps and the inverse-adaptation fitting that produced the saved fits; new simulations with the MATLAB engine or the Python model are new realizations.
- Processing of the raw motion-capture and metabolic recordings, which are included in the DATA folder.
- Schematic panels and figure layout.
