# Release verification

Checks of this release on macOS 26.6 (arm64) with Python 3.12.13, the pinned packages of `environment.yml` and MATLAB R2025a. The automated tests also run on Ubuntu through GitHub Actions; Windows was not tested.

| Check | Result |
|---|---|
| Fresh Conda environment from `environment.yml` | Created in under a minute; `pip check` passed |
| `invLocAd-simulate demo`, pause and resume (README) | Completed; the resumed run equals the uninterrupted run |
| `python scripts/download_data.py --archive ...` | Size and SHA-256 verified; 105 files installed |
| `invLocAd-publication prepare` and `verify-data` | 5,166 inputs verified |
| `invLocAd-publication reproduce --strict` | All 9 analyses passed; 45 figure files and 62 tables |
| `invLocAd-publication validate-manuscript` | 27 of 27 values match the paper |
| `python -m pytest` with MATLAB | 308 passed |
| Manuscript engine regenerates `checkpoints/manuscript_seed42_trial.mat` | Identical |
| `validate_manuscript_model` (MATLAB) | Passed |
| Python and MATLAB baseline, noise-free | Initial-controller objective within 5.8 × 10⁻¹²; controller history over 40 iterations within 5.1 × 10⁻¹⁶ |
| Figure 3D fit, SciPy and MATLAB (`validate_fall_fit`) | Curves within 2.6 × 10⁻⁶ strides |
| MATLAB versions of Supplementary Figures 2 and 4 | Same tables as Python within 3 × 10⁻⁹; identical selections and p-values |
| Source data workbook | 30,099 cells match their CSV tables |

Repeated runs in one environment give identical tables, except the double-exponential columns of `r2_timescale_*.csv`, which vary by about 10⁻⁸ because that fit is poorly conditioned. Environments with differently built numerical libraries (for example, NumPy from conda-forge rather than from pip) agree to within 10⁻⁸; no reported value changes.

[`figure3-protocol-check.json`](figure3-protocol-check.json) records that all 1,750 saved Figure 3 simulations use a 25-second transition and belt speeds scaled by 0.5, as stated in the Methods. The transition and belt speeds in their file names (for example, `trans20_slow0.8_fast1.2`) are identifiers, not the simulated values.

## DATA archive

`invLocAd-DATA.zip`: 13,269,858,121 bytes, SHA-256

```text
16ce92fd0fac4fc5a3686e4bcd0180f80f5bd84a0ec928e95e8de4826cdb7ffd
```
