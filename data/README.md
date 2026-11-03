# Data

The experimental measurements and saved simulations are archived on [Zenodo](https://doi.org/10.5281/zenodo.19547419). The archive's own `README.md` describes file formats, units, participant groups and missing values.

Version 0.4.0 uses `invLocAd-DATA.zip`. Its Zenodo release is awaiting publication; the public link currently opens version 0.3.0, whose differently named DATA ZIP has a different checksum. Until publication, install a locally supplied version 0.4.0 ZIP with the command below.

## Install

With the `invLocAd` environment active, install a local ZIP from the repository root:

```sh
python scripts/download_data.py --archive /path/to/invLocAd-DATA.zip
```

After the matching release is published on Zenodo, download it directly:

```sh
python scripts/download_data.py
```

This downloads `invLocAd-DATA.zip` (13.27 GB), verifies its size and SHA-256 checksum, and extracts it to `data/invLocAd-DATA/` (14.88 GB). Allow 30 GB of free space. If the download is interrupted, rerun the command; it resumes when the server supports it. Downloads are cached in `outputs/data-download/23904047/` and removed after a successful installation unless `--keep-downloads` is given. An existing DATA folder is never overwritten; use `--destination` to install elsewhere.

The local-ZIP command applies the same checks without network access and leaves the ZIP unchanged. Extracting the ZIP manually into `data/` also works; keep the inner `publication-data.zip` compressed.

Then prepare the analysis inputs:

```sh
invLocAd-publication prepare
invLocAd-publication verify-data
```

`prepare` writes the 5,166 checked inputs and their checksums to `data/publication/`. For a DATA folder outside the repository, use `invLocAd-publication prepare --source /path/to/invLocAd-DATA`; the raw recordings are not copied. Repeating `prepare` with the same inputs is safe.

## Contents of the DATA folder

| Path | Contents |
|---|---|
| `processed_csv/` | Six CSV tables of energy, step-length, step-time and double-support asymmetry, one row per participant |
| `processed_mat/` | Per-participant MATLAB files of processed energy and motion-capture data |
| `raw_mocap_c3d/` | Raw motion-capture recordings (C3D) |
| `subject_information/` | Participant characteristics and condition assignments |
| `publication-data.zip` | Inputs of the figure workflow: published fits, saved simulations, cross-validation results and comparison data |
| `checkpoints/` | A paused Python run and a seeded manuscript-engine trial (see the [model guide](../docs/model.md)) |
| `validation/` | MATLAB and SciPy fits of Figure 3D for comparison |
| `SHA256SUMS` | Checksums of all other files |

The CSV tables are rounded to six decimals; `publication-data.zip` contains the full-precision arrays, and `prepare` checks that the two agree. Eight double-support asymmetry records lack their final sample.

Data are released under CC BY 4.0.
