"""Check reproduced results and participant data against the values reported in the paper."""

import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from invLocAd.publication import cli, manuscript

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DATA = ROOT / "data/publication"
DATA = ROOT / "data/invLocAd-DATA"


@pytest.fixture(scope="module")
def reproduction(tmp_path_factory):
    manifest = PUBLICATION_DATA / "manifest.json"
    if not manifest.is_file():
        pytest.skip("Run invLocAd-publication prepare to check values against the paper")
    if json.loads(manifest.read_text()).get("dataset_kind") == "zenodo_experimental":
        pytest.skip("Model results require the companion publication-data.zip")
    output = tmp_path_factory.mktemp("paper") / "publication"
    status = cli.main(
        ["reproduce", "--strict", "--data", str(PUBLICATION_DATA), "--output", str(output)]
    )
    assert status == 0
    return output


@pytest.mark.integration
def test_every_reported_value_matches_the_paper(reproduction):
    report = manuscript.verify_reproduction(PUBLICATION_DATA, reproduction)
    claims = manuscript.numerical_claims(reproduction, report)
    assert len(claims) == 27
    mismatches = [
        f"{c['claim']}: observed {c.get('observed', c.get('reason'))}, paper {c.get('paper')}"
        for c in claims
        if c["status"] != "matched"
    ]
    assert not mismatches, "\n".join(mismatches)


@pytest.mark.integration
def test_validate_manuscript_command_reports_agreement(reproduction, tmp_path):
    output = tmp_path / "validation"
    status = cli.main(
        [
            "validate-manuscript",
            "--data", str(PUBLICATION_DATA),
            "--results", str(reproduction),
            "--output", str(output),
        ]
    )
    assert status == 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "matched"
    assert report["counts"] == {"matched": 27}


@pytest.mark.integration
def test_participant_characteristics_match_methods():
    workbook = DATA / "subject_information/subject_experiment_information.xlsx"
    if not workbook.is_file():
        pytest.skip("The DATA folder is required for participant characteristics")
    openpyxl = pytest.importorskip("openpyxl")
    sheet = openpyxl.load_workbook(workbook, read_only=True, data_only=True).worksheets[0]
    rows = [row for row in sheet.iter_rows(min_row=2, values_only=True) if row[0]]
    sex, age, mass, height, delta, right, left = (np.array(c) for c in list(zip(*rows))[1:8])
    assert len(rows) == 27
    assert np.sum(sex == "F") == 7
    assert [np.sum(delta == d) for d in (0.4, 0.7, 1.0)] == [9, 9, 9]
    np.testing.assert_allclose((left + right) / 2, 1.0)
    np.testing.assert_allclose(left - right, delta)
    assert (round(age.astype(float).mean(), 1), round(age.astype(float).std(ddof=1), 1)) == (
        22.5,
        3.4,
    )
    metres = height.astype(float) / 100
    assert (round(metres.mean(), 2), round(metres.std(ddof=1), 2)) == (1.73, 0.11)
    assert round(mass.astype(float).mean(), 1) == 74.1


def _matlab():
    executable = os.environ.get("MATLAB") or shutil.which("matlab")
    if not executable:
        pytest.skip("Set MATLAB to the MATLAB executable to run engine regressions")
    return executable


def _run_matlab(matlab, script, log):
    """Run a MATLAB batch script and return its exit code and console output.

    The output goes to ``log`` because MATLAB can start helper processes that keep a pipe open.
    """
    with log.open("w") as stream:
        completed = subprocess.run(
            [matlab, "-batch", script], stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    return completed.returncode, log.read_text()


@pytest.mark.integration
def test_manuscript_engine_regenerates_saved_trial(tmp_path):
    saved = DATA / "checkpoints/manuscript_seed42_trial.mat"
    if not saved.is_file():
        pytest.skip("The DATA folder is required for the saved manuscript-engine trial")
    matlab = _matlab()
    fields = (
        "'time','sla','sta','energy','speed','state','steptime_fast','steptime_slow',"
        "'steplength_fast','steplength_slow','param','fall_flag','min_vert_pos'"
    )
    script = (
        f"addpath('{ROOT / 'matlab/manuscript'}'); ref = load('{saved}'); "
        "options = namedargs2cell(ref.metadata.options); "
        f"result = run_manuscript_trial('{tmp_path / 'trial.mat'}', options{{:}}); "
        f"fields = {{{fields}}}; "
        "assert(all(cellfun(@(f) isequaln(ref.(f), result.(f)), fields)), "
        "'Regenerated trial differs from the saved trial');"
    )
    code, output = _run_matlab(matlab, script, tmp_path / "matlab.log")
    assert code == 0, output[-4000:]


def _same_table(first, second, tolerance):
    """Return whether two CSV tables have equal text cells and numbers within ``tolerance``."""
    with first.open(newline="") as a, second.open(newline="") as b:
        rows_a, rows_b = list(csv.reader(a)), list(csv.reader(b))
    if len(rows_a) != len(rows_b) or rows_a[0] != rows_b[0]:
        return False
    for row_a, row_b in zip(rows_a[1:], rows_b[1:], strict=True):
        for x, y in zip(row_a, row_b, strict=True):
            if x == y:
                continue
            try:
                if not math.isclose(float(x), float(y), rel_tol=tolerance, abs_tol=tolerance):
                    return False
            except ValueError:
                return False
    return True


@pytest.mark.integration
def test_matlab_supplementary_figure_scripts_match_python(reproduction, tmp_path):
    matlab = _matlab()
    vendor = ROOT / "src/invLocAd/publication/vendor"
    script = (
        f"addpath('{vendor}'); "
        f"publication_s2_asymmetry('{PUBLICATION_DATA}', '{tmp_path / 's2'}'); "
        f"publication_s4_fall_penalty('{PUBLICATION_DATA}', '{tmp_path / 's4'}');"
    )
    code, output = _run_matlab(matlab, script, tmp_path / "matlab.log")
    assert code == 0, output[-4000:]
    tables = sorted((tmp_path / "s2").glob("*.csv")) + sorted((tmp_path / "s4").glob("*.csv"))
    assert len(tables) == 14
    mismatched = [
        table.name
        for table in tables
        if not _same_table(table, reproduction / "tables" / table.name, 1e-8)
    ]
    assert not mismatched

