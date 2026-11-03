"""Comparison of reproduced results with the numerical claims of the paper."""

import csv
import json

import pytest

from invLocAd.publication import manuscript
from invLocAd.publication.reporting import sha256, verify_data, write_report


def reproduction(tmp_path):
    """Return (data, results): an empty prepared dataset and a reproduction with one table."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": []}))
    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    (results / "tables/value.csv").write_text("value\n1\n")
    write_report(results, [], verify_data(data))
    return data, results


def test_claim_comparison_preserves_discrepancy_and_rounding():
    claim = manuscript.compare_claim("rmse", "Results", [0.37148], [0.33], 0.005, "source")
    assert claim["status"] == "mismatch"
    assert (claim["claim"], claim["location"], claim["paper"]) == ("rmse", "Results", [0.33])
    assert (
        manuscript.compare_claim("rmse", "Results", [8.676], [8.68], 0.005, "source")["status"]
        == "matched"
    )
    assert (
        manuscript.compare_claim("bad", "Methods", [float("nan")], [1], 0.1, "source")["status"]
        == "mismatch"
    )
    assert (
        manuscript.compare_claim("missing", "Methods", [], [1], 0.1, "source")["status"]
        == "mismatch"
    )


def test_verify_reproduction_refuses_altered_table(tmp_path):
    data, results = reproduction(tmp_path)
    manuscript.verify_reproduction(data, results)
    (results / "tables/value.csv").write_text("value\n999\n")
    with pytest.raises(ValueError, match="changed reproduction artifact"):
        manuscript.verify_reproduction(data, results)


def test_verify_reproduction_refuses_stale_code(tmp_path, monkeypatch):
    data, results = reproduction(tmp_path)
    monkeypatch.setattr(manuscript, "source_hashes", lambda: {"changed.py": "changed"})
    with pytest.raises(ValueError, match="code changed"):
        manuscript.verify_reproduction(data, results)


def test_different_pdf_is_recorded_as_not_matching(tmp_path):
    data, results = reproduction(tmp_path)
    pdf = tmp_path / "different.pdf"
    pdf.write_bytes(b"different paper")
    output = tmp_path / "validation"
    report = manuscript.validate_manuscript(pdf, data, results, output)
    assert report["supplied_pdf"] == {"sha256": sha256(pdf), "matches_checked_pdf": False}
    assert report["checked_pdf_sha256"] == manuscript.PAPER_PDF_SHA256
    assert "differs from the version" in (output / "report.md").read_text()


def test_matching_pdf_is_recorded(tmp_path, monkeypatch):
    data, results = reproduction(tmp_path)
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"checked paper")
    monkeypatch.setattr(manuscript, "PAPER_PDF_SHA256", sha256(pdf))
    output = tmp_path / "validation"
    report = manuscript.validate_manuscript(pdf, data, results, output)
    assert report["supplied_pdf"]["matches_checked_pdf"] is True
    assert "differs from the version" not in (output / "report.md").read_text()


def test_missing_analyses_are_unverified_not_matched(tmp_path):
    data, results = reproduction(tmp_path)
    output = tmp_path / "validation"
    report = manuscript.validate_manuscript(None, data, results, output)
    assert report["schema_version"] == 2
    assert report["supplied_pdf"] is None
    assert report["status"] == "incomplete"
    assert report["claims"]
    assert all(item["status"] == "unverified" for item in report["claims"])
    assert all(item["location"] for item in report["claims"])
    assert json.loads((output / "report.json").read_text())["status"] == "incomplete"
    assert (output / "report.md").is_file()
    with pytest.raises(ValueError, match="empty"):
        manuscript.validate_manuscript(None, data, results, output)


@pytest.mark.parametrize(
    "statuses,expected",
    [(["matched", "matched"], "matched"), (["matched", "mismatch"], "incomplete")],
)
def test_status_is_matched_only_when_every_claim_matches(tmp_path, monkeypatch, statuses, expected):
    data, results = reproduction(tmp_path)
    claims = [
        {"claim": f"claim {i}", "location": "Results", "status": status, "source": "table"}
        for i, status in enumerate(statuses)
    ]
    monkeypatch.setattr(manuscript, "numerical_claims", lambda *_: claims)
    report = manuscript.validate_manuscript(None, data, results, tmp_path / "validation")
    assert report["status"] == expected
    assert report["counts"] == {status: statuses.count(status) for status in set(statuses)}


@pytest.mark.parametrize(
    "p_values,t_values,expected",
    [
        ([0.01, 0.01, 0.5, 0.5, 0.5], [-1] * 5, "matched"),
        ([0.5, 0.01, 0.5, 0.5, 0.5], [-1] * 5, "mismatch"),
        ([0.01, 0.01, 0.5, 0.5, 0.5], [1] * 5, "mismatch"),
        ([0.5] * 5, [-1] * 5, "mismatch"),
    ],
)
def test_significant_decreases_must_match_every_interval(tmp_path, p_values, t_values, expected):
    data, results = reproduction(tmp_path)
    table = results / "tables/figure1_paired_tests.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["delta_mps", "current_min", "p_two_sided", "t_statistic"]
        )
        writer.writeheader()
        writer.writerows(
            {"delta_mps": 0.7, "current_min": t, "p_two_sided": p, "t_statistic": stat}
            for t, p, stat in zip([9, 18, 27, 36, 45], p_values, t_values)
        )
    report = write_report(results, [], verify_data(data))
    claims = manuscript.numerical_claims(results, report)
    claim = next(
        item
        for item in claims
        if "Intervals ending in a significant energy decrease at 0.7 m/s" in item["claim"]
    )
    assert claim["status"] == expected


@pytest.mark.parametrize("ratios,expected", [([1.9, 2.2, 2.6], "matched"), ([1.9, 1.5, 2.6], "mismatch"), ([1.9, 2.2], "mismatch")])
def test_grid_advantage_requires_all_three_increasing_comparisons(tmp_path, ratios, expected):
    data, results = reproduction(tmp_path)
    table = results / "tables/s1_grid_sensitivity_summary.csv"
    table.write_text("median_ratio_common_over_specific\n" + "\n".join(map(str, ratios)) + "\n")
    report = write_report(results, [], verify_data(data))
    claims = manuscript.numerical_claims(results, report)
    claim = next(
        item for item in claims if item["claim"].startswith("Cross-validation advantage increases")
    )
    assert claim["status"] == expected


def _claim(tmp_path, name, text, prefix):
    """Return the claim starting with ``prefix`` after writing ``text`` to table ``name``."""
    data, results = reproduction(tmp_path)
    (results / f"tables/{name}.csv").write_text(text)
    report = write_report(results, [], verify_data(data))
    claims = manuscript.numerical_claims(results, report)
    return next(item for item in claims if item["claim"].startswith(prefix))


@pytest.mark.parametrize(
    "rows,expected",
    [
        (["0.4,C,0.00048,0.0,0.0", "0.4,E,0.00048,0.0,0.0"], "matched"),
        (["0.4,C,0.00048,0.0,0.0"], "mismatch"),
        (["0.4,C,0.00048,0.0,0.03", "0.4,E,0.00048,0.0,0.0"], "mismatch"),
    ],
)
def test_identical_low_speed_selection_requires_both_models(tmp_path, rows, expected):
    header = "delta_mps,model,learning_rate,symmetry_weight,fall_weight\n"
    claim = _claim(
        tmp_path, "s4_recomputed_selection", header + "\n".join(rows) + "\n",
        "Models with and without fall term select the same parameters",
    )
    assert claim["status"] == expected


@pytest.mark.parametrize(
    "speeds,expected", [((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), "matched"), ((0.4,), "mismatch")]
)
def test_learning_rate_risk_claim_requires_every_speed(tmp_path, speeds, expected):
    rows = [
        f"{speed},{rate},{probability}"
        for speed in speeds
        for rate, probability in (("0.01", 0.1), ("0.05", 0.6))
    ]
    table = "delta_mps,learning_rate,fall_probability\n" + "\n".join(rows) + "\n"
    claim = _claim(
        tmp_path, "figure3_c_counts", table, "Fall probability at learning rate 0.05 exceeds"
    )
    assert claim["status"] == expected
