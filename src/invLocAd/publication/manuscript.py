"""Compare reproduced numerical results with the values reported in the paper."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from itertools import pairwise
from pathlib import Path

from .reporting import sha256, source_hashes, utc_now, verify_data

PAPER_TITLE = "Safety prioritization shapes locomotor adaptation and performance"
PAPER_PDF_SHA256 = "c8d97a54488858f2a4d918cfed6a9738f6a9d39139abfd35dca3919657e11c8f"
SPEEDS = (0.4, 0.7, 1.0)


def compare_claim(label, location, observed, expected, tolerance, source):
    """Compare observed values with the values printed in the paper.

    Args:
        label: Short description of the claim.
        location: Where the value appears in the paper.
        observed: Values computed from the reproduction tables.
        expected: Values printed in the paper, in the same order.
        tolerance: Maximum absolute difference, normally half the printed precision.
        source: Reproduction table that provides the observed values.

    Returns:
        Claim record whose status is "matched" only when every value is finite and
        within tolerance and the two lists have the same length.
    """
    values = list(observed)
    targets = list(expected)
    matches = len(values) == len(targets) and all(
        math.isfinite(value) and math.isclose(value, target, rel_tol=0, abs_tol=tolerance)
        for value, target in zip(values, targets, strict=False)
    )
    return {
        "claim": label,
        "location": location,
        "status": "matched" if matches else "mismatch",
        "observed": [value if math.isfinite(value) else None for value in values],
        "paper": targets,
        "absolute_tolerance": tolerance,
        "source": source,
    }


def verify_reproduction(data_root: Path, results_root: Path) -> dict:
    """Return the reproduction report after checking that inputs, code and outputs are unchanged.

    Raises:
        ValueError: If the prepared inputs, analysis code or any listed output differs from
            the state recorded when the reproduction ran.
    """
    report = json.loads((results_root / "report.json").read_text())
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported reproduction report.")
    integrity = verify_data(data_root)
    if integrity["errors"] or integrity != report.get("inputs"):
        raise ValueError(
            "Prepared inputs differ from the reproduction run; verify and reproduce again."
        )
    if source_hashes() != report.get("source_sha256"):
        raise ValueError(
            "Analysis code changed after reproduction; reproduce again before comparing claims."
        )
    seen = set()
    for record in report["artifacts"]:
        relative = Path(record["path"])
        path = (results_root / relative).resolve()
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(results_root.resolve())
        ):
            raise ValueError(f"Unsafe reproduction artifact: {relative}")
        if record["path"] in seen or not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing, duplicate or changed reproduction artifact: {relative}")
        seen.add(record["path"])
    return report


def _increasing(values) -> int:
    """Return 1 if ``values`` has at least two elements and strictly increases, else 0."""
    return int(len(values) > 1 and all(b > a for a, b in pairwise(values)))


def _speed(row) -> float | None:
    """Return the row's belt-speed difference in m/s, or None for pooled rows."""
    try:
        return float(row["delta_mps"])
    except ValueError:
        return None


def _by_speed(rows, column, speeds=SPEEDS, **match):
    """Return ``column`` as floats for each speed in ``speeds``, from rows matching ``match``."""
    selected = [r for r in rows if all(r[key] == value for key, value in match.items())]
    return [float(next(r for r in selected if _speed(r) == speed)[column]) for speed in speeds]


def numerical_claims(results_root: Path, reproduction: dict) -> list[dict]:
    """Compare the reproduction tables with every numerical claim checked against the paper.

    Args:
        results_root: Directory written by ``invLocAd-publication reproduce``.
        reproduction: Its verified ``report.json`` contents.

    Returns:
        One record per claim with status "matched", "mismatch" or "unverified" (when the
        required table was not generated).
    """
    listed = {item["path"] for item in reproduction["artifacts"]}
    claims = []

    def check(label, location, name, extract, expected, tolerance):
        """Append the comparison of ``extract(rows of table name)`` with ``expected``."""
        source = f"tables/{name}.csv"
        if source not in listed:
            claims.append(
                {
                    "claim": label,
                    "location": location,
                    "status": "unverified",
                    "source": source,
                    "reason": "Required table was not generated by this run.",
                }
            )
            return
        with (results_root / source).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        try:
            observed = extract(rows)
        except (KeyError, StopIteration, ValueError) as error:
            claims.append(
                {
                    "claim": label,
                    "location": location,
                    "status": "mismatch",
                    "source": source,
                    "reason": f"Table does not contain the required values ({error}).",
                }
            )
            return
        claims.append(compare_claim(label, location, observed, expected, tolerance, source))

    for name, label, expected in (
        ("energy", "metabolic", [8, 9, 9]),
        ("sla", "step-length asymmetry", [7, 8, 8]),
    ):
        check(
            f"Participants with {label} data at 0.4, 0.7, 1.0 m/s",
            "Methods, Experimental protocol",
            f"figure1_{name}_participants",
            lambda rows: [
                len({r["participant_index"] for r in rows if float(r["delta_mps"]) == d})
                for d in SPEEDS
            ],
            expected,
            0,
        )
    for delta, expected in ((0.4, [9]), (0.7, [9, 18]), (1.0, [9, 18, 27])):
        check(
            f"Intervals ending in a significant energy decrease at {delta} m/s (min)",
            "Results; Figure 1E",
            "figure1_paired_tests",
            lambda rows, d=delta: sorted(
                float(r["current_min"])
                for r in rows
                if float(r["delta_mps"]) == d
                and float(r["p_two_sided"]) < 0.05
                and float(r["t_statistic"]) < 0
            ),
            expected,
            0,
        )
    for column, expected in (("mean_rmse", [0.37, 0.90, 1.28]), ("sd_rmse", [0.32, 0.83, 0.49])):
        check(
            f"In-sample inverse-fit RMSE {column.split('_')[0]} (percentage points)",
            "Results, Inverse adaptation model",
            "inverse_fit_rmse_summary",
            lambda rows, col=column: [float(r[col]) for r in rows],
            expected,
            0.005,
        )
    check(
        "Energy minimization: energy change decreases from 0.4 to 1.0 m/s",
        "Results; Figures 1C and 2C",
        "figure2c_energy_minimization",
        lambda rows: [
            _increasing([-v for v in _by_speed(rows, "energy_change_pct", (0.4, 0.6, 0.8, 1.0))])
        ],
        [1],
        0,
    )
    for name, label in (
        ("figure2c_model_summary", "Inverse adaptation model"),
        ("figure2c_experimental_summary", "Experimental data"),
    ):
        check(
            f"{label}: energy change increases with belt speed difference",
            "Results; Figure 2C",
            name,
            lambda rows: [_increasing(_by_speed(rows, "mean_energy_change_pct"))],
            [1],
            0,
        )

    def highest_rate_riskier(rows):
        """Return [1] if learning rate 0.05 is riskier than 0.01 at every Figure 3C speed."""
        speeds = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        low = _by_speed(rows, "fall_probability", speeds, learning_rate="0.01")
        high = _by_speed(rows, "fall_probability", speeds, learning_rate="0.05")
        return [int(all(h > lo for lo, h in zip(low, high, strict=True)))]

    check(
        "Fall probability at learning rate 0.05 exceeds that at 0.01 at every speed",
        "Results; Figure 3C",
        "figure3_c_counts",
        highest_rate_riskier,
        [1],
        0,
    )
    check(
        "Simulations per Figure 3C cell",
        "Methods, Effect of model parameters on fall risk",
        "figure3_c_counts",
        lambda rows: sorted({float(r["trials"]) for r in rows}),
        [50],
        0,
    )
    check(
        "Unchanged fitted parameters have higher fall probability at larger speed differences "
        "(Wilcoxon p < 0.05)",
        "Results, Learning is modulated...; Figure 4",
        "figure4_fall_probability_tests",
        lambda rows: [
            int(
                len(rows) == 3
                and all(
                    float(r["p_one_sided_greater"]) < 0.05
                    and float(r["mean_larger_landscape_pct"]) > float(r["mean_own_landscape_pct"])
                    for r in rows
                )
            )
        ],
        [1],
        0,
    )
    check(
        "Figure 4 mean fall probability within each ellipse",
        "Figure 4",
        "figure4_ellipse_risk",
        lambda rows: [float(r["mean_fall_probability_pct"]) / 100 for r in rows],
        [0.52, 0.81, 0.50, 0.99, 0.83, 0.55],
        0.005,
    )
    check(
        "Held-out RMSE with fall term, without fall term and noise floor",
        "Results, Learning is modulated...; S1 Text Section 4",
        "s4_loso_summary",
        lambda rows: [
            float(
                next(r for r in rows if r["delta_mps"] == "pooled" and r["model"] == model)[
                    "mean_rmse"
                ]
            )
            for model in ("C", "E", "noise_floor")
        ],
        [8.68, 10.20, 8.95],
        0.005,
    )
    check(
        "Best-fit fall penalty weight at 0.4, 0.7, 1.0 m/s",
        "Results, Learning is modulated...",
        "s4_recomputed_selection",
        lambda rows: _by_speed(rows, "fall_weight", model="C"),
        [0, 10, 100],
        0,
    )
    check(
        "Models with and without fall term select the same parameters at 0.4 m/s",
        "Results; Supplementary Figure 4",
        "s4_recomputed_selection",
        lambda rows: [
            int(
                sorted(r["model"] for r in rows if _speed(r) == 0.4) == ["C", "E"]
                and len({
                    (r["learning_rate"], r["symmetry_weight"], r["fall_weight"])
                    for r in rows
                    if _speed(r) == 0.4
                }) == 1
            )
        ],
        [1],
        0,
    )
    check(
        "Paired Wilcoxon p-value for the fall term at 1.0 m/s",
        "Results, Learning is modulated...",
        "s4_loso_tests",
        lambda rows: [
            float(r["wilcoxon_two_sided_p"])
            for r in rows
            if r["delta_mps"] != "pooled" and float(r["delta_mps"]) == 1.0
        ],
        [0.008],
        0.0005,
    )
    for column, expected in (("mean", [0.71, 0.81, 0.86]), ("sd", [0.24, 0.23, 0.14])):
        check(
            f"Single-exponential R^2 of experimental energy, {column}",
            "Methods, Aligning simulated and individual-specific time-varying performance",
            "r2_summary",
            lambda rows, col=column: _by_speed(rows, col, metric="r2_experimental"),
            expected,
            0.005,
        )
    check(
        "Range of the double-exponential R^2 gain across condition means",
        "Methods, Aligning simulated and individual-specific time-varying performance",
        "r2_timescale_summary",
        lambda rows: [
            min(float(r["mean_delta_r2"]) for r in rows),
            max(float(r["mean_delta_r2"]) for r in rows),
        ],
        [0.011, 0.055],
        0.0005,
    )
    for label, column, target, tolerance in (
        ("condition-specific median held-out RMSE (%)", "median_condition_specific", 8.9, 0.05),
        ("common-parameter median held-out RMSE (%)", "median_common_size_matched", 16.9, 0.05),
        ("ratio of median RMSE", "median_ratio_common_over_specific", 1.9, 0.05),
        ("participants better fit by condition-specific parameters", "participants_better", 26, 0),
        (
            "signed-rank p-value (MATLAB signrank)",
            "wilcoxon_two_sided_matlab_signrank_default",
            8.3e-6,
            0.05e-6,
        ),
    ):
        check(
            f"Cross-validation: {label}",
            "S1 Text Section 1",
            "s1_loso_summary",
            lambda rows, col=column: [float(rows[0][col])],
            [target],
            tolerance,
        )
    check(
        "Cross-validation advantage increases when the common grid is relaxed",
        "S1 Text Section 1",
        "s1_grid_sensitivity_summary",
        lambda rows: [
            int(
                len(rows) == 3
                and _increasing([float(r["median_ratio_common_over_specific"]) for r in rows])
            )
        ],
        [1],
        0,
    )
    return claims


SCOPE_NOTES = [
    {
        "topic": "In-sample fit error and fitted parameters",
        "note": "Each participant's fit is the parameter cell containing the single simulation, "
        "among the 50 per cell, with the lowest RMSE; the reported RMSE is that minimum.",
    },
    {
        "topic": "Figure 2B",
        "note": "The plotted trajectories are the means of the nine 0.7 m/s participants and "
        "their nine fitted simulations from the regenerated realization.",
    },
    {
        "topic": "Figures 1C and 2C, energy minimization",
        "note": "The five trajectory-optimization values are the reported OpenSim Moco results; "
        "the optimization itself is not rerun.",
    },
    {
        "topic": "Figure 3B",
        "note": "Each plotted sample is the lowest center-of-mass height between consecutive "
        "midstances, which is reached at the transition between stances.",
    },
    {
        "topic": "Fall-penalty noise floor",
        "note": "RMSE between the exponential fit to each held-out participant and the exponential "
        "fit to the mean trajectory of the remaining participants.",
    },
    {
        "topic": "S1 Text Section 1",
        "note": "Cross-validation uses the regenerated trial-average grid, which scores the mean "
        "normalized trajectory of nonfalling trials in each parameter cell.",
    },
    {
        "topic": "S1 Text Section 2",
        "note": "Model asymmetry uses one parameter set for all conditions (learning rate 6e-5, "
        "symmetry weight 1, fall weight 0); the data are the 20 participants (7/7/6) with "
        "force-plate recordings.",
    },
    {
        "topic": "Fall-penalty comparison at 0.4 m/s",
        "note": "The full-data fits of the two models are identical; three of the eight held-out "
        "fits select a nonzero fall weight.",
    },
    {
        "topic": "Double-exponential R^2 gain",
        "note": "The range 0.011-0.055 spans the condition means of the experimental and model "
        "fits; the experimental fits alone give 0.015-0.027.",
    },
    {
        "topic": "S1 Text Section 4, penalty gradient",
        "note": "Saved per-seed mechanical summaries show mean trends; they cannot show a nonzero "
        "penalty gradient at every step.",
    },
]


def validate_manuscript(
    manuscript: Path | None, data_root: Path, results_root: Path, output: Path
) -> dict:
    """Write a report comparing a verified reproduction with the values reported in the paper.

    Args:
        manuscript: Optional PDF of the paper; only its SHA-256 is recorded.
        data_root: Prepared publication inputs used by the reproduction.
        results_root: Output directory of ``invLocAd-publication reproduce``.
        output: Empty directory for ``report.json`` and ``report.md``.

    Returns:
        The report; its status is "matched" only when every numerical claim matched.
    """
    for source in (data_root, results_root):
        if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(
            output.resolve()
        ):
            raise ValueError(
                "Place manuscript validation outside the input and reproduction directories."
            )
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty manuscript-validation output directory.")
    reproduction = verify_reproduction(data_root, results_root)
    claims = numerical_claims(results_root, reproduction)
    pdf = None
    if manuscript is not None:
        digest = sha256(manuscript)
        pdf = {"sha256": digest, "matches_checked_pdf": digest == PAPER_PDF_SHA256}
    report = {
        "schema_version": 2,
        "created_at": utc_now(),
        "paper_title": PAPER_TITLE,
        "checked_pdf_sha256": PAPER_PDF_SHA256,
        "supplied_pdf": pdf,
        "reproduction_report_sha256": sha256(results_root / "report.json"),
        "status": (
            "matched" if all(item["status"] == "matched" for item in claims) else "incomplete"
        ),
        "claims": claims,
        "counts": dict(Counter(item["status"] for item in claims)),
        "scope_notes": SCOPE_NOTES,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = [
        "# Comparison with the paper",
        "",
        f"Paper: {PAPER_TITLE}. Values are compared at the precision printed in the paper.",
        "",
        "| Claim | Paper location | Status | Observed | Paper |",
        "|---|---|---|---|---|",
    ]
    for item in claims:
        observed = item.get("observed", item.get("reason", ""))
        lines.append(
            f"| {item['claim']} | {item['location']} | {item['status']} | {observed} "
            f"| {item.get('paper', '')} |"
        )
    lines += ["", "## Scope notes", ""]
    lines += [f"- **{item['topic']}:** {item['note']}" for item in SCOPE_NOTES]
    if pdf is not None and not pdf["matches_checked_pdf"]:
        lines += [
            "",
            "This PDF differs from the version these values were checked against.",
        ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return report
