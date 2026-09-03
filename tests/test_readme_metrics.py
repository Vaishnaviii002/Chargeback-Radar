from __future__ import annotations

import json
from pathlib import Path

from src.readme_metrics import (
    END_MARKER,
    START_MARKER,
    render_metrics_block,
    update_readme_metrics,
)


def fake_manifest() -> dict:
    return {
        "dataset": {
            "payments": 80_000,
            "chargebacks": 578,
            "chargeback_rate": 0.007225,
        },
        "held_out_evaluation": {
            "dataset": {
                "test_rows": 12_173,
                "positive_labels": 87,
                "base_rate": 0.00715,
            },
            "model_performance": {
                "average_precision": 0.1493,
                "precision": 0.19466,
                "recall": 0.58621,
                "f1": 0.2923,
            },
            "confusion_matrix": {"false_positive": 211},
            "false_positive_cost": {"total_rupees": 31_650},
        },
        "calibration": {
            "test_raw_brier": 0.01,
            "test_calibrated_brier": 0.006,
        },
        "default_policy": {
            "costs": {
                "gross_avoided_loss_rupees": 200_000,
                "intervention_cost_rupees": 63_132,
                "estimated_net_benefit_rupees": 136_868,
            }
        },
        "effectiveness_sensitivity": {
            "ranges": {
                "estimated_net_benefit_rupees": {
                    "minimum": 90_000,
                    "maximum": 160_000,
                }
            }
        },
    }


def test_rendered_block_contains_source_metrics() -> None:
    block = render_metrics_block(fake_manifest())

    assert "80,000" in block
    assert "14.93%" in block
    assert "20.9×" in block
    assert "₹136,868" in block
    assert START_MARKER in block
    assert END_MARKER in block


def test_updater_preserves_content_outside_markers(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "README.md").write_text(
        (
            f"# Before\n{START_MARKER}\n"
            f"stale\n{END_MARKER}\n## After\n"
        ),
        encoding="utf-8",
    )
    (reports / "submission_manifest.json").write_text(
        json.dumps(fake_manifest()),
        encoding="utf-8",
    )

    update_readme_metrics(tmp_path)
    updated = (tmp_path / "README.md").read_text(
        encoding="utf-8"
    )

    assert updated.startswith("# Before")
    assert updated.endswith("## After\n")
    assert "stale" not in updated
    assert "₹136,868" in updated


def test_updater_requires_markers(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (tmp_path / "README.md").write_text(
        "# Missing markers",
        encoding="utf-8",
    )
    (reports / "submission_manifest.json").write_text(
        json.dumps(fake_manifest()),
        encoding="utf-8",
    )

    try:
        update_readme_metrics(tmp_path)
    except ValueError as error:
        assert "markers" in str(error)
    else:
        raise AssertionError("Expected missing markers to fail.")
