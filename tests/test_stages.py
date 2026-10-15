"""Seam A through the stage scripts: each stage reads and writes the run folder."""

import json

from pwm import stages


def test_stages_write_the_static_baseline_report_to_the_run_folder(tmp_path):
    stages.make_episodes("toy", root=tmp_path)
    stages.embed("toy", "toy", root=tmp_path)
    stages.evaluate("toy", "toy", root=tmp_path)

    results_dir = tmp_path / "toy" / "toy"
    results = json.loads((results_dir / "results.json").read_text())
    static = results["errors"]["static baseline"]
    assert results["horizons"] == [1, 2, 4, 8]
    # Still windows flatter the static baseline: it looks better overall than on moving windows.
    assert all(moving > overall for moving, overall in zip(static["moving"], static["all"]))
    assert "static baseline" in (results_dir / "report.md").read_text()
