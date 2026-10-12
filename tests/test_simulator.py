import csv
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from divine_whisper_regime_simulator import (
    CASES, PhaseClassifier, SimConfig, State, boost_comparison, clarity, run,
    simulate_case, summarize,
)


SCRIPT = Path(__file__).resolve().parents[1] / "divine_whisper_regime_simulator.py"


class RegimeTests(unittest.TestCase):
    def test_default_boost_is_dormant_and_no_case_reaches_breakthrough(self):
        cfg = SimConfig()
        boosted = run(cfg)
        ablated = run(replace(cfg, breakthrough_boost=0))
        for case in CASES:
            self.assertNotIn("breakthrough", boosted[case]["phase"])
            self.assertEqual(boosted[case]["psi"], ablated[case]["psi"])
        self.assertGreater(boosted["high_start"]["D"][-1], cfg.breakthrough_distortion_threshold)

    def test_threshold_is_continuous_in_clarity_but_changes_the_label(self):
        cfg = SimConfig(breakthrough_distortion_threshold=0.20, phase_distortion_threshold=0.20)
        state = State(A=1, D=0.20, C=2, F=0.1, lam=2, S=2)
        at_gate = clarity(state, cfg)
        self.assertEqual(at_gate, clarity(state, replace(cfg, breakthrough_boost=0)))
        state.D -= 1e-8
        self.assertAlmostEqual(clarity(state, cfg) / clarity(state, replace(cfg, breakthrough_boost=0)), 1.0, places=10)
        state.psi = clarity(state, cfg)
        self.assertEqual(PhaseClassifier(cfg).classify(state), "breakthrough")
        state.D = cfg.phase_distortion_threshold
        self.assertNotEqual(PhaseClassifier(cfg).classify(state), "breakthrough")

        boost_only = SimConfig(breakthrough_distortion_threshold=0.20)
        self.assertNotIn("breakthrough", run(boost_only)["high_start"]["phase"])
        results = run(cfg)
        summary = summarize(results, cfg, "feedback")
        comparison = boost_comparison(cfg, "feedback", summary)
        self.assertEqual(summary["cases"]["high_start"]["first_breakthrough_step"], 78)
        self.assertGreater(comparison["cases"]["high_start"]["max_psi_with_boost"],
                           comparison["cases"]["high_start"]["max_psi_without_boost"])

    def test_feedback_routes_from_state_while_scripted_mode_anchors_case(self):
        cfg = SimConfig(steps=4)
        feedback = simulate_case("focused", cfg, "feedback")
        scripted = simulate_case("focused", cfg, "scripted")
        self.assertEqual(feedback["policy_phase"][0], "normal")
        self.assertEqual(scripted["policy_phase"][0], "focused")
        self.assertNotEqual(feedback["D"], scripted["D"])

    def test_cli_exports_reproducible_summary_and_labeled_trajectories(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first", Path(temp) / "second"
            for directory in (first, second):
                proc = subprocess.run(
                    [sys.executable, str(SCRIPT), "--steps", "12", "--compare-boost", "--out-dir", str(directory)],
                    capture_output=True, text=True, check=True,
                )
                self.assertIn("Symbolic regime simulation", proc.stdout)
            for name in ("summary.json", "trajectories.csv", "boost_comparison.json"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            summary = json.loads((first / "summary.json").read_text())
            self.assertEqual(summary["policy"], "feedback")
            with (first / "trajectories.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 12 * len(CASES))
            self.assertEqual(set(row["case"] for row in rows), set(CASES))
            self.assertTrue(all(row["phase"] and row["policy_phase"] for row in rows))


if __name__ == "__main__":
    unittest.main()
