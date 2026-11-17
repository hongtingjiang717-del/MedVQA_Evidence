"""Offline checks for paired statistics and resumable record validation."""
import json
import tempfile
import unittest
from pathlib import Path

from evaluate import aggregate, make_row, percentile, report, validate_record
from medvqa import MODEL_ID


def pair(index, image_prediction="yes", text_prediction="no"):
    item = {"question": "Is this visible?", "answer": "yes"}
    return {
        mode: make_row(
            {"prediction": prediction}, item=item, idx=index, mode=mode,
            split="validation", model=MODEL_ID, img_hash="same-image", origin="test",
        )
        for mode, prediction in (("image", image_prediction), ("text_only", text_prediction))
    }


class EvaluationTests(unittest.TestCase):
    def test_percentile_endpoints_and_singleton(self):
        self.assertEqual(percentile([2, 8], 0), 2)
        self.assertEqual(percentile([2, 8], 1), 8)
        self.assertEqual(percentile([7], 0.975), 7)

    def test_percentile_interpolates_integer_and_fractional_ranks(self):
        self.assertEqual(percentile([1, 3, 5], 0.5), 3)
        self.assertEqual(percentile([1, 3, 5], 0.25), 2)

    def test_percentile_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            percentile([1], 1.1)

    def test_clustered_effect_and_transitions(self):
        result = aggregate([pair(1), pair(2)], seed=42, iters=100)
        self.assertEqual(result["paired_effect"]["n_unique_images"], 1)
        self.assertEqual(result["paired_effect"]["em_cluster_bootstrap_95ci_pp"], [100, 100])
        self.assertEqual(result["paired_transitions"]["image_correct_text_wrong"], 2)
        self.assertEqual(result["paired_transitions"]["answer_changed"], 2)

    def test_bootstrap_is_reproducible(self):
        pairs = [pair(1), pair(2, "no", "yes")]
        pairs[1]["image"]["image_sha256"] = "other-image"
        self.assertEqual(aggregate(pairs, seed=42, iters=100), aggregate(pairs, seed=42, iters=100))

    def test_record_mismatch_is_rejected(self):
        row = pair(1)["image"]
        row["prompt_template_sha256"] = "changed-prompt"
        with self.assertRaises(RuntimeError):
            validate_record(row, item={"question": "Is this visible?", "answer": "yes"},
                            idx=1, mode="image", split="validation", model=MODEL_ID, img_hash="same-image")

    def test_partial_report_does_not_claim_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            report(list(pair(1).values()), indices=[1, 2], output=output,
                   split="validation", model=MODEL_ID, seed=42)
            result = json.loads(output.with_name("predictions_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["missing_predictions"], 2)

    def test_duplicate_records_are_rejected(self):
        row = pair(1)["image"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                report([row, row], indices=[1], output=Path(directory) / "p.jsonl",
                       split="validation", model=MODEL_ID, seed=42)


if __name__ == "__main__":
    unittest.main()
