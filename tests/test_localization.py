import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageChops

from cxr_report_evidence.audit import ground
from cxr_report_evidence.data import read_json
from evidence.maira import MairaOutputError, finding_status, parse_output, pixel_boxes
from evidence.perturbation import matched_area_mask, matched_mask
from evidence.storage import digest_image, file_digest, write_json


class LocalizationTests(unittest.TestCase):
    def test_centered_cardiomegaly_box_uses_an_equal_area_control(self):
        mask = Image.new("L", (1024, 1024))
        mask.paste(255, (301, 384, 777, 770))
        self.assertIsNone(matched_mask(mask))
        control = matched_area_mask(mask)
        self.assertIsNotNone(control)
        self.assertEqual(control.histogram()[255], mask.histogram()[255])
        self.assertIsNone(ImageChops.multiply(mask, control).getbbox())

    def test_uncertain_and_negated_findings_do_not_count_as_positive(self):
        examples = {
            "The heart is mildly enlarged.": "present",
            "No interval change in mild cardiomegaly.": "present",
            "Cardiomegaly is not present.": "no_positive_mention",
            "No cardiomegaly.": "no_positive_mention",
            "Cannot exclude cardiomegaly.": "uncertain",
            "Possible cardiomegaly; however heart is enlarged.": "present",
        }
        for text, status in examples.items():
            with self.subTest(text=text):
                self.assertEqual(finding_status(text), status)

    def test_boxes_use_the_original_image_frame(self):
        class Processor:
            def adjust_box_for_original_image_size(self, box, width, height):
                # A 100x100 crop in the centre of a 200x100 image.
                self.size = (width, height)
                x1, y1, x2, y2 = box
                return (0.25 + x1 / 2, y1, 0.25 + x2 / 2, y2)

        processor = Processor()
        self.assertEqual(pixel_boxes(processor, (200, 100), [(0., .2, 1., .8), (0., 0., 0., 0.)]),
                         [(50, 20, 150, 80)])
        self.assertEqual(processor.size, (200, 100))

    def test_maira_parser_keeps_complete_boxes_before_an_incomplete_phrase(self):
        processor = SimpleNamespace(
            phrase_start_token="<obj>", phrase_end_token="</obj>",
            convert_output_to_plaintext_or_grounded_sequence=Mock(),
        )
        full_phrase = "<obj>Opacity<box><x10><y10><x30><y30></box></obj>"
        expected = [("Opacity", [(0.1, 0.1, 0.3, 0.3)])]

        def parse(text):
            if text == full_phrase:
                return expected
            raise AssertionError

        processor.convert_output_to_plaintext_or_grounded_sequence.side_effect = parse
        self.assertEqual(parse_output(processor, full_phrase + "<obj>Incomplete"), expected)
        with self.assertRaises(MairaOutputError):
            parse_output(processor, "<obj>Incomplete")

    def test_ground_records_malformed_output_and_resumes_past_it(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (16, 16))
            image.save(root / "image.png")
            manifest = root / "manifest.json"
            write_json(manifest, {"studies": [{
                "study_id": "test", "patient_id": "p1", "split": "test", "image": "image.png",
            }]})
            reports = root / "reports.json"
            write_json(reports, {
                "config": {"manifest_sha256": file_digest(manifest), "variants": ["base_image"]},
                "rows": [{"study_id": "test", "image_sha256": digest_image(image),
                          "reports": {"base_image": "Left opacity.\nRight opacity."}}],
            })
            write_json(root / "config.json", {})
            output = root / "ground.json"
            args = SimpleNamespace(manifest=manifest, reports=reports, images=root, model=root,
                                   device="cpu", output=output, max_findings=4)
            with patch("evidence.maira.MairaPredictor") as predictor:
                model = predictor.return_value
                model.generate.side_effect = [MairaOutputError(), [("Right opacity", [(0.25, 0.25, 0.5, 0.5)])]]
                model.processor.adjust_box_for_original_image_size.side_effect = lambda box, w, h: box
                ground(args)
                ground(args)
                self.assertEqual(model.generate.call_count, 2)
            rows = read_json(output)["rows"]
            self.assertEqual([row["ground_status"] for row in rows], ["malformed_output", "grounded"])
            self.assertEqual(rows[0]["boxes"], [])
            self.assertEqual(rows[1]["boxes"], [[4, 4, 8, 8]])
