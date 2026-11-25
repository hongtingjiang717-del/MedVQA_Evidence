"""Run MAIRA-2 chest X-ray research inference in its own Python environment."""

import argparse
import json
from pathlib import Path
from statistics import mean

from PIL import Image

from evidence.maira import CARDIOMEGALY, CXR, MODEL, MairaPredictor

from .dataset import SPLITS, Slake
from .paths import DATA, RESULTS
from evidence.storage import MODEL_ROOT, digest_image, write_json

def box_iou(first, second):
    a, b, c, d = first
    x, y, u, v = second
    overlap = max(0, min(c, u) - max(a, x)) * max(0, min(d, v) - max(b, y))
    return overlap / ((c - a) * (d - b) + (u - x) * (v - y) - overlap)


class ChestRegions:
    """Read box predictions exported in the separate MAIRA-2 environment."""

    def __init__(self, path):
        report = json.loads(path.read_text(encoding="utf-8"))
        if (report.get("schema"), report.get("target")) != (1, CARDIOMEGALY):
            raise ValueError("Expected a MAIRA-2 cardiomegaly prediction file (schema 1).")
        self.cases = {}
        for row in report["cases"]:
            image_id = row["image_id"]
            if image_id in self.cases:
                raise ValueError(f"Duplicate MAIRA-2 image: {image_id}.")
            status = row["finding_status"]
            if status not in {"present", "uncertain", "no_positive_mention"} or (row["boxes_xyxy"] and status != "present"):
                raise ValueError(f"Invalid MAIRA-2 finding or inconsistent boxes: {image_id}.")
            self.cases[image_id] = row

    def region(self, image_id, image):
        row = self.cases.get(image_id)
        if row is None:
            return None, {"finding_status": "not_precomputed", "boxes_xyxy": []}
        if row["image_sha256"] != digest_image(image):
            raise ValueError(f"MAIRA-2 image bytes do not match SLAKE: {image_id}.")
        width, height = image.size
        mask = Image.new("L", image.size)
        for x1, y1, x2, y2 in row["boxes_xyxy"]:
            if not all(type(value) is int for value in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid MAIRA-2 box for {image_id}.")
            mask.paste(255, (x1, y1, x2, y2))
        return mask, {key: row[key] for key in ("finding_status", "matched_findings", "box_source", "boxes_xyxy")}


def evaluate(data, image_ids, predictor, cases, on_case):
    seen = {row["image_id"]: row for row in cases}
    for image_id in image_ids:
        image = data.image(image_id)
        image_hash = digest_image(image)
        if image_id in seen:
            if seen[image_id]["image_sha256"] != image_hash:
                raise ValueError(f"SLAKE image changed since the previous run: {image_id}.")
            continue
        prediction = predictor.predict(image)
        expected = data.detection_boxes(image_id, CARDIOMEGALY)
        iou = max((box_iou(p, t) for p in prediction["boxes_xyxy"] for t in expected), default=0.0) if expected else None
        cases.append({"image_id": image_id, "image_sha256": image_hash, "reference_boxes_xyxy": expected,
                      "best_box_iou": iou, **prediction})
        print(image_id, prediction["finding_status"], prediction["boxes_xyxy"], flush=True)
        on_case()


def summarize(cases):
    positives = [row for row in cases if row["reference_boxes_xyxy"]]
    overlaps = [row["best_box_iou"] for row in positives]
    detected = [row for row in cases if row["finding_status"] == "present"]
    agreement = sum(bool(row["reference_boxes_xyxy"]) for row in detected)
    return {"images": len(cases), "images_with_cardiomegaly_box": len(positives),
            "reported_cardiomegaly": len(detected), "reported_uncertain": sum(row["finding_status"] == "uncertain" for row in cases),
            "matched_box_annotated_images": agreement,
            "precision_against_box_annotations": agreement / len(detected) if detected else None,
            "recall_on_box_annotated_images": agreement / len(positives) if positives else None,
            "mean_box_iou_on_annotated_images": mean(overlaps) if overlaps else None,
            "box_hit_rate_at_0_5": mean(score >= .5 for score in overlaps) if overlaps else None,
            "annotated_images_with_no_prediction_box": sum(not row["boxes_xyxy"] for row in positives)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    local = MODEL_ROOT / "maira-2"
    parser.add_argument("--model", default=str(local) if (local / "config.json").is_file() else MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--split", choices=(*SPLITS, "audit"), default="test",
                        help="audit combines the disjoint validation and test image groups.")
    parser.add_argument("--image-id", help="Run a single public SLAKE chest X-ray from the selected split.")
    parser.add_argument("--output", type=Path, default=RESULTS / "maira_cardiomegaly.json")
    args = parser.parse_args()
    with Slake(args.root) as data:
        splits = ("validation", "test") if args.split == "audit" else (args.split,)
        groups = {split: {row.image_id for row in data.questions(split) if row.modality == CXR} for split in splits}
        ids = sorted(set.union(*groups.values()))
        if args.image_id:
            if args.image_id not in ids:
                parser.error("--image-id must be a chest X-ray in the selected split.")
            ids = [args.image_id]
        if not ids:
            parser.error("No SLAKE chest X-rays match the requested split.")
        config = {"schema": 1, "target": CARDIOMEGALY, "model": args.model, "revision": args.revision,
                  "device": args.device, "split": args.split, "image_ids": ids}
        previous = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
        if previous and any(previous.get(key) != value for key, value in config.items()):
            parser.error("Choose a new --output for a different MAIRA-2 configuration.")
        cases = previous.get("cases", [])
        if [row["image_id"] for row in cases] != ids[:len(cases)]:
            raise ValueError("Saved MAIRA-2 cases do not match this run.")

        def save():
            by_split = {split: summarize([row for row in cases if row["image_id"] in group])
                        for split, group in groups.items()}
            write_json(args.output, {**config, "summary": summarize(cases), "summary_by_split": by_split,
                                    "cases": cases})

        if len(cases) != len(ids):
            predictor = MairaPredictor(args.model, args.device, args.revision)
            evaluate(data, ids, predictor, cases, save)
        else:
            for row in cases:
                if row["image_id"] not in ids or digest_image(data.image(row["image_id"])) != row["image_sha256"]:
                    raise ValueError("Cached MAIRA-2 results belong to a different set of SLAKE images.")
        print(json.dumps(summarize(cases), indent=2))


if __name__ == "__main__":
    main()
