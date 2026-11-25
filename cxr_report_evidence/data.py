"""Prepare a patient-disjoint manifest for a local MIMIC-CXR subset."""

import csv
import gzip
import json
import random
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from evidence.storage import write_json

SPLITS = ("train", "validation", "test")
JPG_URL = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
HEADINGS = re.compile(r"^[ \t]*(findings|impression|indication|history|comparison|technique|exam(?:ination)?|conclusion|clinical history|recommendations?)[ \t]*:[ \t]*", re.I | re.M)
VARIANTS = ("base_image", "base_text", "lora_image", "lora_text")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def sections(report):
    matches = list(HEADINGS.finditer(report))
    return {
        match.group(1).lower(): " ".join(report[match.end():end].split())
        for match, end in zip(matches, [*(item.start() for item in matches[1:]), len(report)])
    }


def image_at(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path must stay within --images.")
    with Image.open(path) as image:
        return image.convert("RGB")


def load_manifest(path):
    rows = read_json(path)["studies"]
    patients = {}
    ids = set()
    for row in rows:
        ident, patient, split = row["study_id"], row["patient_id"], row["split"]
        if ident in ids or split not in SPLITS or patients.setdefault(patient, split) != split:
            raise ValueError(f"Duplicate study or patient overlap across splits: {ident}.")
        ids.add(ident)
    if not rows:
        raise ValueError("The manifest has no studies.")
    return rows


def samples(rows, split, limit):
    allowed = SPLITS if split == "all" else ("validation", "test") if split == "audit" else (split,)
    selected = [row for row in rows if row["split"] in allowed][:limit]
    if not selected:
        raise ValueError(f"No studies for split {split}.")
    return selected


def prepare(args):
    views = {row["dicom_id"]: row["ViewPosition"].upper() for row in csv_rows(args.metadata)}
    candidates = {split: {} for split in SPLITS}
    patients = {}
    for row in csv_rows(args.splits):
        dicom, study, subject = (row[key] for key in ("dicom_id", "study_id", "subject_id"))
        split = {"validate": "validation", "val": "validation"}.get(row["split"], row["split"])
        view = views.get(dicom)
        if split not in SPLITS or patients.setdefault(subject, split) != split:
            raise ValueError("Invalid split or patient overlap in split file.")
        if view in {"AP", "PA"}:
            candidates[split].setdefault((subject, study), []).append((view != "PA", dicom, view))

    selected = []
    rng = random.Random(args.seed)
    with ZipFile(args.reports) as archive:
        reports = {Path(name).stem: name for name in archive.namelist() if name.endswith(".txt")}
        limits = {"train": args.train, "validation": args.validation, "test": args.test}
        for split in SPLITS:
            studies = sorted(candidates[split])
            rng.shuffle(studies)
            count = 0
            for subject, study in studies:
                if count >= limits[split]:
                    break
                name = reports.get(f"s{study}")
                if name is None:
                    continue
                report = sections(archive.read(name).decode("utf-8", errors="replace"))
                findings = report.get("findings", "")
                if not findings or not re.search(r"[A-Za-z]", findings):
                    continue
                _, dicom, view = min(candidates[split][(subject, study)])
                relative = f"files/p{subject[:2]}/p{subject}/s{study}/{dicom}.jpg"
                selected.append({"study_id": study, "patient_id": subject, "split": split, "view": view,
                                 "image": relative, "reference": findings, "indication": report.get("indication", "")})
                count += 1
            if count != limits[split]:
                raise ValueError(f"Only found {count} eligible {split} studies; requested {limits[split]}.")
    if not selected:
        raise ValueError("Request at least one study.")
    write_json(args.output, {"dataset": "MIMIC-CXR 2.1.0 / MIMIC-CXR-JPG 2.1.0", "seed": args.seed, "studies": selected})
    args.urls.parent.mkdir(parents=True, exist_ok=True)
    args.urls.write_text("".join(f"{JPG_URL}{row['image']}\n" for row in selected), encoding="ascii")
    print(dict(Counter(row["split"] for row in selected)))
