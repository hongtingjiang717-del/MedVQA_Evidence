"""Prepare the official SLAKE archive and preserve stable image/question IDs."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import shutil
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

from evidence.storage import digest_image, file_digest, write_json

from .paths import DATA, RESULTS


SPLITS = ("train", "validation", "test")
REVISION = "a9083ce"
URL = f"https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/{REVISION}/"
FILES = ("train.json", "validation.json", "test.json", "imgs.zip", "mask.txt")


@dataclass(frozen=True)
class Question:
    split: str
    qid: str
    image_id: str
    question: str
    answer: str
    modality: str
    category: str


class InvalidMask(ValueError):
    """A mask image contains pixels outside the SLAKE integer codebook."""


def download(root, base_url=URL):
    root.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = root / name
        if path.exists():
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name}", flush=True)
        with urlopen(base_url + name, timeout=90) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        temporary.replace(path)


class Slake:
    def __init__(self, root=DATA):
        self.root = Path(root)
        missing = [name for name in FILES if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing SLAKE files {missing}. Run python -m slake_vqa data --download.")
        self.zip = ZipFile(self.root / "imgs.zip")
        self.members = {}
        kinds = {"source.jpg": "image", "source.png": "image", "mask.png": "mask",
                 "detection.json": "detection"}
        for name in self.zip.namelist():
            path = PurePosixPath(name)
            if path.name.lower() in kinds:
                key = (path.parent.name, kinds[path.name.lower()])
                if key in self.members:
                    raise ValueError(f"Duplicate image or mask in SLAKE: {key}")
                self.members[key] = name
        self.labels = {}
        for line in (self.root / "mask.txt").read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                value, name = line.split(":", 1)
                self.labels[name.strip().casefold()] = int(value)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.zip.close()

    def questions(self, split):
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split}")
        raw = json.loads((self.root / f"{split}.json").read_text(encoding="utf-8-sig"))
        result = []
        seen = set()
        for index, row in enumerate(raw):
            if row.get("q_lang", "en").lower() != "en":
                continue
            image_id = PurePosixPath(row["img_name"].replace("\\", "/")).parent.name
            qid = str(row.get("qid", index))
            if qid in seen or (image_id, "image") not in self.members:
                raise ValueError(f"Unresolved image or duplicate question at {split}/{qid}: {image_id}")
            seen.add(qid)
            result.append(Question(split, qid, image_id, str(row["question"]), str(row["answer"]),
                                   str(row.get("modality", "")).strip().upper(), str(row.get("content_type", ""))))
        return result

    def _read(self, image_id, kind):
        with Image.open(BytesIO(self.zip.read(self.members[image_id, kind]))) as image:
            return image.copy()

    def image(self, image_id):
        return self._read(image_id, "image").convert("RGB")

    def detection_boxes(self, image_id, label):
        rows = json.loads(self.zip.read(self.members[image_id, "detection"]))
        return [(x, y, x + width, y + height) for row in rows for name, (x, y, width, height) in row.items()
                if name.casefold() == label.casefold()]

    def mask_values(self, image_id):
        mask = self._read(image_id, "mask")
        if mask.mode not in {"P", "L", "I", "I;16"}:
            channels = mask.convert("RGB").split()
            if any(channel.tobytes() != channels[0].tobytes() for channel in channels[1:]):
                raise InvalidMask(f"Non-grayscale mask {image_id}")
            mask = channels[0]
        colors = mask.getcolors(maxcolors=len(self.labels) + 1)
        if colors is None or {value for _, value in colors} - {*self.labels.values(), 0}:
            raise InvalidMask(f"Non-codebook values in mask {image_id}")
        return mask

    def mask(self, image_id, organ):
        import numpy as np

        value = self.labels[organ.strip().casefold()]
        mask = np.asarray(self.mask_values(image_id)) == value
        if not mask.any():
            raise ValueError(f"{image_id} has no pixels for {organ}")
        return Image.fromarray(mask).convert("L")


def inspect(slake):
    counts, ids, hashes = {}, {}, {}
    for split in SPLITS:
        rows = slake.questions(split)
        ids[split] = {row.image_id for row in rows}
        hashes[split] = {digest_image(slake.image(image_id)) for image_id in ids[split]}
        categories = Counter(row.category for row in rows)
        by_id = {row.image_id: row for row in rows}
        mask_counts = defaultdict(Counter)
        rejected = Counter()
        labels = {value: name for name, value in slake.labels.items()}
        for image_id, row in by_id.items():
            if (image_id, "mask") not in slake.members:
                continue
            try:
                present = {value for _, value in slake.mask_values(image_id).getcolors()}
            except InvalidMask:
                rejected[row.modality] += 1
                continue
            mask_counts[row.modality].update(labels[value] for value in present if value in labels)
        counts[split] = {"english_questions": len(rows), "unique_image_ids": len(ids[split]),
                         "unique_image_hashes": len(hashes[split]),
                         "questions_by_modality": dict(Counter(row.modality for row in rows)),
                         "questions_by_category": dict(categories),
                         "images_with_organ_by_modality": dict(mask_counts), "non_label_masks_by_modality": dict(rejected)}
    overlaps = {}
    for i, first in enumerate(SPLITS):
        for second in SPLITS[i + 1:]:
            overlaps[f"{first}/{second}"] = {"ids": len(ids[first] & ids[second]),
                                              "hashes": len(hashes[first] & hashes[second])}
    return {"dataset": "BoKelvin/SLAKE", "source_url": URL,
            "files_sha256": {name: file_digest(slake.root / name) for name in FILES},
            "splits": counts, "split_overlap": overlaps}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--mirror", action="store_true", help="Download the same fixed revision through hf-mirror.com.")
    parser.add_argument("--output", type=Path, default=RESULTS / "slake_inventory.json")
    args = parser.parse_args()
    if args.download:
        download(args.root, URL.replace("huggingface.co", "hf-mirror.com") if args.mirror else URL)
    with Slake(args.root) as slake:
        report = inspect(slake)
    write_json(args.output, report)
    for split, counts in report["splits"].items():
        print(split, counts["english_questions"], "questions", counts["unique_image_ids"], "images")
    print(args.output)


if __name__ == "__main__":
    main()
