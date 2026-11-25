"""Pair NLM Open-i reports with frontal IU X-Ray images."""

import random
import re
import tarfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from evidence.storage import digest_image, file_digest, write_json

from .data import SPLITS, image_at


def xml_reports(source):
    if source.is_dir():
        for path in sorted(source.rglob("*.xml")):
            yield path.name, path.read_bytes()
        return
    with tarfile.open(source, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.lower().endswith(".xml"):
                with archive.extractfile(member) as stream:
                    yield member.name, stream.read()


def image_key(name):
    return name.strip().casefold().removesuffix(".png").removesuffix(".dcm")


def report_record(raw):
    document = ElementTree.fromstring(raw)
    uid = document.find("uId")
    if uid is None or not uid.get("id", "").strip():
        raise ValueError("Missing the Open-i report uId.")
    ident = uid.attrib["id"].strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", ident):
        raise ValueError(f"Invalid Open-i report uId: {ident}.")
    sections = {}
    for block in document.iter("AbstractText"):
        label = block.get("Label", "").strip().lower()
        value = " ".join(" ".join(block.itertext()).split())
        if value and label:
            sections[label] = " ".join(filter(None, (sections.get(label), value)))
    images = {image_key(node.attrib["id"]) for node in document.iter("parentImage") if node.get("id")}
    return ident, sections, images


def available_images(root, frontal_list):
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}. Set --images to the extracted PNG directory.")
    names = {line.strip() for line in frontal_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if not names or any(not re.fullmatch(r"[A-Za-z0-9_.-]+\.png", name, re.I) for name in names):
        raise ValueError(f"Expected one frontal PNG filename per line in {frontal_list}.")
    keys = {image_key(name) for name in names}
    available = {}
    png_count, examples = 0, []
    for path in root.rglob("*"):
        if path.suffix.casefold() != ".png" or not path.is_file():
            continue
        png_count += 1
        if len(examples) < 3:
            examples.append(path.name)
        key = image_key(path.name)
        if key not in keys:
            continue
        if key in available:
            raise ValueError(f"Duplicate frontal image ID {key!r}: {available[key]} and {path.relative_to(root)}.")
        available[key] = path.relative_to(root).as_posix()
    if not png_count:
        raise ValueError(f"No PNG files found under {root}. Extract NLMCXR_png.tgz there or correct --images.")
    if not available:
        raise ValueError(f"Found {png_count} PNG files under {root}, but none match {frontal_list}. "
                         f"PNG examples: {', '.join(examples)}. Frontal list examples: {', '.join(sorted(names)[:3])}.")
    return available


def prepare_openi(args):
    paths = available_images(args.images, args.frontal_list)
    candidates, report_ids = {}, set()
    for name, raw in xml_reports(args.reports):
        try:
            ident, sections, linked = report_record(raw)
        except (ElementTree.ParseError, ValueError) as error:
            raise ValueError(f"Cannot read Open-i report {name}: {error}") from error
        if ident in report_ids:
            raise ValueError(f"Duplicate Open-i report uId: {ident}.")
        report_ids.add(ident)
        findings = sections.get("findings", "")
        selected = sorted(linked & paths.keys())
        if selected and re.search(r"[A-Za-z]", findings):
            candidates[ident] = (findings, sections.get("indication", ""), selected)
    if len(candidates) < len(SPLITS):
        raise ValueError("Fewer than three reports with Findings have a matching frontal PNG.")

    ordered = sorted(candidates)
    random.Random(args.seed).shuffle(ordered)
    held_out = max(1, len(ordered) // 10)
    pools = {"validation": ordered[:held_out], "test": ordered[held_out:2 * held_out],
             "train": ordered[2 * held_out:]}
    limits = {"train": args.train, "validation": args.validation, "test": args.test}
    rows, used_images, used_pixels = [], set(), set()
    unreadable = 0
    # Reserve both held-out pools before picking training images.
    for split in ("validation", "test", "train"):
        records = []
        for ident in pools[split]:
            if len(records) >= limits[split]:
                break
            findings, indication, linked = candidates[ident]
            for image_id in linked:
                if image_id in used_images:
                    continue
                path = paths[image_id]
                try:
                    pixels = digest_image(image_at(args.images, path))
                except OSError:
                    unreadable += 1
                    continue
                if pixels in used_pixels:
                    continue
                used_images.add(image_id)
                used_pixels.add(pixels)
                records.append({"study_id": f"iu-{ident}", "patient_id": f"iu-{ident}", "split": split,
                                "view": "FRONTAL", "image": path, "reference": findings, "indication": indication})
                break
        if len(records) != limits[split]:
            raise ValueError(f"Found {len(records)} usable {split} pairs; requested {limits[split]}. "
                             "Check that the PNG and XML archives are complete, or request a smaller subset.")
        rows.extend(records)
    if not rows:
        raise ValueError("Request at least one study.")
    split_sizes = {split: len(pools[split]) for split in SPLITS}
    write_json(args.output, {"dataset": "IU X-Ray (NLM Open-i)", "split_unit": "report_uid", "seed": args.seed,
                             "frontal_list_sha256": file_digest(args.frontal_list), "matched_reports": len(candidates),
                             "split_pool_sizes": split_sizes, "skipped_unreadable_images": unreadable, "studies": rows})
    print(f"Matched reports: {len(candidates)}; split pools: {split_sizes}")
    print(dict(Counter(row["split"] for row in rows)))
