"""Ground individual findings, run visual counterfactuals and estimate review risk."""

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from PIL import Image

from evidence.storage import digest_image, file_digest, write_json

from .data import image_at, load_manifest, read_json
from .reports import findings, resized, resume, save

UNLOCALIZED = re.compile(
    r"^(?:(?:there (?:is|are) )?no|without|negative|normal|unable|cannot|insufficient|unremarkable)\b"
    r"|\b(?:within normal limits|is normal|no acute|prior study|interval change|unchanged|not provided|no image)\b",
    re.I,
)
FEATURES = ("original_support", "original_conflict", "relative_support_drop")
ALIGNMENT = ("original_alignment", "relative_alignment_drop")


def ground(args):
    from evidence.maira import MairaOutputError, MairaPredictor, pixel_boxes

    reports = read_json(args.reports)
    if reports["config"]["manifest_sha256"] != file_digest(args.manifest):
        raise ValueError("The report predictions belong to another manifest.")
    manifest = {row["study_id"]: row for row in load_manifest(args.manifest)}
    config = {"reports_sha256": file_digest(args.reports), "max_findings": args.max_findings,
              "maira_config_sha256": file_digest(args.model / "config.json"), "device": args.device}
    saved = resume(args.output, config, "claim_id")
    model = None
    for record in reports["rows"]:
        if set(record["reports"]) != set(reports["config"]["variants"]):
            raise ValueError("Finish all report variants before running grounding.")
        study = manifest[record["study_id"]]
        image = image_at(args.images, study["image"])
        if digest_image(image) != record["image_sha256"]:
            raise ValueError(f"Grounding image differs from report image: {study['study_id']}.")
        for variant, report in record["reports"].items():
            for index, claim in enumerate(findings(report), 1):
                ident = f"{study['study_id']}:{variant}:{index}"
                if ident in saved:
                    continue
                row = {"claim_id": ident, "study_id": study["study_id"], "split": study["split"],
                       "variant": variant, "index": index, "claim": claim, "boxes": [], "grounding_text": []}
                if index > args.max_findings:
                    row["ground_status"] = "over_limit"
                elif UNLOCALIZED.search(claim):
                    row["ground_status"] = "nonlocal_claim"
                else:
                    if model is None:
                        model = MairaPredictor(str(args.model), args.device)
                    inputs = model.processor.format_and_preprocess_phrase_grounding_input(
                        frontal_image=image, phrase=claim, return_tensors="pt",
                    )
                    try:
                        result = model.generate(inputs, 150)
                    except MairaOutputError:
                        row["ground_status"] = "malformed_output"
                        print(f"{ident} malformed_output", flush=True)
                    else:
                        row["boxes"] = pixel_boxes(model.processor, image.size,
                                                   [box for _, group in result for box in (group or ())])
                        row["grounding_text"] = [str(text) for text, _ in result]
                        row["ground_status"] = "grounded" if row["boxes"] else "no_box"
                saved[ident] = row
                save(args.output, config, saved)
        print(f"{study['study_id']} grounding complete", flush=True)
    save(args.output, config, saved)


def interventions(image, boxes, radius):
    from evidence.perturbation import matched_area_mask, matched_mask, perturb

    mask = Image.new("L", image.size)
    for x1, y1, x2, y2 in boxes:
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("Invalid grounding box for this image.")
        mask.paste(255, (x1, y1, x2, y2))
    control = matched_mask(mask)
    match = "same_shape"
    if control is None:
        control = matched_area_mask(mask)
        match = "same_area"
    if control is None:
        return None, None, None
    return perturb(image, mask, radius), perturb(image, control, radius), match


class VisualVerifier:
    def __init__(self, model, device, edge):
        from evidence.vqa.model import VQAPredictor

        self.predictor = VQAPredictor(str(model), device=device)
        self.edge = edge
        tokenizer = self.predictor.processor.tokenizer
        ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
        if any(len(tokens) != 1 for tokens in ids) or len({tokens[0] for tokens in ids}) != 3:
            raise ValueError("Verifier expects Qwen's distinct single-token A, B and C choices.")
        self.choice_ids = [tokens[0] for tokens in ids]

    def __call__(self, image, claim):
        p = self.predictor
        instruction = ("Assess whether this exact statement is supported by the current chest X-ray. "
                       "Preserve laterality, uncertainty and negation. Use the image only. "
                       "Choose A for supported, B for contradicted, C when evidence is insufficient. "
                       f"Output only A, B or C.\nStatement: {claim}")
        content = [{"type": "image", "image": resized(image, self.edge)}, {"type": "text", "text": instruction}]
        batch = p.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(p.device)
        with p.torch.inference_mode():
            output = p.model(**batch, logits_to_keep=1, use_cache=False)
            scores = output.logits[0, -1, self.choice_ids].float().softmax(0).tolist()
        return dict(zip(("support", "conflict", "uncertain"), scores, strict=True))


def probe(args):
    reports, grounding = read_json(args.reports), read_json(args.grounding)
    if (reports["config"]["manifest_sha256"] != file_digest(args.manifest)
            or grounding["config"]["reports_sha256"] != file_digest(args.reports)):
        raise ValueError("Manifest, reports and grounding do not belong to the same run.")
    studies = {row["study_id"]: row for row in load_manifest(args.manifest)}
    hashes = {row["study_id"]: row["image_sha256"] for row in reports["rows"]}
    protocol = {"version": 1, "verifier_config_sha256": file_digest(args.model / "config.json"),
                "maira_config_sha256": grounding["config"]["maira_config_sha256"], "device": args.device,
                "image_edge": args.image_edge, "radius_fraction": args.radius_fraction, "alignment": args.alignment}
    config = {"grounding_sha256": file_digest(args.grounding), "reports_sha256": file_digest(args.reports),
              "protocol": protocol}
    saved = resume(args.output, config, "claim_id")
    verifier, alignment = None, None
    previous_id = None
    for row in grounding["rows"]:
        ident, claim = row["study_id"], row["claim"]
        if row["claim_id"] in saved:
            continue
        if ident != previous_id:
            image = image_at(args.images, studies[ident]["image"])
            if digest_image(image) != hashes[ident]:
                raise ValueError(f"Audit image differs from report image: {ident}.")
            previous_id = ident
        radius = max(1., min(image.size) * args.radius_fraction)
        result = {**row, "features": None, "checks": None, "control_match": None, "probe_status": row["ground_status"]}
        if row["boxes"]:
            target, control, match = interventions(image, row["boxes"], radius)
            result.update(control_match=match, radius=radius, probe_status="no_control")
            if match:
                if verifier is None:
                    verifier = VisualVerifier(args.model, args.device, args.image_edge)
                checks = {key: verifier(variant, claim) for key, variant in zip(
                    ("original", "target", "control"), (image, target, control), strict=True)}
                features = {"original_support": checks["original"]["support"],
                            "original_conflict": checks["original"]["conflict"],
                            "relative_support_drop": checks["control"]["support"] - checks["target"]["support"]}
                if args.alignment:
                    if alignment is None:
                        from evidence.perturbation import BiomedAlignment

                        alignment = BiomedAlignment(args.device)
                    original, at_target, at_control = alignment([image, target, control], claim)
                    features.update(original_alignment=original, relative_alignment_drop=at_control - at_target)
                result.update(features=features, checks=checks, probe_status="probed")
        saved[row["claim_id"]] = result
        save(args.output, config, saved)
        print(f"{row['claim_id']} {result['probe_status']}", flush=True)
    save(args.output, config, saved)


def summarize_probes(args):
    grouped = defaultdict(list)
    for row in read_json(args.probes)["rows"]:
        grouped[row["split"], row["variant"]].append(row)
    by_split = defaultdict(dict)
    for (split, variant), rows in sorted(grouped.items()):
        differences = [row["features"]["relative_support_drop"] for row in rows
                       if row["probe_status"] == "probed" and row["features"] is not None]
        by_split[split][variant] = {
            "study_count": len({row["study_id"] for row in rows}),
            "findings": len(rows),
            "with_matched_control": len(differences),
            "coverage": len(differences) / len(rows),
            "probe_status_counts": dict(Counter(row["probe_status"] for row in rows)),
            "mean_control_minus_target_support": mean(differences) if differences else None,
            "median_control_minus_target_support": median(differences) if differences else None,
        }
    write_json(args.output, {"probes_sha256": file_digest(args.probes), "by_split": by_split})


def review_id(claim_id):
    return hashlib.sha256(claim_id.encode()).hexdigest()[:24]


def write_review(args):
    rows = read_json(args.probes)["rows"]
    rng = random.Random(42)
    shuffled = sorted(rows, key=lambda row: (row["split"], row["study_id"], rng.random()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ("review_id", "study_id", "split", "claim", "unsupported", "notes")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "review_id": review_id(row["claim_id"])} for row in shuffled)


def fit(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    probes = read_json(args.probes)
    protocol = probes["config"]["protocol"]
    rows = {row["claim_id"]: row for row in probes["rows"]}
    review_ids = {review_id(ident): ident for ident in rows}
    labels, seen = {}, set()
    with args.review.open(encoding="utf-8-sig", newline="") as stream:
        for line in csv.DictReader(stream):
            ident, value = review_ids.get(line["review_id"]), line["unsupported"].strip()
            row = rows.get(ident)
            if not row or ident in seen or any(line[k] != row[k] for k in ("study_id", "split", "claim")):
                raise ValueError(f"Invalid, altered or duplicate review row: {line['review_id']}.")
            seen.add(ident)
            if value not in {"", "uncertain", "0", "1"}:
                raise ValueError(f"Use 0, 1, uncertain or blank for {ident}.")
            if value in {"0", "1"}:
                labels[ident] = int(value)

    def subset(split):
        return [row for ident, row in rows.items() if row["split"] == split and row["features"] and ident in labels]

    validation, test = subset("validation"), subset("test")
    y_val = [labels[row["claim_id"]] for row in validation]
    y_test = [labels[row["claim_id"]] for row in test]
    if set(y_val) != {0, 1}:
        raise ValueError("Validation needs both 0 and 1 on rows with usable counterfactual features.")
    if set(y_test) != {0, 1}:
        raise ValueError("Test needs both 0 and 1 on usable rows for AP and AUROC.")
    full = FEATURES + (ALIGNMENT if protocol["alignment"] else ())
    metrics, estimators = {}, {}
    for name, keys in (("original_only", FEATURES[:2]), ("counterfactual", full)):
        matrix = [[row["features"][key] for key in keys] for row in validation]
        scaler = StandardScaler().fit(matrix)
        model = LogisticRegression(random_state=42, max_iter=1000).fit(scaler.transform(matrix), y_val)
        test_matrix = [[row["features"][key] for key in keys] for row in test]
        pred = model.predict_proba(scaler.transform(test_matrix))[:, 1]
        metrics[name] = {"ap": float(average_precision_score(y_test, pred)), "auroc": float(roc_auc_score(y_test, pred))}
        estimators[name] = {"protocol": protocol, "keys": keys, "mean": scaler.mean_.tolist(),
                            "scale": scaler.scale_.tolist(), "weights": model.coef_[0].tolist(),
                            "intercept": float(model.intercept_[0])}
    test_rows = [row for row in rows.values() if row["split"] == "test"]
    counts = defaultdict(lambda: {"claims": 0, "with_features": 0, "labeled_evaluable": 0})
    for row in test_rows:
        entry = counts[row["variant"]]
        entry["claims"] += 1
        entry["with_features"] += bool(row["features"])
        entry["labeled_evaluable"] += bool(row["features"] and row["claim_id"] in labels)
    write_json(args.estimator, estimators["counterfactual"])
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.predictions.with_suffix(args.predictions.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("claim_id", "study_id", "split", "variant", "claim", "probe_status", "unsupported", "risk_score")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows.values():
            writer.writerow({**row, "unsupported": labels.get(row["claim_id"], ""),
                             "risk_score": risk(row["features"], estimators["counterfactual"])})
    temporary.replace(args.predictions)
    write_json(args.output, {"train_validation_claims": len(y_val), "test_labeled_evaluable": len(y_test),
                            "test_positive_rate": mean(y_test), "test_by_variant": counts, "methods": metrics,
                            "probes_sha256": file_digest(args.probes), "review_sha256": file_digest(args.review)})


def risk(features, estimator):
    import math

    if estimator is None or features is None:
        return None
    logit = estimator["intercept"] + sum((features[key] - mu) / scale * weight for key, mu, scale, weight in zip(
        estimator["keys"], estimator["mean"], estimator["scale"], estimator["weights"], strict=True))
    return 1 / (1 + math.exp(-min(700, max(-700, logit))))
