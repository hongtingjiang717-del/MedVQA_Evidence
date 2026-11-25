"""CLEAR-inspired localized audit with mask, box and matched-area controls."""

import argparse
from bisect import bisect_right
import csv
import json
import math
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageFilter

from evidence.perturbation import BiomedAlignment, matched_area_mask, matched_mask, perturb, spread_to_box

from .dataset import InvalidMask, Slake
from .paths import CHECKPOINTS, DATA, EXAMPLES, RESULTS, default_model
from evidence.storage import file_digest, write_json
from evidence.vqa.model import VQAPredictor, normalize_answer, token_f1


def probe(model, image, question, mask=None, *, radius=12, align=None, seed=42, max_new_tokens=48,
          allow_area_control=False):
    started = perf_counter()
    answer, confidence = model.predict(question, image, max_new_tokens=max_new_tokens, confidence=True)
    blurred = image.filter(ImageFilter.GaussianBlur(radius))
    variants = {"global": blurred}
    region = None
    if mask is not None and mask.getbbox():
        if mask.size != image.size:
            raise ValueError("Image and mask dimensions differ.")
        variants.update(box=perturb(image, spread_to_box(mask), radius), mask=perturb(image, mask, radius))
        control = matched_mask(mask, seed)
        match = "same_shape"
        if control is None and allow_area_control:
            control = matched_area_mask(mask)
            match = "same_area"
        region = {"target_bbox_xyxy": mask.getbbox(), "target_pixels": mask.histogram()[255], "control_pixels": None,
                  "control_match": None}
        if control is not None:
            variants["control"] = perturb(image, control, radius)
            region["control_pixels"] = control.histogram()[255]
            region["control_match"] = match
    comparisons = {}
    for name, variant in variants.items():
        other = model.answer(question, variant, max_new_tokens=max_new_tokens)
        comparisons[name] = {"answer": other, "persistence": token_f1(answer, other) if answer else None}
    if align is not None and answer:
        claim = f"Question: {question} Answer: {answer}"
        original, *scores = align([image, *variants.values()], claim)
        for name, score in zip(variants, scores):
            comparisons[name].update(alignment_original=original, alignment_delta=original - score)
    mask_effect = None
    if answer and "control" in comparisons:
        mask_effect = comparisons["control"]["persistence"] - comparisons["mask"]["persistence"]
    return {"answer": answer, "confidence": confidence, "variants": comparisons,
            "region": region, "target_specific_change": mask_effect, "seconds": perf_counter() - started}


FEATURES = ("confidence", "global", "box", "mask", "control", "target_specific_change", "alignment_mask", "alignment_control")


def review_row(result):
    variants = result["variants"]
    return {"confidence": result["confidence"], "target_specific_change": result["target_specific_change"],
            **{name: variants.get(name, {}).get("persistence") for name in ("global", "box", "mask", "control")},
            **{f"alignment_{name}": variants.get(name, {}).get("alignment_delta") for name in ("mask", "control")}}


def write_review(path, cases):
    columns = ("split", "qid", "image_id", "organ", "question", "answer", "mask_source", "control_match",
               "unsupported", "notes", *FEATURES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path if not path.exists() else path.with_name(path.stem + ".new.csv")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "control_match": (row["region"] or {}).get("control_match"), **review_row(row)}
                         for row in cases)
    print(path)


def estimate_risk(result, parameters):
    return score_cues(review_row(result), parameters)


def cue_vector(cues, keys, reference):
    values = [cues.get(key) for key in keys]
    if any(value in (None, "") for value in values):
        return None
    values = [float(value) for value in values]
    if reference:
        values += [bisect_right(reference[key], value) / len(reference[key])
                   for key, value in zip(keys, values, strict=True)]
    return values


def score_cues(cues, parameters):
    values = cue_vector(cues, parameters["features"], parameters.get("percentile_reference"))
    if values is None:
        return None
    terms = ((value - center) / scale * weight for value, center, scale, weight in
             zip(values, parameters["mean"], parameters["scale"], parameters["weights"], strict=True))
    logit = parameters["intercept"] + sum(terms)
    return 1 / (1 + math.exp(-max(-700, min(700, logit))))


def assess(review, output, *, use_alignment, estimator=None, predictions=None, normalization="standard"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    with review.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if any(row["split"] not in {"validation", "test"} for row in rows):
        raise ValueError("Review supports validation (fit) and test (evaluate) only.")
    if len({(row["split"], row["qid"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate review question IDs.")
    validation_images = {row["image_id"] for row in rows if row["split"] == "validation"}
    test_images = {row["image_id"] for row in rows if row["split"] == "test"}
    if validation_images & test_images:
        raise ValueError("Validation and test overlap by image ID.")
    features = FEATURES if use_alignment else FEATURES[:-2]
    labeled = [row for row in rows if (row.get("unsupported") or "").strip() in {"0", "1"}]
    complete = [row for row in labeled if cue_vector(row, features, None) is not None]
    train = [row for row in complete if row["split"] == "validation"]
    test = [row for row in complete if row["split"] == "test"]
    y_train, y_test = ([int(row["unsupported"]) for row in group] for group in (train, test))
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        raise ValueError("Validation and test each need both classes with complete cues.")

    sources = {row.get("mask_source") for row in rows}
    if "maira" in sources and sources != {"maira"}:
        raise ValueError("Review MAIRA-2 and other audit region sources in separate evaluation files.")
    if estimator and sources not in ({"unet"}, {"maira"}):
        raise ValueError("Export a service estimator from one U-Net or MAIRA-2 region source per review.")

    reference = None
    if normalization == "percentile":
        reference_rows = [row for row in rows if row["split"] == "validation"
                          and cue_vector(row, features, None) is not None]
        reference = {key: sorted(float(row[key]) for row in reference_rows) for key in features}

    def matrix(rows, keys):
        return [cue_vector(row, keys, reference) for row in rows]

    control_name = "same_area_control" if sources == {"maira"} else "random_region"
    feature_sets = {"global_only": ("confidence", "global"), "box": ("confidence", "global", "box"),
                    control_name: ("confidence", "global", "control"), "mask": ("confidence", "global", "mask"),
                    "matched_control": features}
    if use_alignment:
        feature_sets["response_only"] = FEATURES[:-2]
    scores = {"low_confidence": [1 - float(row["confidence"]) for row in test]}
    for name, keys in feature_sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        model.fit(matrix(train, keys), y_train)
        scores[name] = model.predict_proba(matrix(test, keys))[:, 1]
        if name == "matched_control":
            scaler, linear = model.steps[0][1], model.steps[1][1]
            parameters = {"features": keys, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                          "weights": linear.coef_[0].tolist(), "intercept": float(linear.intercept_[0])}
            if reference is not None:
                parameters["percentile_reference"] = reference
            if estimator:
                parameters["mask_source"] = next(iter(sources))
                parameters["organs"] = sorted({row["organ"].strip().casefold() for row in train if row.get("organ")})
                write_json(estimator, parameters)
    metrics = {name: {"auprc_ap": average_precision_score(y_test, score), "auroc": roc_auc_score(y_test, score)}
               for name, score in scores.items()}
    result = {"validation_fit": len(train), "test_evaluated": len(test), "test_positive_rate": sum(y_test) / len(y_test),
              "excluded_missing_features": len(labeled) - len(complete), "features": features,
              "normalization": normalization, "metrics": metrics}
    if reference is not None:
        result["validation_percentile_reference"] = len(reference_rows)
    if predictions:
        columns = ("qid", "image_id", "question", "answer", "unsupported", "risk_score", "score_status")
        predictions.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        temporary = predictions.with_suffix(predictions.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                if row["split"] != "test":
                    continue
                risk = score_cues(row, parameters)
                count += int(risk is not None)
                writer.writerow({**row, "risk_score": risk,
                                 "score_status": "scored" if risk is not None else "missing_features"})
        temporary.replace(predictions)
        result["test_scored"] = count
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates", help="Export organ-linked cases for human review.")
    candidates.add_argument("--root", type=Path, default=DATA)
    candidates.add_argument("--organ", required=True)
    candidates.add_argument("--modality", required=True)
    candidates.add_argument("--query-term", help="Optional broader keyword. Review its matches for anatomical relevance.")
    candidates.add_argument("--output", type=Path, default=RESULTS / "candidate_audit_cases.csv")
    run = sub.add_parser("run")
    run.add_argument("--cases", type=Path)
    run.add_argument("--root", type=Path, default=DATA)
    run.add_argument("--mask-source", choices=["annotation", "unet", "maira"], default="annotation")
    run.add_argument("--maira-results", type=Path, help="Chest X-ray predictions exported by slake_vqa maira.")
    run.add_argument("--checkpoint", type=Path, default=CHECKPOINTS / "unet.pt")
    run.add_argument("--model", default=default_model())
    run.add_argument("--adapter", type=Path)
    run.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run.add_argument("--alignment", action="store_true")
    run.add_argument("--radius", type=float, default=12)
    run.add_argument("--max-new-tokens", type=int, default=48)
    run.add_argument("--output", type=Path)
    score = sub.add_parser("score")
    score.add_argument("--review", type=Path, required=True)
    score.add_argument("--alignment", action="store_true")
    score.add_argument("--output", type=Path, default=RESULTS / "audit_summary.json")
    score.add_argument("--export-estimator", type=Path)
    score.add_argument("--predictions", type=Path, help="Export a CSV with a risk score for each complete test case.")
    score.add_argument("--normalization", choices=["standard", "percentile"], default="standard",
                       help="Append empirical percentiles computed on validation before fitting the scorer.")
    args = parser.parse_args()
    if args.command == "candidates":
        export_candidates(args.root, args.organ, args.modality, args.output, query_term=args.query_term)
        return
    if args.command == "score":
        paths = [path.resolve() for path in (args.review, args.output, args.export_estimator, args.predictions) if path]
        if len(set(paths)) != len(paths):
            parser.error("Use distinct files for the review and each output.")
        assess(args.review, args.output, use_alignment=args.alignment, estimator=args.export_estimator,
               predictions=args.predictions, normalization=args.normalization)
        return
    if args.radius <= 0 or args.max_new_tokens < 1:
        parser.error("--radius and --max-new-tokens must be positive.")
    if args.mask_source == "maira":
        args.maira_results = args.maira_results or RESULTS / "maira_cardiomegaly.json"
        if not args.maira_results.is_file():
            parser.error("Run python -m slake_vqa maira first, or specify --maira-results.")
    elif args.maira_results:
        parser.error("--maira-results requires --mask-source maira.")
    args.cases = args.cases or EXAMPLES / ("chest-audit.csv" if args.mask_source == "maira" else "audit.csv")
    output_names = {"annotation": "audit_cases.json", "unet": "audit-unet.json", "maira": "audit-chest.json"}
    args.output = args.output or RESULTS / output_names[args.mask_source]
    if args.output.exists():
        parser.error("Choose a new output path to preserve the existing audit.")
    if args.mask_source == "unet":
        from .segmentation.unet import MaskPredictor

        segmenter = MaskPredictor(args.checkpoint, device=args.device)
    chest = None
    if args.maira_results:
        from .maira import CARDIOMEGALY, CXR, ChestRegions

        chest = ChestRegions(args.maira_results)
    region_file = args.maira_results or (args.checkpoint if args.mask_source == "unet" else None)
    region_provenance = {"path": str(region_file), "sha256": file_digest(region_file)} if region_file else None
    model = VQAPredictor(args.model, adapter=args.adapter, device=args.device)
    alignment = BiomedAlignment(args.device) if args.alignment else None
    with args.cases.open(encoding="utf-8-sig", newline="") as f:
        selected = list(csv.DictReader(f))
    cases = []
    with Slake(args.root) as slake:
        questions = {(split, row.qid): row for split in ("validation", "test") for row in slake.questions(split)}
        seen = set()
        for entry in selected:
            key = entry["split"].strip(), entry["qid"].strip()
            if key in seen or key not in questions:
                raise ValueError(f"Invalid or duplicate audit case {key}.")
            seen.add(key)
            row = questions[key]
            organ = entry["organ"].strip()
            image = slake.image(row.image_id)
            localization = None
            if args.mask_source == "annotation":
                mask = slake.mask(row.image_id, organ)
            elif chest:
                if (row.modality, organ.casefold()) != (CXR, CARDIOMEGALY):
                    raise ValueError(f"MAIRA-2 audit currently supports {CXR}/{CARDIOMEGALY}: {key}.")
                mask, localization = chest.region(row.image_id, image)
                if mask is None:
                    raise ValueError(f"No MAIRA-2 result for {row.image_id}. Run the matching chest X-ray split.")
            else:
                if not segmenter.supports(row.modality, organ):
                    raise ValueError(f"Segmentation checkpoint does not cover {key}: {row.modality}/{organ}.")
                mask = segmenter.mask(image, organ)
            result = probe(model, image, row.question, mask, radius=args.radius, align=alignment,
                           max_new_tokens=args.max_new_tokens, allow_area_control=chest is not None)
            if localization is not None:
                result["localization"] = localization
            cases.append({"split": row.split, "qid": row.qid, "image_id": row.image_id, "organ": organ,
                          "mask_source": args.mask_source,
                          "question": row.question, "reference_answer": row.answer, **result})
            write_json(args.output, {"model": args.model, "adapter": str(args.adapter) if args.adapter else None,
                       "mask_source": args.mask_source, "region_file": region_provenance,
                       "radius": args.radius, "max_new_tokens": args.max_new_tokens,
                       "cases": cases})
            print(f"{row.split}/{row.qid}: {result['answer']}", flush=True)
    if not cases:
        parser.error("No audit cases in the supplied CSV.")
    write_review(args.output.with_suffix(".csv"), cases)


def export_candidates(root, organ, modality, output, *, query_term=None):
    from .maira import CARDIOMEGALY, CXR

    organ_text = normalize_answer(query_term or organ)
    if not organ_text:
        raise ValueError("Choose a non-empty organ keyword.")
    rows = []
    covered = {}
    chest = (modality.upper(), organ.strip().casefold()) == (CXR, CARDIOMEGALY)
    with Slake(root) as slake:
        for split in ("validation", "test"):
            for row in slake.questions(split):
                text = normalize_answer(row.question + " " + row.answer)
                if row.modality != modality.upper() or f" {organ_text} " not in f" {text} ":
                    continue
                if chest:
                    covered[row.image_id] = True
                if row.image_id not in covered:
                    try:
                        covered[row.image_id] = bool(slake.mask(row.image_id, organ).getbbox())
                    except (InvalidMask, KeyError, ValueError):
                        covered[row.image_id] = False
                if covered[row.image_id]:
                    rows.append({"split": split, "qid": row.qid, "organ": organ, "image_id": row.image_id,
                                 "question": row.question, "reference_answer": row.answer})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("split", "qid", "organ", "image_id", "question", "reference_answer"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} candidates for review: {output}")


if __name__ == "__main__":
    main()
