"""Shared MAIRA-2 inference and original-image box coordinates."""

import math
import re


MODEL = "microsoft/maira-2"
CARDIOMEGALY = "cardiomegaly"
CXR = "X-RAY"
FINDING = re.compile(r"\b(?:cardiomegaly|enlarged (?:heart|cardiac silhouette)|"
                     r"enlargement of (?:the )?(?:heart|cardiac silhouette)|"
                     r"(?:heart(?: size)?|cardiac silhouette)"
                     r"(?: is| appears| remains)?(?: mildly| slightly| moderately| severely)? enlarged)\b", re.I)
UNCERTAINTY = re.compile(r"\b(?:possible|possibly|probable|probably|may|might|could|suspected|questionable|"
                         r"borderline|(?:cannot|can't) exclude)\b", re.I)
NEGATION = re.compile(r"\b(?:no|without|negative for|not|absent|resolved)\b", re.I)


def finding_status(sentence):
    # Keep separate clauses from turning a positive mention into a negated one.
    uncertain = False
    for clause in re.split(r";|\b(?:but|however)\b", sentence):
        match = FINDING.search(clause)
        if not match:
            continue
        left, right = clause[:match.start()], clause[match.end():]
        left = re.sub(r"\bno (?:significant |interval )*change in\b", "", left, flags=re.I)
        if NEGATION.search(UNCERTAINTY.sub("", left)) or re.match(r"\s+(?:is\s+)?(?:not|absent|resolved)\b", right, re.I):
            continue
        if UNCERTAINTY.search(left) or UNCERTAINTY.search(right):
            uncertain = True
            continue
        return "present"
    return "uncertain" if uncertain else "no_positive_mention"


def pixel_boxes(processor, image_size, boxes):
    width, height = image_size
    parsed = []
    for box in boxes or ():
        if len(box) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in box):
            continue
        x1, y1, x2, y2 = processor.adjust_box_for_original_image_size(box, width, height)
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            continue
        parsed.append((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))
    return sorted(set(box for box in parsed if box[0] < box[2] and box[1] < box[3]))


class MairaOutputError(ValueError):
    """A malformed response that contains no recoverable grounding."""


def parse_output(processor, text):
    parse = processor.convert_output_to_plaintext_or_grounded_sequence
    try:
        parsed = parse(text)
    except (AssertionError, ValueError) as error:
        start = re.escape(processor.phrase_start_token)
        end = re.escape(processor.phrase_end_token)
        complete_phrase = rf"{start}(?:(?!{start}|{end}).)*{end}"
        recovered = []
        for phrase in re.findall(complete_phrase, text, flags=re.S):
            try:
                recovered.extend(parse(phrase))
            except (AssertionError, ValueError):
                continue
        if not any(boxes for _, boxes in recovered):
            raise MairaOutputError("MAIRA-2 produced malformed output without a complete grounding box.") from error
        return recovered
    if isinstance(parsed, str):
        return [(sentence, None) for sentence in re.split(r"(?<=[.!?])\s+", parsed) if sentence.strip()]
    return parsed


class MairaPredictor:
    def __init__(self, model, device, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this Python environment.")
        self.torch, self.device = torch, device
        params = {"trust_remote_code": True, "revision": revision}
        self.processor = AutoProcessor.from_pretrained(model, **params)
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=self.dtype, use_safetensors=True, **params,
        ).eval().to(device)

    def generate(self, inputs, max_tokens):
        inputs = inputs.to(self.device, self.dtype)
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, do_sample=False)
        prefix = inputs["input_ids"].shape[-1]
        text = self.processor.decode(tokens[0][prefix:], skip_special_tokens=True).lstrip()
        return parse_output(self.processor, text)

    def predict(self, image):
        inputs = self.processor.format_and_preprocess_reporting_input(
            current_frontal=image, current_lateral=None, prior_frontal=None, indication=None,
            technique=None, comparison=None, prior_report=None, return_tensors="pt", get_grounding=True,
        )
        report = self.generate(inputs, 450)
        findings = [str(text) for text, _ in report]
        positive = [(text, boxes) for text, boxes in report if finding_status(text) == "present"]
        status = "present" if positive else ("uncertain" if any(finding_status(s) == "uncertain" for s in findings)
                                             else "no_positive_mention")
        boxes = pixel_boxes(self.processor, image.size, [box for _, item in positive for box in (item or ())])
        source = "grounded_report" if boxes else None
        if positive and not boxes:
            phrase = positive[0][0]
            inputs = self.processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=phrase, return_tensors="pt",
            )
            grounded = self.generate(inputs, 150)
            boxes = pixel_boxes(self.processor, image.size, [box for text, item in grounded
                                if finding_status(text) == "present" for box in (item or ())])
            source = "phrase_grounding" if boxes else None
        return {"finding_status": status, "report": findings, "matched_findings": [text for text, _ in positive],
                "box_source": source, "boxes_xyxy": boxes}
