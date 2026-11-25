"""Image perturbations and alignment shared by the two evidence audits."""

import math
import random

from PIL import Image, ImageFilter


def spread_to_box(mask):
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError("Empty target mask.")
    box = Image.new("L", mask.size)
    box.paste(255, bbox)
    return box


def matched_mask(mask, seed=42):
    """Translate the same shape without cropping or touching the target mask."""
    from PIL import ImageChops

    bbox = mask.getbbox()
    if bbox is None:
        return None
    shape = mask.crop(bbox)
    max_x = mask.width - shape.width
    max_y = mask.height - shape.height
    rng = random.Random(seed)
    origins = [(rng.randint(0, max_x), rng.randint(0, max_y)) for _ in range(96)]
    origins += [(x, y) for x in (0, max_x) for y in (0, max_y)]
    for origin in origins:
        trial = Image.new("L", mask.size)
        trial.paste(shape, origin)
        if not ImageChops.multiply(trial, mask).getbbox():
            return trial
    return None


def matched_area_mask(mask):
    """Choose a nonoverlapping same-area region when translation cannot fit."""
    bbox = mask.getbbox()
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    area = mask.histogram()[255]
    aspect = (x2 - x1) / (y2 - y1)
    strips = ((0, 0, x1, mask.height), (x2, 0, mask.width, mask.height),
              (0, 0, mask.width, y1), (0, y2, mask.width, mask.height))
    best = None
    for left, top, right, bottom in strips:
        for width in range(1, min(right - left, area) + 1):
            height = (area + width - 1) // width
            if height > bottom - top:
                continue
            distance = abs(math.log(width / height / aspect))
            origin_x = left + (right - left - width) // 2
            origin_y = top + (bottom - top - height) // 2
            candidate = (distance, origin_x, origin_y, width)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _, x, y, width = best
    rows, extra = divmod(area, width)
    control = Image.new("L", mask.size)
    control.paste(255, (x, y, x + width, y + rows))
    if extra:
        control.paste(255, (x, y + rows, x + extra, y + rows + 1))
    return control


def perturb(image, mask, radius):
    if image.size != mask.size:
        raise ValueError("Image and mask dimensions differ.")
    return Image.composite(image.filter(ImageFilter.GaussianBlur(radius)), image, mask)


class BiomedAlignment:
    def __init__(self, device="cuda"):
        import open_clip
        import torch

        name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        self.torch, self.device = torch, device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(name, device=device)
        self.model.eval()
        self.tokenize = open_clip.get_tokenizer(name)

    def __call__(self, images, claim):
        torch = self.torch
        with torch.inference_mode():
            pixels = torch.stack([self.preprocess(image) for image in images]).to(self.device)
            text = self.tokenize([claim], context_length=256).to(self.device)
            image_features = torch.nn.functional.normalize(self.model.encode_image(pixels), dim=-1)
            text_features = torch.nn.functional.normalize(self.model.encode_text(text), dim=-1)
        return (image_features @ text_features.T).flatten().tolist()
