"""Train and evaluate a small 2D U-Net for one SLAKE organ and modality."""

import argparse
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..dataset import InvalidMask, Slake
from ..paths import CHECKPOINTS, DATA, RESULTS
from evidence.storage import write_json


def conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())


class UNet(nn.Module):
    def __init__(self, classes=1):
        super().__init__()
        self.encoder1 = conv(1, 16)
        self.encoder2 = conv(16, 32)
        self.bridge = conv(32, 64)
        self.decoder2 = conv(96, 32)
        self.decoder1 = conv(48, 16)
        self.output = nn.Conv2d(16, classes, 1)

    def forward(self, image):
        x1 = self.encoder1(image)
        x2 = self.encoder2(F.max_pool2d(x1, 2))
        bridge = self.bridge(F.max_pool2d(x2, 2))
        y2 = self.decoder2(torch.cat((F.interpolate(bridge, size=x2.shape[-2:], mode="bilinear", align_corners=False), x2), dim=1))
        y1 = self.decoder1(torch.cat((F.interpolate(y2, size=x1.shape[-2:], mode="bilinear", align_corners=False), x1), dim=1))
        return self.output(y1)


def image_tensor(image, size):
    return torch.from_numpy(np.array(image.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)[None] / 255)


class OrganImages(Dataset):
    def __init__(self, slake, split, modality, organ, size):
        self.images = []
        image_ids = dict.fromkeys(row.image_id for row in slake.questions(split) if row.modality == modality)
        code = slake.labels[organ.casefold()]
        for image_id in image_ids:
            if (image_id, "mask") not in slake.members:
                continue
            try:
                original_mask = np.asarray(slake.mask_values(image_id)) == code
            except InvalidMask:
                continue
            if not original_mask.any():
                continue
            mask = Image.fromarray(original_mask).resize((size, size), Image.Resampling.NEAREST)
            self.images.append((image_id, image_tensor(slake.image(image_id), size),
                                torch.from_numpy(np.array(mask, dtype=np.float32)[None])))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.images[item]


def overlap_metrics(model, loader, device):
    dice, iou = [], []
    model.eval()
    with torch.inference_mode():
        for _, image, mask in loader:
            pred = model(image.to(device)).cpu() > 0
            target = mask.bool()
            for p, t in zip(pred, target):
                intersection = (p & t).sum().item()
                union = (p | t).sum().item()
                dice.append(2 * intersection / (p.sum().item() + t.sum().item()) if union else 1.0)
                iou.append(intersection / union if union else 1.0)
    return {"images": len(dice), "dice": mean(dice), "iou": mean(iou)}


class MaskPredictor:
    def __init__(self, checkpoint, *, device="cpu"):
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.organs = tuple(weights["organs"]) if "organs" in weights else (weights["organ"],)
        self.organ = self.organs[0] if len(self.organs) == 1 else None
        self.classes = {name.casefold(): index for index, name in enumerate(self.organs, start=1)}
        self.multiclass = "organs" in weights
        self.modality = weights["modality"]
        self.size = weights["size"]
        self.device = torch.device(device)
        self.model = UNet(len(self.organs) + 1 if self.multiclass else 1).eval()
        self.model.load_state_dict(weights["model"])
        self.model.to(self.device)

    def supports(self, modality, organ):
        return modality == self.modality and organ.strip().casefold() in self.classes

    def mask(self, image, organ=None):
        if organ is None:
            organ = self.organ
        if not organ or organ.strip().casefold() not in self.classes:
            raise ValueError(f"Choose a checkpoint organ: {', '.join(self.organs)}.")
        with torch.inference_mode():
            logits = self.model(image_tensor(image, self.size)[None].to(self.device))
            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
            if self.multiclass:
                mask = logits[0].argmax(dim=0) == self.classes[organ.strip().casefold()]
            else:
                mask = logits[0, 0] > 0
        return Image.fromarray(mask.cpu().numpy()).convert("L")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    organs = parser.add_mutually_exclusive_group()
    organs.add_argument("--organ", default="left lung")
    organs.add_argument("--organs", nargs="+", help="Train a multi-class model on two or more named SLAKE labels.")
    parser.add_argument("--modality", default="CT")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--unweighted", action="store_true", help="Ablate class weights in pixel BCE or cross-entropy.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--test", action="store_true", help="Also evaluate on held-out test after training settings are frozen.")
    args = parser.parse_args()
    if args.epochs < 1 or args.size < 16 or args.size % 4 or args.threads < 1:
        parser.error("Use positive --epochs/--threads and --size divisible by 4 and >=16.")
    torch.manual_seed(42)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = "unet_ct_multi" if args.organs else "unet"
    summary = "unet_ct_multi" if args.organs else "segmentation"
    if args.unweighted:
        checkpoint += "-no-weight"
        summary += "-no-weight"
    args.checkpoint = args.checkpoint or CHECKPOINTS / f"{checkpoint}.pt"
    args.result = args.result or RESULTS / f"{summary}_summary.json"
    if args.organs:
        from .multiclass import train

        args.organs = tuple(organ.strip().casefold() for organ in args.organs)
        if len(set(args.organs)) != len(args.organs) or len(args.organs) < 2:
            parser.error("--organs needs at least two distinct organ names.")
        train(args, device)
        return
    with Slake(args.root) as slake:
        splits = ["train", "validation"] + (["test"] if args.test else [])
        datasets = {split: OrganImages(slake, split, args.modality.upper(), args.organ, args.size) for split in splits}
    if any(not len(data) for data in datasets.values()):
        raise ValueError(f"Missing organ/mode masks for {[split for split, data in datasets.items() if not len(data)]}. Check slake_inventory.json.")
    seen = set()
    for data in datasets.values():
        ids = {image_id for image_id, _, _ in data.images}
        if seen & ids:
            raise ValueError("Image ID overlap across segmentation splits.")
        seen.update(ids)
    loaders = {split: DataLoader(data, batch_size=8, shuffle=split == "train") for split, data in datasets.items()}
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    foreground = mean(mask.mean().item() for _, _, mask in datasets["train"].images)
    if not 0 < foreground < 1:
        raise ValueError("Training masks need foreground and background pixels after resizing.")
    weight = 1.0 if args.unweighted else min(20.0, ((1 - foreground) / foreground) ** 0.5)
    pos_weight = torch.tensor(weight, device=device)
    best_dice, best, history = -1, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _, image, mask in loaders["train"]:
            optimizer.zero_grad()
            mask = mask.to(device)
            logits = model(image.to(device))
            prob = logits.sigmoid()
            soft_dice = (2 * (prob * mask).sum() + 1) / (prob.sum() + mask.sum() + 1)
            loss = F.binary_cross_entropy_with_logits(logits, mask, pos_weight=pos_weight) + 1 - soft_dice
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = overlap_metrics(model, loaders["validation"], device)
        row = {"epoch": epoch, "loss": mean(losses), "validation": validation}
        history.append(row)
        print(row, flush=True)
        if validation["dice"] > best_dice:
            best_dice = validation["dice"]
            best = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "organ": args.organ, "modality": args.modality.upper(), "size": args.size, "epoch": epoch}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    torch.save(best, temporary)
    temporary.replace(args.checkpoint)
    model.load_state_dict(best["model"])
    report = {"dataset": "BoKelvin/SLAKE", "organ": args.organ, "modality": args.modality.upper(),
              "device": str(device), "size": args.size, "epochs": args.epochs, "train_images": len(datasets["train"]),
              "included_images": "positive annotated organ masks only", "best_epoch": best["epoch"],
              "train_foreground_fraction": foreground, "bce_positive_weight": weight,
              "history": history, "validation": overlap_metrics(model, loaders["validation"], device)}
    if args.test:
        report["test"] = overlap_metrics(model, loaders["test"], device)
    write_json(args.result, report)


if __name__ == "__main__":
    main()
