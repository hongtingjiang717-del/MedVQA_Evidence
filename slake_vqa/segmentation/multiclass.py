"""Train a multi-class CT U-Net from the original SLAKE label map."""

from statistics import mean

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..dataset import InvalidMask, Slake
from .unet import UNet, image_tensor
from evidence.storage import write_json


class MultiOrganImages(Dataset):
    def __init__(self, slake, split, modality, organs, size):
        self.images = []
        lookup = np.zeros(256, dtype=np.uint8)
        for index, organ in enumerate(organs, start=1):
            lookup[slake.labels[organ]] = index
        ids = dict.fromkeys(row.image_id for row in slake.questions(split) if row.modality == modality)
        for image_id in ids:
            if (image_id, "mask") not in slake.members:
                continue
            try:
                mask = slake.mask_values(image_id)
            except InvalidMask:
                continue
            labels = lookup[np.asarray(mask.resize((size, size), Image.Resampling.NEAREST))]
            self.images.append((image_id, image_tensor(slake.image(image_id), size), torch.from_numpy(labels)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def metrics(model, loader, device, organs):
    by_organ = {organ: [] for organ in organs}
    false_positives = dict.fromkeys(organs, 0)
    model.eval()
    with torch.inference_mode():
        for _, images, targets in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            for label, organ in enumerate(organs, 1):
                for prediction, target in zip(predicted == label, targets == label, strict=True):
                    if not target.any():
                        false_positives[organ] += int(prediction.any())
                        continue
                    intersection = (prediction & target).sum().item()
                    by_organ[organ].append((2 * intersection / (prediction.sum().item() + target.sum().item()),
                                            intersection / (prediction | target).sum().item()))
    scores = {organ: {"positive_images": len(pairs),
                      "absent_label_images": len(loader.dataset) - len(pairs),
                      "absent_label_images_with_prediction": false_positives[organ],
                      "dice": mean(dice for dice, _ in pairs) if pairs else None,
                      "iou": mean(iou for _, iou in pairs) if pairs else None}
              for organ, pairs in by_organ.items()}
    observed = [row for row in scores.values() if row["positive_images"]]
    return {"images": len(loader.dataset), "per_organ": scores,
            "macro_dice": mean(row["dice"] for row in observed) if observed else None,
            "macro_iou": mean(row["iou"] for row in observed) if observed else None}


def train(args, device):
    modality = args.modality.strip().upper()
    organs = args.organs
    n_classes = len(organs) + 1
    with Slake(args.root) as data:
        unknown = set(organs) - data.labels.keys()
        if unknown:
            raise ValueError(f"Unknown SLAKE labels: {', '.join(sorted(unknown))}.")
        splits = ("train", "validation", "test") if args.test else ("train", "validation")
        datasets = {split: MultiOrganImages(data, split, modality, organs, args.size) for split in splits}
    seen = set()
    presence = {}
    for split, dataset in datasets.items():
        ids = {image_id for image_id, _, _ in dataset.images}
        if not ids or seen & ids:
            raise ValueError(f"Empty or overlapping multi-class segmentation split: {split}.")
        seen.update(ids)
        presence[split] = {organ: sum(bool((labels == index).any()) for _, _, labels in dataset.images)
                           for index, organ in enumerate(organs, 1)}
        if split != "test" and not all(presence[split].values()):
            raise ValueError(f"Each selected label needs {split} examples after resizing: {presence[split]}.")

    loaders = {split: DataLoader(data, batch_size=8, shuffle=split == "train") for split, data in datasets.items()}
    model = UNet(n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    counts = sum(torch.bincount(labels.flatten().long(), minlength=n_classes) for _, _, labels in datasets["train"].images)
    if not counts[0]:
        raise ValueError("Training needs background pixels for the selected labels.")
    weights = torch.ones(n_classes, device=device) if args.unweighted else (
        (counts[0] / counts).sqrt().clamp(max=20).to(device)
    )
    best_dice, best, history = -1, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for _, images, targets in loaders["train"]:
            optimizer.zero_grad()
            targets = targets.to(device=device, dtype=torch.long)
            logits = model(images.to(device))
            probs = logits.softmax(dim=1)[:, 1:]
            reference = F.one_hot(targets, n_classes).movedim(-1, 1)[:, 1:]
            axes = (0, 2, 3)
            dice = (2 * (probs * reference).sum(axes) + 1) / (probs.sum(axes) + reference.sum(axes) + 1)
            loss = F.cross_entropy(logits, targets, weight=weights) + 1 - dice.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = metrics(model, loaders["validation"], device, organs)
        history.append({"epoch": epoch, "loss": mean(losses), "validation": validation})
        print(f"epoch {epoch}/{args.epochs}: loss={mean(losses):.4f} val_macro_dice={validation['macro_dice']:.4f}", flush=True)
        if validation["macro_dice"] > best_dice:
            best_dice = validation["macro_dice"]
            best = {"model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "organs": organs, "modality": modality, "size": args.size, "epoch": epoch}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    torch.save(best, temporary)
    temporary.replace(args.checkpoint)
    model.load_state_dict(best["model"])
    report = {"dataset": "BoKelvin/SLAKE", "modality": modality, "organs": organs, "device": str(device),
              "size": args.size, "epochs": args.epochs, "best_epoch": best["epoch"], "positive_images": presence,
              "loss": "cross-entropy + foreground macro soft Dice",
              "cross_entropy_class_weights": dict(zip(("background", *organs), weights.tolist(), strict=True)),
              "history": history}
    for split in splits[1:]:
        report[split] = metrics(model, loaders[split], device, organs)
    write_json(args.result, report)
