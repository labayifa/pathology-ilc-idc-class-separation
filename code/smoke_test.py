"""Stage A smoke test (plan sec. 4): prove feature extraction -> linear probe -> metrics
runs end to end on UNI2-h, using BACH patches. Not meant to produce a claimable result.

Setup before running:
    huggingface-cli login     # needs HF access to MahmoodLab/UNI2-h
    pip install -r requirements.txt
    ./download_bach.sh

Usage:
    python smoke_test.py --data-dir data/bach
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

CLASSES = ["Normal", "Benign", "InSitu", "Invasive"]
IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SEED = 42


def find_patches(data_dir: Path) -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for label, cls in enumerate(CLASSES):
        cls_dir = next((p for p in data_dir.rglob(cls) if p.is_dir()), None)
        if cls_dir is None:
            raise FileNotFoundError(f"Class folder '{cls}' not found under {data_dir}")
        # macOS external drives litter directories with "._foo.tif" AppleDouble
        # metadata shadow files that match the extension filter but aren't real
        # images -- exclude anything starting with a dot.
        paths = sorted(
            p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
        )
        if not paths:
            raise FileNotFoundError(f"No images found in {cls_dir}")
        items.extend((p, label) for p in paths)
    return items


def load_model(device: str):
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    timm_kwargs = dict(
        img_size=224,
        patch_size=14,
        depth=24,
        num_heads=24,
        init_values=1e-5,
        embed_dim=1536,
        mlp_ratio=2.66667 * 2,
        num_classes=0,
        no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.inference_mode()
def extract_features(
    items: list[tuple[Path, int]], model, transform, device: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []

    def run_batch(imgs: list[torch.Tensor]) -> np.ndarray:
        x = torch.stack(imgs).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device in ("cuda", "mps")):
            out = model(x)
        return out.float().cpu().numpy()

    batch_imgs: list[torch.Tensor] = []
    batch_labels: list[int] = []
    for path, label in items:
        img = Image.open(path).convert("RGB")
        batch_imgs.append(transform(img))
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            feats.append(run_batch(batch_imgs))
            labels.extend(batch_labels)
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        feats.append(run_batch(batch_imgs))
        labels.extend(batch_labels)

    return np.concatenate(feats, axis=0), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="BACH smoke test: UNI2-h frozen features + linear probe")
    parser.add_argument("--data-dir", type=Path, default=Path("data/bach"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    t0 = time.time()
    items = find_patches(args.data_dir)
    print(f"Found {len(items)} patches across {len(CLASSES)} classes")

    model, transform = load_model(device)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    X, y = extract_features(items, model, transform, device, args.batch_size)
    print(f"Extracted features: {X.shape} in {time.time() - t1:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nSmoke test accuracy: {acc:.3f}  (not a result -- just confirms the pipeline runs)")
    print(classification_report(y_test, preds, target_names=CLASSES))
    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
