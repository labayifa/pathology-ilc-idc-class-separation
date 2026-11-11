"""Plan step 5: combined tiling + feature extraction, one streaming pass per
slide. Tissue detection + tile-grid generation is model-independent (done
once); both the baseline CNN and UNI2-h then run on the exact same tile
crops, so the comparison between them is never confounded by different
tiling. Validated interactively in
notebooks/01_slide_exploration_and_preprocessing.ipynb before this script
existed -- same tissue-detection/tiling logic, now production-ized.

No individual tile images are saved to disk (ground rule 4: the deliverable
repo must not contain slide-derived images) -- only the resulting per-slide
feature tensors, tile coordinates, and tissue fractions.

Usage:
    python tile_and_extract.py \
        --cohort ../cohort/cohort.csv \
        --slide-dir "/Volumes/Carmel Disk/Cancer_Data_Carmel/Pathology_Project/tcga_slides" \
        --out-dir "/Volumes/Carmel Disk/Cancer_Data_Carmel/Pathology_Project/features"
"""
from __future__ import annotations

import argparse
import csv
import logging
import ssl
import time
from pathlib import Path

import certifi
import numpy as np
import openslide
import timm
import torch
import torchvision
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision.models import ResNet18_Weights

# This Python install can't find the system CA bundle by default (same issue
# hit earlier with aiohttp/urllib for GDC/cBioPortal) -- point the *default*
# HTTPS context at certifi's bundle so torch.hub's internal downloads work too.
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

TARGET_MPP = 0.5  # see plan: smallest target that only requires downsampling for this cohort
TILE_PX = 224
MAX_TILES = 1000
MIN_TISSUE_FRACTION = 0.5
BATCH_SIZE = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tile_and_extract")


def otsu_threshold(gray: np.ndarray) -> int:
    """Same implementation validated in the exploration notebook."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    total = hist.sum()
    sum_all = np.dot(np.arange(256), hist)
    sum_bg, weight_bg, best_thresh, best_var = 0.0, 0.0, 0, 0.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = t
    return best_thresh


def load_uni2h(device: str):
    timm_kwargs = dict(
        img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
        embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
        mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8,
        dynamic_img_size=True,
    )
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


def load_baseline_cnn(device: str):
    model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()  # drop classifier head -> 512-dim penultimate features
    model.eval().to(device)
    weights_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
    return model, weights_transforms


def tile_grid_for_slide(slide: openslide.OpenSlide, native_mpp: float) -> list[tuple[int, int, float]]:
    """Returns up to MAX_TILES (x0, y0, tissue_fraction) in level-0 coordinates,
    highest tissue fraction first. Same logic as the exploration notebook."""
    thumbnail = slide.get_thumbnail((1024, 1024))
    gray = np.array(thumbnail.convert("L"))
    thresh = otsu_threshold(gray)
    tissue_mask = gray < thresh

    level0_w, level0_h = slide.level_dimensions[0]
    thumb_w, thumb_h = thumbnail.size
    scale_x, scale_y = level0_w / thumb_w, level0_h / thumb_h

    level0_px_per_tile = int(round(TILE_PX * TARGET_MPP / native_mpp))
    thumb_tile_w = max(1, int(level0_px_per_tile / scale_x))
    thumb_tile_h = max(1, int(level0_px_per_tile / scale_y))

    candidates = []
    for ty in range(0, thumb_h - thumb_tile_h, max(1, thumb_tile_h)):
        for tx in range(0, thumb_w - thumb_tile_w, max(1, thumb_tile_w)):
            patch = tissue_mask[ty:ty + thumb_tile_h, tx:tx + thumb_tile_w]
            frac = float(patch.mean()) if patch.size else 0.0
            if frac >= MIN_TISSUE_FRACTION:
                candidates.append((int(tx * scale_x), int(ty * scale_y), frac))

    candidates.sort(key=lambda c: -c[2])
    return candidates[:MAX_TILES], level0_px_per_tile


@torch.inference_mode()
def extract_slide_features(
    slide: openslide.OpenSlide,
    tiles: list[tuple[int, int, float]],
    level0_px_per_tile: int,
    models: dict,
    device: str,
) -> tuple[dict[str, torch.Tensor], list[tuple[int, int, float]]]:
    all_feats = {name: [] for name in models}
    batch_imgs: list[Image.Image] = []

    def flush():
        if not batch_imgs:
            return
        for name, (model, transform) in models.items():
            x = torch.stack([transform(im) for im in batch_imgs]).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device in ("cuda", "mps")):
                out = model(x)
            all_feats[name].append(out.float().cpu())

    used_coords = []
    for x0, y0, _frac in tiles:
        try:
            region = slide.read_region((x0, y0), 0, (level0_px_per_tile, level0_px_per_tile)).convert("RGB")
        except Exception as exc:
            # Some candidate positions (usually near tissue-mask edges, derived
            # from a coarse thumbnail) land on a tile this specific file can't
            # deliver -- skip that one tile, don't lose the whole slide over it.
            log.warning(f"  tile ({x0},{y0}) unreadable, skipping: {exc}")
            continue
        tile = region.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        batch_imgs.append(tile)
        used_coords.append((x0, y0, _frac))
        if len(batch_imgs) == BATCH_SIZE:
            flush()
            batch_imgs = []
    flush()

    return {name: torch.cat(feats, dim=0) for name, feats in all_feats.items()}, used_coords


def main() -> None:
    parser = argparse.ArgumentParser(description="Tile + extract frozen features for the TCGA-BRCA cohort")
    parser.add_argument("--cohort", type=Path, default=Path("../cohort/cohort.csv"))
    parser.add_argument("--slide-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Loading models...")
    models = {
        "baseline_cnn": load_baseline_cnn(device),
        "uni2h": load_uni2h(device),
    }

    with open(args.cohort) as f:
        rows = list(csv.DictReader(f))
    log.info(f"{len(rows)} slides in cohort")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    skipped_no_mpp = []
    skipped_open_failed = []

    for i, row in enumerate(rows):
        out_path = args.out_dir / f"{row['patient_id']}.pt"
        if out_path.exists():
            continue  # idempotent: already processed

        slide_path = args.slide_dir / row["file_name"]
        t0 = time.time()
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as exc:
            log.warning(f"[{i+1}/{len(rows)}] {row['patient_id']}: failed to open ({exc}), skipping")
            skipped_open_failed.append(row["patient_id"])
            continue

        native_mpp_str = slide.properties.get("openslide.mpp-x")
        if native_mpp_str is None:
            log.warning(f"[{i+1}/{len(rows)}] {row['patient_id']}: no MPP metadata, skipping (not guessing)")
            skipped_no_mpp.append(row["patient_id"])
            continue
        native_mpp = float(native_mpp_str)

        tiles, level0_px_per_tile = tile_grid_for_slide(slide, native_mpp)
        if not tiles:
            log.warning(f"[{i+1}/{len(rows)}] {row['patient_id']}: no tiles passed tissue threshold, skipping")
            continue

        features, used_tiles = extract_slide_features(slide, tiles, level0_px_per_tile, models, device)
        if len(used_tiles) < len(tiles):
            log.warning(
                f"  {row['patient_id']}: {len(tiles) - len(used_tiles)}/{len(tiles)} "
                f"candidate tiles were unreadable and skipped"
            )

        torch.save(
            {
                "patient_id": row["patient_id"],
                "dx_class": row["dx_class"],
                "tss_site": row["tss_site"],
                "native_mpp": native_mpp,
                "n_tiles": len(used_tiles),
                "tile_coords": torch.tensor([[x, y] for x, y, _ in used_tiles], dtype=torch.float32),
                "tissue_fraction": torch.tensor([f for _, _, f in used_tiles], dtype=torch.float32),
                "features": features,
            },
            out_path,
        )
        elapsed = time.time() - t0
        log.info(
            f"[{i+1}/{len(rows)}] {row['patient_id']} ({row['dx_class']}): "
            f"{len(used_tiles)} tiles in {elapsed:.1f}s ({len(used_tiles)/elapsed:.1f} tiles/s)"
        )

    log.info(f"Done. Skipped (no MPP): {len(skipped_no_mpp)} {skipped_no_mpp}")
    log.info(f"Skipped (open failed): {len(skipped_open_failed)} {skipped_open_failed}")


if __name__ == "__main__":
    main()
