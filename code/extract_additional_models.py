"""Follow-up pass: add Phikon-v2 and Virchow2 features to the per-slide .pt
files already produced by tile_and_extract.py. Reuses each slide's saved
tile_coords (+native_mpp, to recompute the exact same level0_px_per_tile) so
tissue detection/tile-grid generation never reruns -- same tiles as
baseline_cnn/uni2h, just two more models through them, keeping the
cross-model comparison apples-to-apples.

CONCH v1.5 and H-optimus-1 are NOT included here: CONCH v1.5's official
loading path (MahmoodLab/TITAN, trust_remote_code=True) executes third-party
code and was blocked by the sandbox; H-optimus-1's HF access request is still
pending Bioptimus's manual review. Revisit both later if time allows.

Usage:
    python extract_additional_models.py \
        --features-dir "/Volumes/Carmel Disk/Cancer_Data_Carmel/Pathology_Project/features" \
        --slide-dir "/Volumes/Carmel Disk/Cancer_Data_Carmel/Pathology_Project/tcga_slides"
"""
from __future__ import annotations

import argparse
import logging
import ssl
import time
from pathlib import Path

import certifi
import openslide
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from transformers import AutoImageProcessor, AutoModel

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

TILE_PX = 224
TARGET_MPP = 0.5  # must match tile_and_extract.py -- used to recompute level0_px_per_tile
BATCH_SIZE = 16

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_additional_models")


def load_phikon_v2(device: str):
    processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
    model = AutoModel.from_pretrained("owkin/phikon-v2")
    model.eval().to(device)
    return model, processor


def load_virchow2(device: str):
    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                               mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    model.eval().to(device)
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform


@torch.inference_mode()
def extract_features_for_tiles(
    slide: openslide.OpenSlide,
    tile_coords: torch.Tensor,
    level0_px_per_tile: int,
    phikon_bundle: tuple,
    virchow_bundle: tuple,
    device: str,
) -> dict[str, torch.Tensor]:
    phikon_model, phikon_processor = phikon_bundle
    virchow_model, virchow_transform = virchow_bundle

    phikon_feats, virchow_feats = [], []
    batch_imgs: list[Image.Image] = []

    def flush():
        if not batch_imgs:
            return
        phikon_inputs = phikon_processor(batch_imgs, return_tensors="pt").to(device)
        phikon_out = phikon_model(**phikon_inputs)
        phikon_feats.append(phikon_out.last_hidden_state[:, 0, :].float().cpu())

        vx = torch.stack([virchow_transform(im) for im in batch_imgs]).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device in ("cuda", "mps")):
            vout = virchow_model(vx)
        class_token = vout[:, 0]
        patch_tokens = vout[:, 5:]
        embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
        virchow_feats.append(embedding.float().cpu())

    for x0, y0 in tile_coords.tolist():
        try:
            region = slide.read_region((int(x0), int(y0)), 0, (level0_px_per_tile, level0_px_per_tile)).convert("RGB")
        except Exception as exc:
            log.warning(f"  tile ({x0},{y0}) unreadable, skipping: {exc}")
            continue
        tile = region.resize((TILE_PX, TILE_PX), Image.LANCZOS)
        batch_imgs.append(tile)
        if len(batch_imgs) == BATCH_SIZE:
            flush()
            batch_imgs = []
    flush()

    return {
        "phikon_v2": torch.cat(phikon_feats, dim=0),
        "virchow2": torch.cat(virchow_feats, dim=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Phikon-v2 + Virchow2 features to existing per-slide .pt files")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--slide-dir", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    log.info(f"Device: {device}")

    log.info("Loading models...")
    phikon_bundle = load_phikon_v2(device)
    virchow_bundle = load_virchow2(device)

    pt_files = sorted(p for p in args.features_dir.glob("*.pt") if not p.name.startswith("."))
    log.info(f"{len(pt_files)} slide feature files to process")

    for i, pt_path in enumerate(pt_files):
        data = torch.load(pt_path, weights_only=False)
        if "phikon_v2" in data["features"] and "virchow2" in data["features"]:
            continue  # idempotent: already processed

        patient_id = data["patient_id"]
        slide_filename_matches = list(args.slide_dir.glob(f"{patient_id}-*.svs"))
        if not slide_filename_matches:
            log.warning(f"[{i+1}/{len(pt_files)}] {patient_id}: slide file not found, skipping")
            continue
        slide_path = slide_filename_matches[0]

        t0 = time.time()
        slide = openslide.OpenSlide(str(slide_path))
        native_mpp = data["native_mpp"]
        level0_px_per_tile = int(round(TILE_PX * TARGET_MPP / native_mpp))

        new_features = extract_features_for_tiles(
            slide, data["tile_coords"], level0_px_per_tile, phikon_bundle, virchow_bundle, device
        )
        data["features"]["phikon_v2"] = new_features["phikon_v2"]
        data["features"]["virchow2"] = new_features["virchow2"]
        torch.save(data, pt_path)

        elapsed = time.time() - t0
        n = data["n_tiles"]
        log.info(f"[{i+1}/{len(pt_files)}] {patient_id} ({data['dx_class']}): "
                  f"{n} tiles in {elapsed:.1f}s ({n/elapsed:.1f} tiles/s)")

    log.info("Done.")


if __name__ == "__main__":
    main()
