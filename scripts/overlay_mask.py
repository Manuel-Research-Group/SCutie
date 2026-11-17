#!/usr/bin/env python3
"""
Overlay a frame's segmentation mask on top of the corresponding image from a
workspace produced by the GUI (reads images/ and masks/) and write the result.

Example:
    python scripts/overlay_mask.py workspaces/iph_solo_inferior_ds15_v3 81 \
        -o overlay_0081.png
    # all frames that have a mask, into workspaces/.../visualization/
    python scripts/overlay_mask.py workspaces/iph_solo_inferior_ds15_v3 --all
"""
import argparse
import os

import numpy as np
from PIL import Image

from cutie.utils.palette import davis_palette


def _color_map():
    cm = np.frombuffer(davis_palette, dtype=np.uint8).reshape(-1, 3).copy()
    # scales for better visualization (same as the GUI)
    return (cm.astype(np.float32) * 1.5).clip(0, 255).astype(np.uint8)


COLOR_MAP = _color_map()


def overlay_davis(image, mask, alpha=0.5, fade=False):
    """Overlay an indexed segmentation mask on an RGB image (from DAVIS)."""
    im_overlay = image.copy()
    colored_mask = COLOR_MAP[mask]
    foreground = image * alpha + (1 - alpha) * colored_mask
    binary_mask = mask > 0
    im_overlay[binary_mask] = foreground[binary_mask]
    if fade:
        im_overlay[~binary_mask] = im_overlay[~binary_mask] * 0.6
    return im_overlay.astype(image.dtype)


def _frame_name(name_or_idx):
    """Accept either '81', '000081' or '000081.jpg' and return the 6-digit stem."""
    stem = os.path.splitext(str(name_or_idx))[0]
    return stem.zfill(6) if stem.isdigit() else stem


def render_overlay(workspace, frame, out_path, alpha=0.5, fade=False):
    stem = _frame_name(frame)
    img_path = os.path.join(workspace, "images", stem + ".jpg")
    mask_path = os.path.join(workspace, "masks", stem + ".png")
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path))  # indexed / palette PNG
    if mask.shape[:2] != image.shape[:2]:
        mask = np.array(Image.fromarray(mask).resize(
            (image.shape[1], image.shape[0]), Image.NEAREST))

    result = overlay_davis(image, mask, alpha=alpha, fade=fade)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    Image.fromarray(result).save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("workspace", help="path to the workspace directory")
    ap.add_argument("frame", nargs="?", help="frame number, e.g. 81 or 000081")
    ap.add_argument("-o", "--out", help="output image path (single-frame mode)")
    ap.add_argument("--all", action="store_true",
                    help="overlay every frame that has a mask into <workspace>/visualization/")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="image weight in the blend (higher = fainter mask)")
    ap.add_argument("--fade", action="store_true", help="darken the background")
    args = ap.parse_args()

    if args.all:
        masks_dir = os.path.join(args.workspace, "masks")
        out_dir = os.path.join(args.workspace, "visualization")
        stems = [os.path.splitext(f)[0] for f in sorted(os.listdir(masks_dir))
                 if f.endswith(".png")]
        for stem in stems:
            out = os.path.join(out_dir, stem + ".png")
            render_overlay(args.workspace, stem, out, alpha=args.alpha, fade=args.fade)
        print(f"Wrote {len(stems)} overlays to {out_dir}")
        return

    if args.frame is None:
        ap.error("provide a frame number, or use --all")
    out = args.out or f"overlay_{_frame_name(args.frame)}.png"
    print("Wrote", render_overlay(args.workspace, args.frame, out,
                                  alpha=args.alpha, fade=args.fade))


if __name__ == "__main__":
    main()
