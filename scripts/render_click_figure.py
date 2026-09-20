#!/usr/bin/env python3
"""
Render a publication-quality figure showing the image, the mask of one object,
and the positive/negative clicks that generated it, from a workspace produced by
the GUI (reads images/, masks/ and clicks.json).

Standalone by design: keeps dissertation-figure styling out of the tool.
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _frame_names(workspace):
    images_dir = os.path.join(workspace, "images")
    names = [f[:-4] for f in sorted(os.listdir(images_dir))]
    return names


def _load_clicks(workspace):
    with open(os.path.join(workspace, "clicks.json")) as f:
        data = json.load(f)
    return {int(o): {int(fr): e for fr, e in frames.items()}
            for o, frames in data.items()}


def render_click_figure(workspace, obj_id, frame_idx, out_path, *,
                        mask_alpha=0.45, point_size=220.0, dpi=200,
                        show_legend=False):
    clicks = _load_clicks(workspace)
    if obj_id not in clicks or frame_idx not in clicks[obj_id]:
        raise KeyError(f"No recorded clicks for object {obj_id}, frame {frame_idx}")
    entry = clicks[obj_id][frame_idx]

    names = _frame_names(workspace)
    name = names[frame_idx]
    img_path = os.path.join(workspace, "images", name + ".jpg")
    mask_path = os.path.join(workspace, "masks", name + ".png")
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing image or mask for frame {frame_idx} ({name})")

    image = np.array(Image.open(img_path).convert("RGB"))
    mask_arr = np.array(Image.open(mask_path))
    obj_mask = mask_arr == obj_id

    h, w = image.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=dpi)
    ax.imshow(image)

    # Semitransparent green fill for the object, plus its contour.
    overlay = np.zeros((h, w, 4), dtype=float)
    overlay[obj_mask] = [0.15, 0.85, 0.25, mask_alpha]
    ax.imshow(overlay)
    ax.contour(obj_mask, levels=[0.5], colors="#0a7d1f", linewidths=1.5)

    pos = np.array(entry.get("pos", []), dtype=float).reshape(-1, 2)
    neg = np.array(entry.get("neg", []), dtype=float).reshape(-1, 2)
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], s=point_size, marker="o",
                   facecolors="#22c55e", edgecolors="white", linewidths=1.5,
                   label="Clique positivo", zorder=5)
    if len(neg):
        ax.scatter(neg[:, 0], neg[:, 1], s=point_size, marker="o",
                   facecolors="#ef4444", edgecolors="white", linewidths=1.5,
                   label="Clique negativo", zorder=5)

    if show_legend and (len(pos) or len(neg)):
        ax.legend(loc="upper right", framealpha=0.9)

    ax.set_axis_off()
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Render click-provenance figure(s).")
    p.add_argument("--workspace", required=True)
    p.add_argument("--obj", type=int, required=True)
    p.add_argument("--frame", type=int, default=None,
                   help="0-based frame index; omit to render all annotated frames.")
    p.add_argument("--out", required=True,
                   help="Output PNG path, or output directory when --frame is omitted.")
    p.add_argument("--mask-alpha", type=float, default=0.45)
    p.add_argument("--point-size", type=float, default=220.0)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--legend", action="store_true",
                   help="Show the positive/negative legend (off by default).")
    args = p.parse_args()

    kwargs = dict(mask_alpha=args.mask_alpha, point_size=args.point_size,
                  dpi=args.dpi, show_legend=args.legend)

    if args.frame is not None:
        render_click_figure(args.workspace, args.obj, args.frame, args.out, **kwargs)
        print(f"Wrote {args.out}")
        return

    clicks = _load_clicks(args.workspace)
    frames = sorted(clicks.get(args.obj, {}).keys())
    if not frames:
        raise SystemExit(f"No recorded clicks for object {args.obj}")
    os.makedirs(args.out, exist_ok=True)
    for fr in frames:
        out = os.path.join(args.out, f"obj{args.obj}_frame{fr}.png")
        render_click_figure(args.workspace, args.obj, fr, out, **kwargs)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
