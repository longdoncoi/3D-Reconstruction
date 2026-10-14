"""Generate a minimal YOLO-format dataset for the cvtrain real-deps smoke test.

The nightly CI job has to make ultralytics' actual train -> export path run end
to end, so it builds a tiny synthetic set — a few random rectangles, one class —
instead of downloading a stock dataset. Everything is written under
``--output``, in the layout ``cvtrain.preflight`` and ultralytics both expect::

    <output>/data.yaml
    <output>/images/{train,val}/*.jpg
    <output>/labels/{train,val}/*.txt

Usage (see .github/workflows/cvtrain-nightly.yml)::

    python .github/scripts/make_toy_dataset.py --output "$GITHUB_WORKSPACE/Dataset"
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMGSZ = 64
IMAGES_PER_SPLIT = 3
SEED = 20261011


def _write_sample(image_dir: Path, label_dir: Path, name: str, rng: random.Random) -> None:
    """One synthetic image plus its YOLO label (class ``0`` quad polygon).

    The label is written in segmentation form (class + four corner points,
    9 columns): ultralytics treats any row with more than 6 columns as a
    polygon and derives the detection bounding box from it via
    ``segments2boxes``, so one label file serves both ``--det`` and ``--seg``
    training (a plain 5-column box label crashes the seg transforms with an
    empty mask tensor).
    """

    image = Image.new("RGB", (IMGSZ, IMGSZ), (rng.randrange(256), rng.randrange(256), rng.randrange(256)))
    draw = ImageDraw.Draw(image)
    x0 = rng.randint(4, IMGSZ // 2)
    y0 = rng.randint(4, IMGSZ // 2)
    x1 = rng.randint(IMGSZ // 2, IMGSZ - 4)
    y1 = rng.randint(IMGSZ // 2, IMGSZ - 4)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
    image.save(image_dir / f"{name}.jpg")
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    polygon = " ".join(f"{cx / IMGSZ:.6f} {cy / IMGSZ:.6f}" for cx, cy in corners)
    (label_dir / f"{name}.txt").write_text(f"0 {polygon}\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="dataset root (e.g. $GITHUB_WORKSPACE/Dataset); required so a local "
        "run can never clobber a real Dataset/ by accident",
    )
    args = parser.parse_args()

    rng = random.Random(SEED)
    for split in ("train", "val"):
        images = args.output / "images" / split
        labels = args.output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for index in range(IMAGES_PER_SPLIT):
            _write_sample(images, labels, f"img{index:03d}", rng)

    (args.output / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnames:\n  0: part\n",
        encoding="utf-8",
    )
    print(f"toy dataset written under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())