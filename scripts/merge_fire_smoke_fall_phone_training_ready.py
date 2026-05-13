"""Combine fire_detection + fall_detection + phone_usage training_ready/ into fire_smoke_fall_phone/.

Class remap:
  fire_detection:  0 fire           -> 0
  fire_detection:  1 smoke          -> 1
  fall_detection:  0 person         -> dropped
  fall_detection:  1 fallen_person  -> 2
  phone_usage:     0 phone_usage    -> 3
Fall images with no fallen_person labels after remap are skipped.
Images are symlinked (no copy); labels are rewritten.
Filenames are prefixed with `fire__` / `fall__` / `phone__` to avoid collisions.
Idempotent: re-running overwrites symlinks/labels in place.
"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_store" / "training_ready"

SOURCES = {
    "fire": {
        "src": DATASET_ROOT / "fire_detection",
        "remap": {0: 0, 1: 1},  # fire->0, smoke->1
        "drop": set(),
        "require_label": False,
    },
    "fall": {
        "src": DATASET_ROOT / "fall_detection",
        "remap": {1: 2},  # fallen_person -> 2
        "drop": {0},  # person dropped
        "require_label": True,
    },
    "phone": {
        "src": DATASET_ROOT / "safety_poketenashi_phone_usage",
        "remap": {0: 3},  # phone_usage -> 3
        "drop": set(),
        "require_label": True,
    },
}

SPLITS = ("train", "val", "test")


def remap_label_file(src_label: Path, remap: dict[int, int], drop: set[int]) -> list[str]:
    out: list[str] = []
    if not src_label.exists():
        return out
    for line in src_label.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls = int(parts[0])
        if cls in drop:
            continue
        if cls not in remap:
            continue
        out.append(" ".join([str(remap[cls]), *parts[1:]]))
    return out


def merge(out_root: Path) -> None:
    for split in SPLITS:
        img_out = out_root / split / "images"
        lbl_out = out_root / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for tag, cfg in SOURCES.items():
            src_img_dir = cfg["src"] / split / "images"
            src_lbl_dir = cfg["src"] / split / "labels"
            if not src_img_dir.exists():
                print(f"  [{tag}/{split}] SKIP -- missing {src_img_dir}")
                continue

            kept = 0
            dropped_no_label = 0
            dropped_empty_after_remap = 0
            for img_path in sorted(src_img_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                src_label = src_lbl_dir / (img_path.stem + ".txt")
                remapped = remap_label_file(src_label, cfg["remap"], cfg["drop"])
                if cfg["require_label"] and not remapped:
                    if src_label.exists():
                        dropped_empty_after_remap += 1
                    else:
                        dropped_no_label += 1
                    continue

                new_stem = f"{tag}__{img_path.stem}"
                new_img = img_out / (new_stem + img_path.suffix)
                new_lbl = lbl_out / (new_stem + ".txt")

                if new_img.is_symlink() or new_img.exists():
                    new_img.unlink()
                new_img.symlink_to(img_path.resolve())
                new_lbl.write_text("\n".join(remapped) + ("\n" if remapped else ""))
                kept += 1

            print(
                f"  [{tag}/{split}] kept={kept} "
                f"dropped_no_label={dropped_no_label} "
                f"dropped_empty_after_remap={dropped_empty_after_remap}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=DATASET_ROOT / "fire_smoke_fall_phone",
        help="Destination training_ready/ subdir",
    )
    args = parser.parse_args()
    print(f"Merging into {args.out}")
    merge(args.out)
    print("Done.")


if __name__ == "__main__":
    main()
