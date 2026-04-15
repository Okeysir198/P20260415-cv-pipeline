"""
Split assignment + filesystem-backed split layout helpers.

Split membership is where the file lives — `<feature>/train/`, `<feature>/val/`,
`<feature>/test/`. `splits.json` is only an audit snapshot (ratios + seed +
counts + timestamp), never load-bearing. These helpers keep the filesystem
and the audit snapshot consistent.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SPLIT_NAMES = ("train", "val", "test")
DROPPED_DIR = "dropped"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class SplitGenerator:
    """
    Generate train/val/test splits using stratified sampling by default.

    Writes an audit snapshot at `<output_dir>/splits.json` (ratios + seed +
    counts + timestamp). Does NOT write filename lists — split membership is
    recorded on the filesystem by the caller (see `run.py`).
    """

    def __init__(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        stratified: bool = True,
    ):
        total = sum(ratios)
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        self.ratios = ratios
        self.seed = seed
        self.stratified = stratified
        self.rng = np.random.default_rng(seed)

    def assign_splits(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Return {'train': [...], 'val': [...], 'test': [...]} assignment.

        Does NOT write anything to disk. Use this before copying files so you
        know which split subdir each sample goes into.
        """
        if not samples:
            raise ValueError("Cannot split empty sample list")
        if self.stratified:
            return self._stratified_split(samples)
        return self._random_split(samples)

    # --- back-compat: callers that used generate_splits() still work, but it
    # now writes an audit snapshot instead of filename lists.
    def generate_splits(
        self,
        samples: List[Dict],
        output_file: Path,
        task_type: str = "detection",
    ) -> Dict[str, List[str]]:
        """Compute assignment, write audit snapshot, return {split: [filenames]}.

        The written JSON is an audit snapshot only (no filename lists).
        The returned dict is a convenience for callers that still want the
        filename-per-split mapping in memory.
        """
        assignment = self.assign_splits(samples)
        counts = {k: len(v) for k, v in assignment.items()}
        write_audit_snapshot(
            Path(output_file),
            counts=counts,
            ratios=self.ratios,
            seed=self.seed,
            task_type=task_type,
            stratified=self.stratified,
            total=sum(counts.values()),
        )
        return {split: [s["filename"] for s in items] for split, items in assignment.items()}

    def _random_split(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        n = len(samples)
        indices = self.rng.permutation(n)
        if n >= 3:
            n_val = max(1, round(n * self.ratios[1]))
            n_test = max(1, round(n * self.ratios[2]))
            n_train = n - n_val - n_test
        else:
            n_train = n
            n_val = 0
            n_test = 0
        return {
            "train": [samples[i] for i in indices[:n_train]],
            "val": [samples[i] for i in indices[n_train:n_train + n_val]],
            "test": [samples[i] for i in indices[n_train + n_val:]],
        }

    def _stratified_split(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        by_class: Dict[str, List[Dict]] = {}
        for sample in samples:
            if sample.get("labels") and len(sample["labels"]) > 0:
                primary = sample["labels"][0]
            else:
                primary = "__unknown__"
            by_class.setdefault(primary, []).append(sample)

        splits = {"train": [], "val": [], "test": []}
        for class_samples in by_class.values():
            class_splits = self._random_split(class_samples)
            for k in splits:
                splits[k].extend(class_splits[k])
        for split_name in splits:
            self.rng.shuffle(splits[split_name])
        return splits

    def load_existing_splits(self, splits_file: Path) -> Dict[str, List[str]]:
        """Back-compat shim. Prefer `rescan_splits()` which reads the filesystem."""
        return rescan_splits(Path(splits_file).parent)


# --- Filesystem helpers -----------------------------------------------------

def ensure_split_dirs(feature_root: Path, include_dropped: bool = False) -> None:
    """Create train/val/test (+ optional dropped) images/ and labels/ subdirs."""
    feature_root = Path(feature_root)
    for split in SPLIT_NAMES:
        (feature_root / split / "images").mkdir(parents=True, exist_ok=True)
        (feature_root / split / "labels").mkdir(parents=True, exist_ok=True)
    if include_dropped:
        (feature_root / DROPPED_DIR / "images").mkdir(parents=True, exist_ok=True)
        (feature_root / DROPPED_DIR / "labels").mkdir(parents=True, exist_ok=True)


def rescan_splits(feature_root: Path) -> Dict[str, List[str]]:
    """Walk `{train,val,test}/images/` and return {split: [filename]}.

    Source of truth for split membership. Call after any mv/drop operation.
    """
    feature_root = Path(feature_root)
    result: Dict[str, List[str]] = {s: [] for s in SPLIT_NAMES}
    for split in SPLIT_NAMES:
        img_dir = feature_root / split / "images"
        if not img_dir.is_dir():
            continue
        for ext in IMAGE_EXTS:
            result[split].extend(sorted(p.name for p in img_dir.glob(f"*{ext}")))
    return result


def find_image_file(feature_root: Path, split: str, stem: str) -> Optional[Path]:
    """Locate the image file for a given stem within a split. Returns None if absent."""
    img_dir = Path(feature_root) / split / "images"
    for ext in IMAGE_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def move_sample(
    feature_root: Path,
    stem: str,
    from_split: str,
    to_split: str,
    hard_drop: bool = False,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Physically move an image + its label between split subdirs.

    - `to_split='drop'` moves into `dropped/` (or deletes when `hard_drop=True`).
    - No-op when `from_split == to_split`.
    - Idempotent: missing files are silently skipped.

    Returns (new_image_path, new_label_path) — both `None` when hard-dropped.
    """
    feature_root = Path(feature_root)
    if from_split == to_split:
        img = find_image_file(feature_root, from_split, stem)
        lbl = feature_root / from_split / "labels" / f"{stem}.txt"
        return img, (lbl if lbl.exists() else None)

    src_img = find_image_file(feature_root, from_split, stem)
    src_lbl = feature_root / from_split / "labels" / f"{stem}.txt"

    if to_split == "drop":
        if hard_drop:
            if src_img and src_img.exists():
                src_img.unlink()
            if src_lbl.exists():
                src_lbl.unlink()
            return None, None
        dst_img_dir = feature_root / DROPPED_DIR / "images"
        dst_lbl_dir = feature_root / DROPPED_DIR / "labels"
    else:
        if to_split not in SPLIT_NAMES:
            raise ValueError(f"Unknown split '{to_split}'. Must be one of {SPLIT_NAMES} or 'drop'.")
        dst_img_dir = feature_root / to_split / "images"
        dst_lbl_dir = feature_root / to_split / "labels"

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    new_img = dst_img_dir / src_img.name if src_img else None
    new_lbl = dst_lbl_dir / src_lbl.name if src_lbl else None

    if src_img and src_img.exists() and new_img is not None:
        _atomic_move(src_img, new_img)
    if src_lbl.exists() and new_lbl is not None:
        _atomic_move(src_lbl, new_lbl)

    return new_img, new_lbl


def _atomic_move(src: Path, dst: Path) -> None:
    """os.rename when possible (same FS), fall back to shutil.move (cross-FS)."""
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def write_audit_snapshot(
    snapshot_path: Path,
    *,
    counts: Dict[str, int],
    ratios: Tuple[float, float, float],
    seed: int,
    task_type: str = "detection",
    stratified: bool = True,
    total: Optional[int] = None,
) -> None:
    """Write `splits.json` as a tiny audit snapshot (no filename lists)."""
    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_type": task_type,
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": int(seed),
            "ratios": list(ratios),
            "stratified": bool(stratified),
            "total_samples": int(total if total is not None else sum(counts.values())),
        },
        "counts": {k: int(counts.get(k, 0)) for k in SPLIT_NAMES},
    }
    if DROPPED_DIR in counts:
        payload["counts"][DROPPED_DIR] = int(counts[DROPPED_DIR])
    with open(snapshot_path, "w") as f:
        json.dump(payload, f, indent=2)


def refresh_audit_snapshot(feature_root: Path, seed: int = 42,
                           ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                           task_type: str = "detection") -> Dict[str, int]:
    """Rescan filesystem and rewrite `<feature>/splits.json`. Returns counts dict."""
    feature_root = Path(feature_root)
    split_files = rescan_splits(feature_root)
    counts = {k: len(v) for k, v in split_files.items()}

    dropped_dir = feature_root / DROPPED_DIR / "images"
    if dropped_dir.is_dir():
        n_dropped = sum(len(list(dropped_dir.glob(f"*{ext}"))) for ext in IMAGE_EXTS)
        counts[DROPPED_DIR] = n_dropped

    write_audit_snapshot(
        feature_root / "splits.json",
        counts=counts,
        ratios=ratios,
        seed=seed,
        task_type=task_type,
        total=sum(split_files[k].__len__() for k in SPLIT_NAMES),
    )
    return counts
