"""Robustness sweep — task-metric degradation under standard corruptions.

Applies four families of input-space corruptions at three severities each to
the val loader and reports the primary task metric at every step. Mirrors the
sibling analyzers (``label_quality.py``, ``distribution_mismatch.py``,
``learning_ability.py``) in entry-point style and output layout.

Corruption families (three severities each):

* ``gaussian_blur``  — ``torchvision.transforms.v2.GaussianBlur`` with
  sigma ∈ {1, 2, 4} px.
* ``jpeg``           — PIL ``Image.save(..., quality=q)`` round-trip at
  quality ∈ {50, 30, 15}.
* ``brightness``     — torchvision ``ColorJitter``-style multiplicative
  brightness at factor ∈ {0.2, 0.4, 0.6} (applied as ``1 ± factor``, random
  sign per image).
* ``rotation``       — ±{5°, 10°, 20°}. **Classification only.** Detection,
  segmentation, and keypoint are skipped because rotating inputs without
  also rotating labels (with proper edge/fill semantics) biases the metric
  in a way that is not a real robustness signal. Skipped tasks appear in
  the JSON with ``skipped_reason``.

Outputs (under the ``error_analysis/`` dir the caller passes in):

* ``14_robustness_sweep.png``  — single multi-line plot: primary metric
  (y-axis) vs severity step 0..3 (x-axis), one line per family. Step 0 is
  the clean baseline (identity transform).
* ``14_robustness_sweep.json`` — ``{family: {severity: metric}}`` plus
  per-family AUC (trapezoidal integral over 0..3) and worst-case severity.

Metric reuse: this module does **not** reimplement metric code. The caller
passes ``metric_fn(model, loader) -> float`` — typically a thin closure
around ``core.p08_evaluation.evaluate`` or ``error_analysis_runner``'s
existing dispatch. That keeps this module task-agnostic and avoids
drifting from the authoritative metric in the rest of p08.
"""
from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from utils.viz import apply_plot_style

from loguru import logger

matplotlib.use("Agg")


ROBUSTNESS_FILENAMES: dict[str, str] = {
    "sweep": "14_robustness_sweep.png",
    "json":  "14_robustness_sweep.json",
}

# Severities are indexed 1..3; step 0 is the clean baseline.
_SEVERITIES: dict[str, list[float]] = {
    "gaussian_blur": [1.0, 2.0, 4.0],   # sigma px
    "jpeg":          [50, 30, 15],      # quality (lower = worse)
    "brightness":    [0.2, 0.4, 0.6],   # ± factor
    "rotation":      [5.0, 10.0, 20.0], # ± degrees
}

_ROTATION_SKIPPABLE_TASKS = {"detection", "segmentation", "keypoint"}


def run(
    *,
    model: torch.nn.Module,
    val_loader: Iterable,
    task: str,
    metric_fn: Callable[[torch.nn.Module, Iterable], float],
    primary_metric_name: str,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Run the robustness sweep and write chart + JSON.

    Args:
        model: trained model in eval mode.
        val_loader: the val DataLoader; wrapped per (family, severity).
        task: canonical task string (``"classification"`` / ``"detection"`` /
            ``"segmentation"`` / ``"keypoint"``).
        metric_fn: ``(model, loader) -> float`` returning the primary metric
            (higher-is-better). Reuse the dispatch from
            ``core.p08_evaluation.evaluate`` / ``error_analysis_runner``.
        primary_metric_name: human label for the y-axis (e.g. ``"mAP@0.5"``).
        output_dir: error-analysis dir; artefacts land directly inside it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_low = task.lower()
    clean = float(metric_fn(model, val_loader))
    logger.info("robustness_sweep: clean %s = %.4f", primary_metric_name, clean)

    results: dict[str, dict[str, Any]] = {}
    for family, severities in _SEVERITIES.items():
        skip = (family == "rotation" and task_low in _ROTATION_SKIPPABLE_TASKS)
        if skip:
            results[family] = {
                "skipped_reason":
                    f"rotation not applied to {task_low}: rotating inputs "
                    "without matching label rotation (with correct edge/fill "
                    "semantics) is not a faithful robustness signal.",
                "severities": {str(s): None for s in severities},
            }
            continue

        per_severity: dict[str, float] = {"0": clean}
        for sev_val in severities:
            tfm = _build_transform(family, sev_val)
            wrapped = _CorruptedLoader(val_loader, tfm)
            m = float(metric_fn(model, wrapped))
            per_severity[str(sev_val)] = m
            logger.info(
                "robustness_sweep: %s @ %s → %.4f", family, sev_val, m,
            )

        values = [per_severity["0"], *[per_severity[str(s)] for s in severities]]
        # Trapezoidal AUC over steps 0..3 (x spacing = 1).
        auc = float(np.trapezoid(values, dx=1.0))
        worst_idx = int(np.argmin(values))
        worst_step = worst_idx  # 0 = clean, 1..3 = severity index
        results[family] = {
            "severities": per_severity,
            "auc": round(auc, 6),
            "worst_case_step": worst_step,
            "worst_case_value": round(float(values[worst_idx]), 6),
        }

    sweep_png = _plot_sweep(
        clean=clean,
        results=results,
        primary_metric_name=primary_metric_name,
        out_path=output_dir / ROBUSTNESS_FILENAMES["sweep"],
    )

    payload = {
        "task": task_low,
        "primary_metric": primary_metric_name,
        "clean": round(clean, 6),
        "families": results,
    }
    json_path = output_dir / ROBUSTNESS_FILENAMES["json"]
    json_path.write_text(json.dumps(payload, indent=2))

    return {
        "artifacts": {"sweep": sweep_png, "json": json_path},
        "payload": payload,
    }


# ---------------------------------------------------------------------------
# Corruption transforms (applied to the input image tensor only)
# ---------------------------------------------------------------------------


def _build_transform(family: str, severity: float) -> Callable[[torch.Tensor], torch.Tensor]:
    if family == "gaussian_blur":
        # kernel size ~ ceil(6*sigma) | odd, min 3
        k = int(max(3, int(np.ceil(6 * severity)) | 1))
        blur = v2.GaussianBlur(kernel_size=k, sigma=float(severity))
        return lambda x: blur(x)
    if family == "jpeg":
        q = int(severity)
        return lambda x: _jpeg_roundtrip(x, quality=q)
    if family == "brightness":
        # Match torchvision ColorJitter semantics: brightness factor in
        # [1 - f, 1 + f]. Sample a per-call scalar so batches vary.
        f = float(severity)
        def _apply(x: torch.Tensor) -> torch.Tensor:
            sign = 1.0 if torch.rand(()) < 0.5 else -1.0
            factor = 1.0 + sign * f
            return torch.clamp(x * factor, 0.0, 1.0) if x.is_floating_point() \
                else torch.clamp(x.float() * factor, 0.0, 255.0).to(x.dtype)
        return _apply
    if family == "rotation":
        deg = float(severity)
        rot = v2.RandomRotation(degrees=(-deg, deg))
        return lambda x: rot(x)
    raise ValueError(f"unknown corruption family: {family!r}")


def _jpeg_roundtrip(x: torch.Tensor, *, quality: int) -> torch.Tensor:
    """JPEG encode/decode each image in a batch via PIL. Preserves dtype."""
    if x.ndim == 3:
        return _jpeg_one(x, quality=quality)
    out = torch.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = _jpeg_one(x[i], quality=quality)
    return out


def _jpeg_one(img: torch.Tensor, *, quality: int) -> torch.Tensor:
    was_float = img.is_floating_point()
    arr = img.detach().cpu()
    if was_float:
        arr = (arr.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    else:
        arr = arr.to(torch.uint8)
    # CHW → HWC for PIL
    hwc = arr.permute(1, 2, 0).numpy()
    if hwc.shape[2] == 1:
        pil = Image.fromarray(hwc[:, :, 0], mode="L")
    else:
        pil = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    decoded = np.asarray(Image.open(buf).convert(pil.mode))
    if decoded.ndim == 2:
        decoded = decoded[:, :, None]
    t = torch.from_numpy(decoded).permute(2, 0, 1).to(img.device)
    if was_float:
        t = t.float() / 255.0
    return t.to(img.dtype)


# ---------------------------------------------------------------------------
# Loader wrapper — applies the transform to input tensors only
# ---------------------------------------------------------------------------


class _CorruptedLoader:
    """Iterable proxy that applies a per-image transform to batch inputs.

    Supports the two batch shapes used across p08 metric callers:
    * tuple/list ``(images, targets, ...)`` — transforms index 0.
    * dict ``{"pixel_values": ..., ...}`` — transforms that key.
    """

    def __init__(self, base: Iterable, transform: Callable[[torch.Tensor], torch.Tensor]):
        self._base = base
        self._transform = transform

    def __iter__(self):
        for batch in self._base:
            yield self._apply(batch)

    def __len__(self) -> int:  # pragma: no cover
        return len(self._base)  # type: ignore[arg-type]

    def _apply(self, batch):
        if isinstance(batch, dict):
            for k in ("pixel_values", "images", "image"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch = {**batch, k: self._transform(batch[k])}
                    break
            return batch
        if isinstance(batch, (tuple, list)) and len(batch) > 0 and torch.is_tensor(batch[0]):
            new0 = self._transform(batch[0])
            return type(batch)((new0, *batch[1:])) if isinstance(batch, tuple) \
                else [new0, *batch[1:]]
        if torch.is_tensor(batch):
            return self._transform(batch)
        return batch


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


def _plot_sweep(
    *,
    clean: float,
    results: dict[str, dict[str, Any]],
    primary_metric_name: str,
    out_path: Path,
) -> Path:
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    x_steps = [0, 1, 2, 3]
    color_cycle = {
        "gaussian_blur": "#1f77b4",
        "jpeg":          "#ff7f0e",
        "brightness":    "#2ca02c",
        "rotation":      "#d62728",
    }

    for family, data in results.items():
        if "skipped_reason" in data:
            continue
        sev = _SEVERITIES[family]
        ys = [clean] + [data["severities"][str(s)] for s in sev]
        ax.plot(
            x_steps, ys,
            marker="o", linewidth=1.8, markersize=5,
            color=color_cycle.get(family, None),
            label=f"{family}  (AUC={data['auc']:.2f})",
        )

    ax.set_xticks(x_steps)
    ax.set_xticklabels(["clean", "s1", "s2", "s3"])
    ax.set_xlabel("severity step")
    ax.set_ylabel(primary_metric_name)
    ax.set_title(
        f"Robustness sweep — {primary_metric_name} vs corruption severity",
    )
    ax.set_ylim(0, max(1.0, clean * 1.15))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Standalone smoke — verifies import + end-to-end on tiny tensors.
# ---------------------------------------------------------------------------


def _smoke() -> None:
    import tempfile

    torch.manual_seed(0)

    class _Identity(torch.nn.Module):
        def forward(self, x):
            return x

    # Fake loader: 4 batches of (B=2, 3, 32, 32) float images in [0,1],
    # paired with a dummy target tensor.
    def _loader():
        for _ in range(4):
            yield (torch.rand(2, 3, 32, 32), torch.zeros(2, dtype=torch.long))

    def _metric(model, loader):
        # Toy "metric": 1 - mean absolute deviation from the clean mean.
        # Degrades monotonically with corruption.
        model.eval()
        with torch.no_grad():
            diffs = []
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                diffs.append((x - 0.5).abs().mean().item())
            return float(max(0.0, 1.0 - np.mean(diffs)))

    with tempfile.TemporaryDirectory() as td:
        out = run(
            model=_Identity(),
            val_loader=list(_loader()),
            task="classification",
            metric_fn=_metric,
            primary_metric_name="toy_score",
            output_dir=Path(td),
        )
        png = out["artifacts"]["sweep"]
        js = out["artifacts"]["json"]
        assert png.exists() and png.stat().st_size > 0, "sweep png missing"
        assert js.exists(), "json missing"
        payload = json.loads(js.read_text())
        assert set(payload["families"]) == set(_SEVERITIES), "missing family"
        # Rotation is run for classification (not skipped).
        assert "skipped_reason" not in payload["families"]["rotation"]
        print("smoke OK:", payload["clean"], {k: v.get("auc") for k, v in payload["families"].items()})

    # Also verify the detection skip path.
    with tempfile.TemporaryDirectory() as td:
        out = run(
            model=_Identity(),
            val_loader=list(_loader()),
            task="detection",
            metric_fn=_metric,
            primary_metric_name="mAP@0.5",
            output_dir=Path(td),
        )
        payload = json.loads(out["artifacts"]["json"].read_text())
        assert payload["families"]["rotation"].get("skipped_reason"), \
            "rotation should be skipped for detection"
        print("smoke OK (detection skip):", payload["families"]["rotation"]["skipped_reason"][:60])


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    _smoke()
