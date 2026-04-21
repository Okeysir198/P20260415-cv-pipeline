"""Re-export RT-DETRv2 / D-FINE to ONNX after fixing the `hf_model.` prefix bug.

Our in-repo HF backend saved `pytorch_model.bin` via `HFModelWrapper.state_dict()`
which keyed every tensor `hf_model.model.*`. Optimum's `main_export` calls
`AutoModelForObjectDetection.from_pretrained(run_dir)` internally, which can't
match those keys → the exported ONNX contained a random-weight model.

Fix: rewrite `pytorch_model.bin` with the prefix stripped, then re-export.
After this script, re-run `scripts/quantize_detr_int8_static.py` to rebuild
the INT8 QDQ variants on top of the correct fp32 ONNX.
"""

from __future__ import annotations

from pathlib import Path

import torch
from optimum.exporters.onnx import main_export

REPO = Path(__file__).resolve().parent.parent

RUNS = [
    ("rtdetr_v2", REPO / "notebooks/detr_finetune_reference/our_rtdetr_v2_torchvision/runs/seed42"),
    ("dfine_50ep", REPO / "notebooks/detr_finetune_reference/our_dfine_torchvision/runs/seed42_50ep"),
]


def fix_and_export(name: str, run_dir: Path) -> None:
    print(f"\n=== {name} ({run_dir}) ===")
    orig = run_dir / "pytorch_model.bin.orig"
    fixed = run_dir / "pytorch_model.bin"

    if orig.exists() and not fixed.exists():
        print("Rewriting pytorch_model.bin with `hf_model.` prefix stripped ...")
        sd = torch.load(orig, map_location="cpu", weights_only=True)
        n_stripped = sum(1 for k in sd if k.startswith("hf_model."))
        sd = {k.removeprefix("hf_model."): v for k, v in sd.items()}
        torch.save(sd, fixed)
        print(f"  stripped prefix from {n_stripped}/{len(sd)} keys → {fixed.name}")
    else:
        print(f"pytorch_model.bin already present — skipping rewrite")

    print("Re-exporting ONNX via Optimum ...")
    out_dir = run_dir / "_reexport"
    out_dir.mkdir(exist_ok=True)
    main_export(
        model_name_or_path=str(run_dir),
        output=out_dir,
        task="object-detection",
        opset=17,
        device="cpu",
    )
    # Move model.onnx back up to run_dir root
    src = out_dir / "model.onnx"
    dst = run_dir / "model.onnx"
    src.rename(dst)
    # Keep supplementary files (config, preprocessor) that Optimum copied
    for f in out_dir.iterdir():
        if f.name in {"ort_config.json"}:
            f.unlink()  # tiny, quantizer regenerates
        elif f.name in {"config.json", "preprocessor_config.json"}:
            f.unlink()  # already in run_dir
    out_dir.rmdir()
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  wrote {dst} ({size_mb:.1f} MB)")


def main() -> None:
    for name, run_dir in RUNS:
        if not run_dir.exists():
            print(f"SKIP {name}: {run_dir} not found")
            continue
        try:
            fix_and_export(name, run_dir)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
