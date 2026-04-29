"""Export a Paddle checkpoint to ONNX — runs in `.venv-paddle/`.

Usage:
    .venv-paddle/bin/python core/p06_paddle/export.py \\
        --config configs/_test/06_training_paddle_det.yaml \\
        --checkpoint <save_dir>/best.pdparams \\
        --out <save_dir>/model.onnx

Two-step process:
1. `tools/export_model.py` (ppdet) writes a paddle inference model
   (`<dir>/model.pdmodel` + `model.pdiparams`) — this is the "deploy" format
   ppdet uses; differs from training-time `.pdparams`.
2. `paddle2onnx` converts the inference model to ONNX.

After this, downstream phases (eval, error analysis, inference, demo) all use
the ONNX file via the standard main-venv ORT path:
    uv run core/p08_evaluation/evaluate.py --model <out.onnx> --config <05_data>.yaml
    uv run core/p10_inference/predictor.py --model <out.onnx> ...
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.p06_paddle._translator import (  # noqa: E402
    detection_overrides,
    load_our_yaml,
    ppdet_base_config_path,
)
from core.p06_paddle.train import _find_ppdet_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PaddlePaddle → ONNX exporter")
    p.add_argument("--config", required=True, type=Path,
                   help="06_training_paddle_*.yaml")
    p.add_argument("--checkpoint", required=True, type=Path,
                   help=".pdparams checkpoint")
    p.add_argument("--out", required=True, type=Path,
                   help="Output ONNX file path")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def _export_inference_model(our: dict, ckpt: Path, work_dir: Path) -> Path:
    import yaml as _yaml
    from ppdet.core.workspace import load_config, merge_config

    base_path = _find_ppdet_config(ppdet_base_config_path(our["model"]["arch"]))
    cfg = load_config(str(base_path))
    merge_config(detection_overrides(our))
    cfg["weights"] = str(ckpt)

    tmp_yaml = work_dir / "exported.yml"
    tmp_yaml.write_text(_yaml.safe_dump(dict(cfg)))

    import ppdet
    pkg_root = Path(ppdet.__file__).resolve().parent
    candidates = [
        pkg_root.parent / "tools" / "export_model.py",
        pkg_root / "tools" / "export_model.py",
    ]
    export_tool = next((c for c in candidates if c.exists()), None)
    if export_tool is None:
        raise FileNotFoundError(
            "ppdet's tools/export_model.py not found. "
            f"Searched: {candidates}. Re-run scripts/setup-paddle-venv.sh."
        )

    cmd = [
        sys.executable, str(export_tool),
        "-c", str(tmp_yaml),
        "--output_dir", str(work_dir),
        "-o", f"weights={ckpt}",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    # ppdet writes to <work_dir>/<arch>/{model.pdmodel,model.pdiparams,...}
    out_dirs = [d for d in work_dir.iterdir() if d.is_dir() and (d / "model.pdmodel").exists()]
    if not out_dirs:
        raise FileNotFoundError(f"export_model produced no model.pdmodel under {work_dir}")
    return out_dirs[0]


def _to_onnx(inference_dir: Path, out_onnx: Path, opset: int) -> None:
    """Run paddle2onnx CLI."""
    paddle2onnx = shutil.which("paddle2onnx") or "paddle2onnx"
    cmd = [
        paddle2onnx,
        "--model_dir", str(inference_dir),
        "--model_filename", "model.pdmodel",
        "--params_filename", "model.pdiparams",
        "--save_file", str(out_onnx),
        "--opset_version", str(opset),
        "--enable_onnx_checker", "True",
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = _parse_args()
    our = load_our_yaml(args.config)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="paddle_export_") as tmp:
        work = Path(tmp)
        inf_dir = _export_inference_model(our, args.checkpoint, work)
        _to_onnx(inf_dir, args.out, args.opset)

    print(f"\n[paddle] exported to: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
