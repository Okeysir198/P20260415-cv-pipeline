"""Lazy-loaded SAM3.1 multiplex predictor singleton."""

from __future__ import annotations

import gzip
import os
import shutil
import threading

from src.config import config, logger

_predictor = None
_predictor_lock = threading.Lock()
# Global inference lock: SAM3.1 predictor is not thread-safe for concurrent sessions
inference_lock = threading.Lock()


def device() -> str:
    """Return the device the predictor is running on."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def loaded_models() -> dict[str, bool]:
    """Return model load status — single predictor serves all roles."""
    loaded = _predictor is not None
    return {"text": loaded, "tracker": loaded, "video": loaded}


def get_predictor():
    """Return the SAM3.1 multiplex predictor (lazy-loaded, thread-safe singleton)."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from huggingface_hub import hf_hub_download
                from sam3.model_builder import build_sam3_multiplex_video_predictor

                name = config["model"]["name"]  # "facebook/sam3.1"
                logger.info("Loading SAM3.1 multiplex predictor from %s", name)

                ckpt_path = hf_hub_download(name, "sam3.1_multiplex.pt")
                merges_path = hf_hub_download(name, "merges.txt")

                # SAM3 tokenizer requires a gzip-compressed BPE merges file
                bpe_gz = merges_path + ".gz"
                if not os.path.exists(bpe_gz):
                    with open(merges_path, "rb") as f_in:
                        with gzip.open(bpe_gz, "wb") as f_out:
                            shutil.copyfileobj(f_in, f_out)

                _predictor = build_sam3_multiplex_video_predictor(
                    checkpoint_path=ckpt_path,
                    bpe_path=bpe_gz,
                    max_num_objects=200,  # support large auto_mask sweeps
                    use_fa3=False,        # FA3 requires special CUDA build
                    use_rope_real=True,
                    compile=False,
                    warm_up=False,
                )
                logger.info("SAM3.1 multiplex predictor loaded")
    return _predictor
