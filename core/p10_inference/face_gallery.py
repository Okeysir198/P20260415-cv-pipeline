"""Face gallery for enrollment and identity matching.

Stores face embeddings as a ``.npz`` file with numpy arrays. Matching uses
cosine similarity (equivalent to L2 distance on L2-normalized vectors) and
runs on CUDA via a persistent ``torch.Tensor`` gallery matrix.
"""

from loguru import logger
from pathlib import Path

import numpy as np
import torch



class FaceGallery:
    """Manage enrolled face embeddings for identity matching.

    Gallery stored as ``.npz``: embeddings ``(N, D)`` float32 +
    identities ``(N,)`` str array.

    Matching executes on CUDA: the gallery matrix is uploaded once and
    cached as a ``torch.Tensor``; each query is uploaded and the cosine
    similarity is computed with ``torch.matmul`` + ``torch.argmax``. The
    cache is invalidated when ``enroll()``/``remove()``/``load()`` mutates
    the gallery.

    Args:
        gallery_path: Path to ``.npz`` gallery file.
        similarity_threshold: Minimum cosine similarity to accept a match.
        embedding_dim: Expected embedding dimension (default 512).
    """

    def __init__(
        self,
        gallery_path: str,
        similarity_threshold: float = 0.4,
        embedding_dim: int = 512,
    ) -> None:
        self.gallery_path = Path(gallery_path)
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self._embeddings: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self._identities: list[str] = []
        self._embeddings_gpu: torch.Tensor | None = None
        self._device = torch.device("cuda")

        if self.gallery_path.exists():
            self.load()

    def _invalidate_gpu_cache(self) -> None:
        """Drop the cached CUDA tensor; rebuilt lazily on next match call."""
        self._embeddings_gpu = None

    def _ensure_gpu_cache(self) -> torch.Tensor:
        """Return the gallery matrix on CUDA, building+caching it if needed."""
        if self._embeddings_gpu is None:
            self._embeddings_gpu = torch.from_numpy(
                self._embeddings.astype(np.float32, copy=False)
            ).to(self._device)
        return self._embeddings_gpu

    def enroll(self, identity: str, embedding: np.ndarray) -> None:
        """Add a face embedding to the gallery.

        Args:
            identity: Person identifier (e.g. employee ID or name).
            embedding: ``(D,)`` float32 L2-normalized embedding.
        """
        embedding = embedding.astype(np.float32).reshape(1, -1)
        # Ensure L2 normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        self._embeddings = np.vstack([self._embeddings, embedding])
        self._identities.append(identity)
        self._invalidate_gpu_cache()
        logger.info("Enrolled '%s' (gallery size: %d)", identity, self.size)

    def match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Find the closest identity in the gallery.

        Args:
            embedding: ``(D,)`` float32 L2-normalized query embedding.

        Returns:
            Tuple of ``(identity, similarity)``. Returns
            ``("unknown", 0.0)`` if gallery is empty or below threshold.
        """
        if self.size == 0:
            return ("unknown", 0.0)

        gallery = self._ensure_gpu_cache()  # (N, D)

        # Upload + L2-normalize the query on GPU
        q = torch.from_numpy(embedding.astype(np.float32, copy=False)).to(self._device)
        q = q.reshape(-1)
        q_norm = torch.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity via matmul (gallery rows are L2-normalized)
        sims = gallery @ q  # (N,)
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())

        if best_sim >= self.similarity_threshold:
            return (self._identities[best_idx], best_sim)
        return ("unknown", best_sim)

    def match_batch(self, embeddings: np.ndarray) -> list[tuple[str, float]]:
        """Match multiple query embeddings against the gallery.

        Args:
            embeddings: ``(M, D)`` float32 L2-normalized embeddings.

        Returns:
            List of *M* ``(identity, similarity)`` tuples.
        """
        if self.size == 0 or len(embeddings) == 0:
            return [("unknown", 0.0)] * len(embeddings)

        gallery = self._ensure_gpu_cache()  # (N, D)

        # Upload queries and L2-normalize row-wise (handles un-normalized input)
        q = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(self._device)
        q_norms = torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-8)
        q = q / q_norms

        sim_matrix = q @ gallery.T  # (M, N)
        best_sims, best_indices = sim_matrix.max(dim=1)

        # One round-trip to CPU at the boundary
        best_sims_cpu = best_sims.cpu().tolist()
        best_indices_cpu = best_indices.cpu().tolist()

        results: list[tuple[str, float]] = []
        for sim, idx in zip(best_sims_cpu, best_indices_cpu, strict=True):
            if sim >= self.similarity_threshold:
                results.append((self._identities[idx], float(sim)))
            else:
                results.append(("unknown", float(sim)))
        return results

    def remove(self, identity: str) -> int:
        """Remove all embeddings for an identity.

        Args:
            identity: Person identifier to remove.

        Returns:
            Number of embeddings removed.
        """
        mask = np.array([ident != identity for ident in self._identities])
        removed = int(np.sum(~mask))
        self._embeddings = self._embeddings[mask]
        self._identities = [ident for ident, keep in zip(self._identities, mask, strict=True) if keep]
        if removed > 0:
            self._invalidate_gpu_cache()
            logger.info(
                "Removed '%s' (%d embeddings, gallery size: %d)",
                identity, removed, self.size,
            )
        return removed

    def save(self) -> None:
        """Persist gallery to ``.npz`` file."""
        self.gallery_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(self.gallery_path),
            embeddings=self._embeddings,
            identities=np.array(self._identities, dtype=str),
        )
        logger.info("Saved gallery to %s (%d identities)", self.gallery_path, self.size)

    def load(self) -> None:
        """Load gallery from ``.npz`` file."""
        data = np.load(str(self.gallery_path), allow_pickle=False)
        self._embeddings = data["embeddings"].astype(np.float32)
        self._identities = list(data["identities"])
        self._invalidate_gpu_cache()
        logger.info("Loaded gallery from %s (%d identities)", self.gallery_path, self.size)

    @property
    def size(self) -> int:
        """Number of enrolled embeddings."""
        return len(self._identities)

    @property
    def unique_identities(self) -> list[str]:
        """List of unique enrolled identity names."""
        return sorted(set(self._identities))
