"""Face gallery for enrollment and identity matching.

Stores face embeddings as a ``.npz`` file with numpy arrays. Matching uses
cosine similarity (equivalent to L2 distance on L2-normalized vectors).
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FaceGallery:
    """Manage enrolled face embeddings for identity matching.

    Gallery stored as ``.npz``: embeddings ``(N, D)`` float32 +
    identities ``(N,)`` str array.

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
        self._identities: List[str] = []

        if self.gallery_path.exists():
            self.load()

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
        logger.info("Enrolled '%s' (gallery size: %d)", identity, self.size)

    def match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Find the closest identity in the gallery.

        Args:
            embedding: ``(D,)`` float32 L2-normalized query embedding.

        Returns:
            Tuple of ``(identity, similarity)``. Returns
            ``("unknown", 0.0)`` if gallery is empty or below threshold.
        """
        if self.size == 0:
            return ("unknown", 0.0)

        embedding = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Cosine similarity (embeddings are L2-normalized -> dot product)
        similarities = (self._embeddings @ embedding.T).squeeze(-1)  # (N,)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self.similarity_threshold:
            return (self._identities[best_idx], best_sim)
        return ("unknown", best_sim)

    def match_batch(self, embeddings: np.ndarray) -> List[Tuple[str, float]]:
        """Match multiple query embeddings against the gallery.

        Args:
            embeddings: ``(M, D)`` float32 L2-normalized embeddings.

        Returns:
            List of *M* ``(identity, similarity)`` tuples.
        """
        if self.size == 0 or len(embeddings) == 0:
            return [("unknown", 0.0)] * len(embeddings)

        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        # (M, N) similarity matrix
        sim_matrix = embeddings @ self._embeddings.T
        best_indices = np.argmax(sim_matrix, axis=1)
        best_sims = sim_matrix[np.arange(len(embeddings)), best_indices]

        results = []
        for i in range(len(embeddings)):
            sim = float(best_sims[i])
            if sim >= self.similarity_threshold:
                results.append((self._identities[best_indices[i]], sim))
            else:
                results.append(("unknown", sim))
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
        self._identities = [ident for ident, keep in zip(self._identities, mask) if keep]
        if removed > 0:
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
        logger.info("Loaded gallery from %s (%d identities)", self.gallery_path, self.size)

    @property
    def size(self) -> int:
        """Number of enrolled embeddings."""
        return len(self._identities)

    @property
    def unique_identities(self) -> List[str]:
        """List of unique enrolled identity names."""
        return sorted(set(self._identities))
