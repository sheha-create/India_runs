"""
modules/embedder.py — Embedding Layer (PRD §9, step 3)

Uses sentence-transformers to generate dense vector embeddings of role
profile and candidate profile texts, then computes cosine similarity for
fast first-pass ranking.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import config
from schemas.candidate_profile import CandidateProfile
from schemas.role_profile import RoleProfile

logger = logging.getLogger(__name__)


def _load_model():
    """Lazy-load sentence-transformers model (downloaded once, cached)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        return SentenceTransformer(config.EMBEDDING_MODEL)
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers not installed. Run: pip install sentence-transformers"
        ) from exc


class Embedder:
    """Generates embeddings and computes similarity scores."""

    def __init__(self) -> None:
        self._model = None  # lazy init

    @property
    def model(self):
        if self._model is None:
            self._model = _load_model()
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return a 2D numpy array of shape (len(texts), embedding_dim)."""
        return self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def rank_candidates(
        self,
        role_profile: RoleProfile,
        candidate_profiles: List[CandidateProfile],
        top_n: int | None = None,
    ) -> List[Tuple[CandidateProfile, float]]:
        """
        Rank candidates by cosine similarity to the role profile.

        Args:
            role_profile: Structured job requirements.
            candidate_profiles: List of normalized candidate profiles.
            top_n: If set, return only the top-N candidates.

        Returns:
            List of (CandidateProfile, similarity_score) tuples, sorted descending.
            similarity_score is normalized to 0–100.
        """
        if not candidate_profiles:
            return []

        role_text = role_profile.to_embedding_text()
        candidate_texts = [c.to_embedding_text() for c in candidate_profiles]

        logger.info(
            "Generating embeddings for 1 role profile + %d candidates...",
            len(candidate_profiles),
        )
        all_texts = [role_text] + candidate_texts
        all_embeddings = self.embed_texts(all_texts)

        role_embedding = all_embeddings[0:1]          # shape (1, dim)
        candidate_embeddings = all_embeddings[1:]     # shape (N, dim)

        # Cosine similarity → values in [-1, 1], but practically [0, 1] for text
        similarities = cosine_similarity(role_embedding, candidate_embeddings)[0]

        # Normalize to 0–100
        scores_normalized = (similarities.clip(0, 1) * 100).tolist()

        ranked = sorted(
            zip(candidate_profiles, scores_normalized),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_n is not None:
            ranked = ranked[:top_n]

        logger.info(
            "Embedding ranking complete. Top score: %.1f | Bottom score: %.1f",
            ranked[0][1] if ranked else 0,
            ranked[-1][1] if ranked else 0,
        )
        return ranked
