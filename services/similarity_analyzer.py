"""
Similarity Analyzer Module

Calculates semantic similarity between URLs and pairs them based on content similarity.
"""

import pandas as pd
import numpy as np
import torch
from typing import List
from sentence_transformers import util
import streamlit as st


class SimilarityAnalyzer:
    """Analyzes semantic similarity between URL pairs from two domains."""

    def __init__(self, top_k: int = 5):
        """
        Initialize the similarity analyzer.

        Args:
            top_k: Number of top matches to return per URL
        """
        self.top_k = top_k

    def calculate_similarity_matrix(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Calculate cosine similarity matrix between two sets of embeddings.

        Args:
            embeddings_a: Embeddings for Domain A URLs (shape: [n_urls_a, embedding_dim])
            embeddings_b: Embeddings for Domain B URLs (shape: [n_urls_b, embedding_dim])
            show_progress: Whether to show progress feedback

        Returns:
            Similarity matrix of shape (n_urls_a, n_urls_b) with values from 0 to 1
        """
        if show_progress:
            with st.spinner(
                f"🔄 Calculating similarity for {len(embeddings_a)} × {len(embeddings_b)} = "
                f"{len(embeddings_a) * len(embeddings_b):,} URL pairs..."
            ):
                similarity_matrix = util.cos_sim(embeddings_a, embeddings_b)
            st.success("✅ Similarity calculation complete")
        else:
            similarity_matrix = util.cos_sim(embeddings_a, embeddings_b)

        # Convert to numpy array for easier manipulation
        return similarity_matrix.cpu().numpy()

    def pair_urls(
        self,
        urls_a: List[str],
        urls_b: List[str],
        similarity_matrix: np.ndarray
    ) -> pd.DataFrame:
        """
        Create URL pairs based on similarity scores.

        For each URL in Domain A, find the top K most similar URLs from Domain B.

        Args:
            urls_a: List of URLs from Domain A
            urls_b: List of URLs from Domain B
            similarity_matrix: Similarity matrix (shape: [len(urls_a), len(urls_b)])

        Returns:
            DataFrame with columns:
                - domain_a_url: URL from Domain A
                - domain_b_url: URL from Domain B
                - similarity_score: Similarity score (0-100%)
                - rank: Rank of match (1 = most similar)
        """
        results = []

        with st.spinner(f"📊 Pairing URLs (top {self.top_k} matches per URL)..."):
            for i, url_a in enumerate(urls_a):
                # Get similarity scores for this URL
                similarities = similarity_matrix[i]

                # Get top K indices (most similar)
                top_indices = np.argsort(similarities)[::-1][:self.top_k]

                # Create pairs
                for rank, idx in enumerate(top_indices, start=1):
                    results.append({
                        'domain_a_url': url_a,
                        'domain_b_url': urls_b[idx],
                        'similarity_score': round(similarities[idx] * 100, 2),
                        'rank': rank
                    })

        st.success(f"✅ Created {len(results)} URL pairs")

        return pd.DataFrame(results)

    def generate_summary_stats(self, results_df: pd.DataFrame) -> dict:
        """
        Generate summary statistics from results.

        Args:
            results_df: DataFrame of URL pairs

        Returns:
            Dictionary with summary statistics
        """
        # Get only rank 1 matches for summary stats
        top_matches = results_df[results_df['rank'] == 1]

        stats = {
            'total_domain_a_urls': results_df['domain_a_url'].nunique(),
            'total_domain_b_urls': results_df['domain_b_url'].nunique(),
            'total_pairs': len(results_df),
            'avg_similarity': round(top_matches['similarity_score'].mean(), 2),
            'median_similarity': round(top_matches['similarity_score'].median(), 2),
            'min_similarity': round(top_matches['similarity_score'].min(), 2),
            'max_similarity': round(top_matches['similarity_score'].max(), 2),
            'high_similarity_count': len(top_matches[top_matches['similarity_score'] >= 80]),
            'medium_similarity_count': len(
                top_matches[
                    (top_matches['similarity_score'] >= 50) &
                    (top_matches['similarity_score'] < 80)
                ]
            ),
            'low_similarity_count': len(top_matches[top_matches['similarity_score'] < 50])
        }

        return stats

    def filter_by_threshold(
        self,
        results_df: pd.DataFrame,
        min_similarity: float = 0.0,
        max_similarity: float = 100.0
    ) -> pd.DataFrame:
        """
        Filter results by similarity threshold.

        Args:
            results_df: DataFrame of URL pairs
            min_similarity: Minimum similarity score (0-100)
            max_similarity: Maximum similarity score (0-100)

        Returns:
            Filtered DataFrame
        """
        filtered = results_df[
            (results_df['similarity_score'] >= min_similarity) &
            (results_df['similarity_score'] <= max_similarity)
        ]

        return filtered

    def get_top_matches_for_url(
        self,
        results_df: pd.DataFrame,
        url: str
    ) -> pd.DataFrame:
        """
        Get top matches for a specific URL from Domain A.

        Args:
            results_df: DataFrame of URL pairs
            url: URL from Domain A to filter by

        Returns:
            DataFrame of top matches for this URL
        """
        matches = results_df[results_df['domain_a_url'] == url].copy()
        matches = matches.sort_values('rank')

        return matches
