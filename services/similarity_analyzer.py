"""
Similarity Analyzer Module

Calculates semantic similarity between URLs and pairs them based on content similarity.
"""

import pandas as pd
import numpy as np
import torch
from typing import List, Optional
from sentence_transformers import util
import streamlit as st
from utils.page_classifier import PageClassifier
from utils.url_similarity import URLSimilarityCalculator


class SimilarityAnalyzer:
    """Analyzes semantic similarity between URL pairs from two domains."""

    def __init__(
        self,
        top_k: int = 5,
        use_hybrid_scoring: bool = True,
        semantic_weight: float = 0.5,
        url_weight: float = 0.3,
        page_type_weight: float = 0.2
    ):
        """
        Initialize the similarity analyzer.

        Args:
            top_k: Number of top matches to return per URL
            use_hybrid_scoring: Whether to use hybrid scoring (semantic + URL + page type)
            semantic_weight: Weight for semantic similarity (0-1)
            url_weight: Weight for URL similarity (0-1)
            page_type_weight: Weight for page type compatibility (0-1)
        """
        self.top_k = top_k
        self.use_hybrid_scoring = use_hybrid_scoring
        self.semantic_weight = semantic_weight
        self.url_weight = url_weight
        self.page_type_weight = page_type_weight

        # Initialize helper classes for hybrid scoring
        if use_hybrid_scoring:
            self.page_classifier = PageClassifier()
            self.url_calculator = URLSimilarityCalculator()
        else:
            self.page_classifier = None
            self.url_calculator = None

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
        similarity_matrix: np.ndarray,
        titles_a: Optional[List[str]] = None,
        titles_b: Optional[List[str]] = None,
        h1s_a: Optional[List[str]] = None,
        h1s_b: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Create URL pairs based on similarity scores.

        For each URL in Domain A, find the top K most similar URLs from Domain B.

        Args:
            urls_a: List of URLs from Domain A
            urls_b: List of URLs from Domain B
            similarity_matrix: Semantic similarity matrix (shape: [len(urls_a), len(urls_b)])
            titles_a: Optional list of titles for Domain A (for page type classification)
            titles_b: Optional list of titles for Domain B (for page type classification)
            h1s_a: Optional list of H1s for Domain A (for page type classification)
            h1s_b: Optional list of H1s for Domain B (for page type classification)

        Returns:
            DataFrame with columns:
                - domain_a_url: URL from Domain A
                - domain_b_url: URL from Domain B
                - similarity_score: Overall similarity score (0-100%)
                - semantic_score: Semantic similarity component (0-100%)
                - url_score: URL similarity component (0-100%)
                - page_type_score: Page type compatibility component (0-100%)
                - page_type_a: Page type for Domain A URL
                - page_type_b: Page type for Domain B URL
                - rank: Rank of match (1 = most similar)
        """
        results = []

        # Classify page types if hybrid scoring enabled
        if self.use_hybrid_scoring:
            with st.spinner("🏷️ Classifying page types..."):
                page_types_a = self.page_classifier.classify_batch(
                    urls_a,
                    titles_a,
                    h1s_a
                )
                page_types_b = self.page_classifier.classify_batch(
                    urls_b,
                    titles_b,
                    h1s_b
                )
            st.success("✅ Page types classified")

        with st.spinner(f"📊 Pairing URLs with hybrid scoring (top {self.top_k} matches per URL)..."):
            for i, url_a in enumerate(urls_a):
                # Calculate hybrid scores for this URL against all URLs in Domain B
                hybrid_scores = []

                for j, url_b in enumerate(urls_b):
                    # Get semantic similarity
                    semantic_score = similarity_matrix[i][j]

                    if self.use_hybrid_scoring:
                        # Calculate URL similarity
                        url_score = self.url_calculator.calculate_similarity(url_a, url_b)

                        # Get page type compatibility
                        page_type_a = page_types_a[i]
                        page_type_b = page_types_b[j]
                        page_type_score = self.page_classifier.get_page_type_compatibility(
                            page_type_a,
                            page_type_b
                        )

                        # Combine scores with weights
                        hybrid_score = (
                            semantic_score * self.semantic_weight +
                            url_score * self.url_weight +
                            page_type_score * self.page_type_weight
                        )
                    else:
                        # Use only semantic score
                        hybrid_score = semantic_score
                        url_score = 0.0
                        page_type_score = 0.0
                        page_type_a = 'unknown'
                        page_type_b = 'unknown'

                    hybrid_scores.append({
                        'hybrid_score': hybrid_score,
                        'semantic_score': semantic_score,
                        'url_score': url_score,
                        'page_type_score': page_type_score,
                        'page_type_a': page_type_a if self.use_hybrid_scoring else 'unknown',
                        'page_type_b': page_type_b if self.use_hybrid_scoring else 'unknown',
                        'url_b': url_b,
                        'index': j
                    })

                # Sort by hybrid score
                hybrid_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)

                # Get top K matches
                top_matches = hybrid_scores[:self.top_k]

                # Create pairs
                for rank, match in enumerate(top_matches, start=1):
                    results.append({
                        'domain_a_url': url_a,
                        'domain_b_url': match['url_b'],
                        'similarity_score': round(match['hybrid_score'] * 100, 2),
                        'semantic_score': round(match['semantic_score'] * 100, 2),
                        'url_score': round(match['url_score'] * 100, 2),
                        'page_type_score': round(match['page_type_score'] * 100, 2),
                        'page_type_a': match['page_type_a'],
                        'page_type_b': match['page_type_b'],
                        'rank': rank
                    })

        st.success(f"✅ Created {len(results)} URL pairs with hybrid scoring")

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
