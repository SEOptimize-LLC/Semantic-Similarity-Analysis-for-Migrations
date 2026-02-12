"""
URL Similarity Calculator

Calculates URL similarity based on path structure, segments, and patterns
to complement semantic similarity scoring.
"""

import re
from typing import List
from urllib.parse import urlparse
from difflib import SequenceMatcher


class URLSimilarityCalculator:
    """Calculates similarity between URLs based on structure."""

    def __init__(self):
        """Initialize the URL similarity calculator."""
        pass

    def calculate_similarity(self, url_a: str, url_b: str) -> float:
        """
        Calculate similarity between two URLs.

        Combines multiple similarity metrics:
        - Path similarity (Levenshtein-like)
        - Segment overlap
        - Slug similarity

        Args:
            url_a: First URL
            url_b: Second URL

        Returns:
            Similarity score from 0.0 to 1.0
        """
        # Parse URLs
        parsed_a = urlparse(url_a)
        parsed_b = urlparse(url_b)

        path_a = parsed_a.path.lower().strip('/').rstrip('/')
        path_b = parsed_b.path.lower().strip('/').rstrip('/')

        # If both are homepage/root
        if not path_a and not path_b:
            return 1.0

        # If one is homepage and other is not
        if (not path_a and path_b) or (path_a and not path_b):
            return 0.0

        # Calculate path similarity (sequence matcher)
        path_similarity = SequenceMatcher(None, path_a, path_b).ratio()

        # Calculate segment overlap
        segments_a = [s for s in path_a.split('/') if s]
        segments_b = [s for s in path_b.split('/') if s]

        segment_similarity = self._calculate_segment_similarity(segments_a, segments_b)

        # Calculate slug similarity (last segment - often most important)
        slug_similarity = self._calculate_slug_similarity(segments_a, segments_b)

        # Weighted combination
        url_similarity = (
            path_similarity * 0.3 +
            segment_similarity * 0.4 +
            slug_similarity * 0.3
        )

        return url_similarity

    def _calculate_segment_similarity(
        self,
        segments_a: List[str],
        segments_b: List[str]
    ) -> float:
        """
        Calculate similarity based on segment overlap.

        Args:
            segments_a: URL path segments from URL A
            segments_b: URL path segments from URL B

        Returns:
            Segment similarity score (0-1)
        """
        if not segments_a or not segments_b:
            return 0.0

        # Count matching segments
        common_segments = set(segments_a) & set(segments_b)

        # Jaccard similarity
        union_segments = set(segments_a) | set(segments_b)

        if not union_segments:
            return 0.0

        return len(common_segments) / len(union_segments)

    def _calculate_slug_similarity(
        self,
        segments_a: List[str],
        segments_b: List[str]
    ) -> float:
        """
        Calculate similarity between the last segments (slugs).

        Args:
            segments_a: URL path segments from URL A
            segments_b: URL path segments from URL B

        Returns:
            Slug similarity score (0-1)
        """
        if not segments_a or not segments_b:
            return 0.0

        slug_a = segments_a[-1]
        slug_b = segments_b[-1]

        # Exact match
        if slug_a == slug_b:
            return 1.0

        # Sequence similarity
        return SequenceMatcher(None, slug_a, slug_b).ratio()

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL for comparison.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        path = parsed.path.lower().strip('/').rstrip('/')

        # Remove common file extensions
        path = re.sub(r'\.(html|php|aspx|htm)$', '', path)

        # Remove query parameters and fragments
        return path
