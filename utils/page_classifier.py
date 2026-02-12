"""
Page Type Classifier

Detects page types based on URL patterns and content characteristics
to prevent mismatched pairings (e.g., contact pages with blog posts).
"""

import re
from typing import Dict, List
from urllib.parse import urlparse


class PageClassifier:
    """Classifies pages into types for better matching."""

    # Page type patterns (ordered by priority)
    PAGE_TYPES = {
        'homepage': {
            'url_patterns': [r'^/$', r'^/index', r'^/home$', r'^/default'],
            'exact_paths': ['/', '/index.html', '/index.php', '/home', '/default.html']
        },
        'contact': {
            'url_patterns': [r'/contact', r'/get-in-touch', r'/reach-us', r'/inquiry'],
            'keywords': ['contact', 'reach us', 'get in touch', 'contact us']
        },
        'about': {
            'url_patterns': [r'/about', r'/who-we-are', r'/our-story', r'/company'],
            'keywords': ['about us', 'about', 'our story', 'who we are', 'our team']
        },
        'services': {
            'url_patterns': [r'/services?/', r'/what-we-do', r'/solutions'],
            'keywords': ['services', 'what we do', 'solutions', 'offerings']
        },
        'products': {
            'url_patterns': [r'/products?/', r'/shop', r'/store', r'/catalog'],
            'keywords': ['products', 'shop', 'store', 'buy', 'catalog']
        },
        'blog': {
            'url_patterns': [r'/blog/', r'/articles?/', r'/posts?/', r'/news/', r'/resources/'],
            'keywords': ['blog', 'article', 'post', 'published', 'author', 'read more']
        },
        'faq': {
            'url_patterns': [r'/faq', r'/questions', r'/help'],
            'keywords': ['frequently asked', 'faq', 'questions', 'help center']
        },
        'privacy': {
            'url_patterns': [r'/privacy', r'/terms', r'/legal', r'/policy'],
            'keywords': ['privacy policy', 'terms of service', 'legal', 'cookie policy']
        },
        'location': {
            'url_patterns': [r'/locations?/', r'/find-us', r'/stores/', r'/branches/'],
            'keywords': ['location', 'find us', 'stores', 'branches', 'near you']
        },
        'careers': {
            'url_patterns': [r'/careers?', r'/jobs', r'/hiring', r'/join-us'],
            'keywords': ['careers', 'jobs', 'hiring', 'join our team', 'opportunities']
        }
    }

    def __init__(self):
        """Initialize the page classifier."""
        pass

    def classify_url(self, url: str, title: str = '', h1: str = '') -> str:
        """
        Classify a URL into a page type.

        Args:
            url: The page URL
            title: Page title (optional, helps classification)
            h1: Page H1 (optional, helps classification)

        Returns:
            Page type string (e.g., 'blog', 'contact', 'product', 'other')
        """
        # Parse URL
        parsed = urlparse(url)
        path = parsed.path.lower().rstrip('/')

        # Combine text content for keyword matching
        content_text = f"{title} {h1}".lower()

        # Check each page type
        for page_type, patterns in self.PAGE_TYPES.items():
            # Check exact paths
            if 'exact_paths' in patterns:
                if path in patterns['exact_paths']:
                    return page_type

            # Check URL patterns
            if 'url_patterns' in patterns:
                for pattern in patterns['url_patterns']:
                    if re.search(pattern, path):
                        return page_type

            # Check keywords in content
            if 'keywords' in patterns and content_text:
                for keyword in patterns['keywords']:
                    if keyword in content_text:
                        return page_type

        # If no match, classify as 'other'
        return 'other'

    def classify_batch(
        self,
        urls: List[str],
        titles: List[str] = None,
        h1s: List[str] = None
    ) -> List[str]:
        """
        Classify a batch of URLs.

        Args:
            urls: List of URLs
            titles: List of titles (optional)
            h1s: List of H1s (optional)

        Returns:
            List of page types
        """
        titles = titles or [''] * len(urls)
        h1s = h1s or [''] * len(urls)

        return [
            self.classify_url(url, title, h1)
            for url, title, h1 in zip(urls, titles, h1s)
        ]

    def get_page_type_compatibility(self, type_a: str, type_b: str) -> float:
        """
        Get compatibility score between two page types.

        Returns:
            Compatibility score from 0.0 (incompatible) to 1.0 (perfect match)
        """
        # Exact match
        if type_a == type_b:
            return 1.0

        # Incompatible pairs (should never match)
        incompatible_pairs = [
            {'contact', 'blog'},
            {'contact', 'products'},
            {'contact', 'services'},
            {'about', 'blog'},
            {'about', 'products'},
            {'faq', 'blog'},
            {'privacy', 'blog'},
            {'careers', 'blog'},
            {'homepage', 'blog'}
        ]

        pair = {type_a, type_b}
        for incompatible in incompatible_pairs:
            if pair == incompatible:
                return 0.0

        # Related but not exact (moderate compatibility)
        related_pairs = {
            ('services', 'products'): 0.5,
            ('blog', 'other'): 0.3,  # Blog can match unknown content pages
            ('products', 'other'): 0.4,
            ('services', 'other'): 0.4,
            ('location', 'contact'): 0.6,
            ('homepage', 'other'): 0.2
        }

        for (t1, t2), score in related_pairs.items():
            if (type_a == t1 and type_b == t2) or (type_a == t2 and type_b == t1):
                return score

        # Default for unrelated types
        return 0.3
