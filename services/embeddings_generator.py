"""
Embeddings Generator Module

Generates semantic embeddings from text content using sentence-transformers.
"""

import pandas as pd
import torch
from typing import List
from sentence_transformers import SentenceTransformer
import streamlit as st


class EmbeddingsGenerator:
    """Handles embedding generation for semantic similarity analysis."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embeddings generator.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the sentence-transformers model."""
        if self.model is None:
            with st.spinner("🤖 Loading semantic similarity model..."):
                self.model = SentenceTransformer(self.model_name)
            st.success("✅ Semantic model loaded successfully")

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate embeddings from text content.

        Args:
            texts: List of text strings to encode
            show_progress: Whether to show progress bar (for Streamlit)

        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            self.load_model()

        if show_progress:
            with st.spinner(f"🔄 Generating embeddings for {len(texts)} URLs..."):
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False  # We'll use Streamlit's spinner instead
                )
            st.success(f"✅ Generated {len(texts)} embeddings")
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )

        return embeddings

    def load_pregenerated_embeddings(
        self,
        df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Load pre-generated embeddings from DataFrame.

        Args:
            df: DataFrame with embedding columns (embedding_0, embedding_1, etc.)

        Returns:
            Tensor of embeddings
        """
        # Get embedding columns
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]

        if len(embedding_cols) == 0:
            raise ValueError("No embedding columns found in DataFrame")

        # Sort by column number
        embedding_cols = sorted(
            embedding_cols,
            key=lambda x: int(x.split('_')[1])
        )

        # Convert to numpy array then to tensor
        embeddings_array = df[embedding_cols].values

        # Convert to tensor
        embeddings_tensor = torch.tensor(
            embeddings_array,
            dtype=torch.float32
        )

        st.success(f"✅ Loaded {len(embeddings_tensor)} pre-generated embeddings")

        return embeddings_tensor

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the loaded model.

        Returns:
            Embedding dimension (int)
        """
        if not self.model:
            self.load_model()

        # Get dimension from model config
        return self.model.get_sentence_embedding_dimension()
