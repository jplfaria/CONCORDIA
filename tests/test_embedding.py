"""
Tests for the embedding module.
"""

import unittest
from unittest import mock

import torch

from concord.embedding import (
    batch_embed,
    clear_cache,
    cosine_sim,
    embed_sentence,
    preload_model,
    similarity,
)


class TestEmbedding(unittest.TestCase):
    """Test suite for embedding functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config
        self.cfg = {"embedding": {"device": "cpu", "batch_size": 8}}

        # Clear cache before each test
        clear_cache()

    def test_embed_sentence(self):
        """Test embedding a single sentence."""
        text = "Test protein sequence"
        embedding = embed_sentence(text, self.cfg)

        # Verify embedding shape and type
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.dim(), 1)  # Should be a 1D tensor
        self.assertEqual(
            embedding.size(0), 768
        )  # PubMedBERT embeddings are 768-dimensional

    def test_embedding_cache(self):
        """Test that embeddings are cached and reused."""
        text = "This text should be cached"

        # First call should compute the embedding
        with mock.patch("concord.embedding._get_model") as mock_get_model:
            # Setup the mock to return a real model for the test
            mock_model = mock.MagicMock()
            mock_model.encode.return_value = torch.ones(768)
            mock_get_model.return_value = mock_model

            # First call
            embed_1 = embed_sentence(text, self.cfg)
            self.assertEqual(mock_model.encode.call_count, 1)

            # Second call should use cache
            embed_2 = embed_sentence(text, self.cfg)
            self.assertEqual(mock_model.encode.call_count, 1)  # Still 1, not 2

            # Verify both embeddings are the same
            self.assertTrue(torch.equal(embed_1, embed_2))

    def test_batch_embed(self):
        """Test batch embedding functionality."""
        texts = ["Text one", "Text two", "Text three"]

        embeddings = batch_embed(texts, self.cfg)

        # Verify we got the right number of embeddings
        self.assertEqual(len(embeddings), len(texts))

        # Verify each embedding is the correct shape
        for embedding in embeddings:
            self.assertIsInstance(embedding, torch.Tensor)
            self.assertEqual(embedding.size(0), 768)

    def test_cosine_sim(self):
        """Test cosine similarity calculation."""
        # Create two simple vectors
        vec1 = torch.tensor([1.0, 0.0, 0.0])
        vec2 = torch.tensor([0.0, 1.0, 0.0])

        # Cosine similarity should be 0 for orthogonal vectors
        sim = cosine_sim(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0)

        # Cosine similarity should be 1 for identical vectors
        sim = cosine_sim(vec1, vec1)
        self.assertAlmostEqual(sim, 1.0)

    def test_similarity(self):
        """Test end-to-end similarity calculation."""
        # Using real sentences
        text1 = "Protein kinase A"
        text2 = "PKA enzyme"

        sim = similarity(text1, text2, self.cfg)

        # Should be a float between -1 and 1
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

    def test_preload_model(self):
        """Test model preloading."""
        with mock.patch("concord.embedding._get_model") as mock_get_model:
            preload_model(self.cfg)
            mock_get_model.assert_called_once()

    def test_clear_cache(self):
        """Test cache clearing."""
        # Add something to the cache
        text = "Cache me"
        embed_sentence(text, self.cfg)

        # Clear cache
        with mock.patch("concord.embedding._embedding_cache", {text: torch.ones(768)}):
            clear_cache()
            from concord.embedding import _embedding_cache

            self.assertEqual(len(_embedding_cache), 0)


if __name__ == "__main__":
    unittest.main()
