"""
Tests for the pipeline module.
"""

import unittest
import pathlib as P
import pandas as pd
import tempfile
import yaml
from unittest import mock
from typing import Dict, Any, List, Tuple

from concord.pipeline import (
    run_pair, run_file, _annotate_pair, 
    _load_cfg, _call_llm, _duo_vote
)


class TestPipeline(unittest.TestCase):
    """Test suite for the annotation pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary config file
        self.config = {
            "engine": {"mode": "llm"},
            "llm": {"model": "test-model", "temperature": 0.7},
            "embedding": {"device": "cpu"}
        }
        
        # Create a temporary config file - write as string instead of using yaml.dump directly
        self.config_file = tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w", encoding="utf-8")
        yaml.dump(self.config, self.config_file)
        self.config_file.close()
        self.config_path = P.Path(self.config_file.name)
        
        # Test data for annotation
        self.text_a = "Glucose phosphate isomerase"
        self.text_b = "Phosphoglucose isomerase"
        
        # Create a small test DataFrame
        self.test_df = pd.DataFrame({
            "text_a": ["Protein A", "Enzyme B", "Factor C"],
            "text_b": ["Protein alpha", "Transferase B", "Unknown protein"]
        })
        
        # Create a temporary CSV file
        self.csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8")
        self.csv_file.close()
        self.test_df.to_csv(self.csv_file.name, index=False)
        self.csv_path = P.Path(self.csv_file.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Delete temporary files
        self.config_path.unlink()
        self.csv_path.unlink()
    
    def test_load_cfg(self):
        """Test loading and validating configuration."""
        cfg = _load_cfg(self.config_path)
        
        # Check that we got a valid config
        self.assertEqual(cfg["engine"]["mode"], "llm")
        self.assertEqual(cfg["llm"]["model"], "test-model")
    
    @mock.patch('concord.pipeline._call_llm')
    @mock.patch('concord.pipeline.embed_sentence')
    @mock.patch('concord.pipeline.cosine_sim')
    def test_annotate_pair_llm_mode(self, mock_sim, mock_embed, mock_call_llm):
        """Test annotating a single pair with LLM mode."""
        # Setup mocks
        mock_call_llm.return_value = ("Synonym", "These are different names for the same enzyme")
        
        # Run annotation
        result = _annotate_pair(self.text_a, self.text_b, self.config)
        
        # Verify results
        self.assertEqual(result["label"], "Synonym")
        self.assertEqual(result["evidence"], "These are different names for the same enzyme")
        
        # Verify LLM was called
        mock_call_llm.assert_called_once()
        
        # Verify embed_sentence was NOT called (llm mode doesn't use embeddings)
        mock_embed.assert_not_called()
    
    @mock.patch('concord.pipeline.embed_sentence')
    @mock.patch('concord.pipeline.cosine_sim')
    def test_annotate_pair_local_mode(self, mock_sim, mock_embed):
        """Test annotating a single pair with local mode."""
        # Setup local mode config and mocks
        local_config = {**self.config, "engine": {"mode": "local"}}
        mock_embed.return_value = "mock_embedding"
        mock_sim.return_value = 0.99  # High similarity
        
        # Run annotation
        result = _annotate_pair(self.text_a, self.text_b, local_config)
        
        # Verify results (should be "Exact" with high similarity)
        self.assertEqual(result["label"], "Exact")
        self.assertEqual(result["evidence"], "")
        self.assertEqual(result["similarity"], 0.99)
        
        # Verify embedding was used
        mock_embed.assert_called()
        mock_sim.assert_called_once()
    
    @mock.patch('concord.pipeline._call_llm')
    def test_duo_vote_agreement(self, mock_call_llm):
        """Test duo voting when calls agree."""
        # Setup mock to return the same label twice
        mock_call_llm.side_effect = [
            ("Synonym", "Evidence 1"),
            ("Synonym", "Evidence 2")
        ]
        
        # Use a valid template string with placeholders
        valid_template = "A: {A}\nB: {B}\nLabel: "
        
        # Run duo vote
        label, evidence, conflict = _duo_vote(
            self.text_a, self.text_b, valid_template, self.config
        )
        
        # Verify results
        self.assertEqual(label, "Synonym")
        self.assertEqual(evidence, "Evidence 1")
        self.assertFalse(conflict)
        
        # Verify LLM was called twice (no tie-breaker needed)
        self.assertEqual(mock_call_llm.call_count, 2)
    
    @mock.patch('concord.pipeline._call_llm')
    def test_duo_vote_disagreement(self, mock_call_llm):
        """Test duo voting when calls disagree (needs tie-breaker)."""
        # Setup mock to return different labels
        mock_call_llm.side_effect = [
            ("Synonym", "Evidence 1"),
            ("Broader", "Evidence 2"),
            ("Synonym", "Evidence 3")  # Tie-breaker
        ]
        
        # Use a valid template string with placeholders
        valid_template = "A: {A}\nB: {B}\nLabel: "
        
        # Run duo vote
        label, evidence, conflict = _duo_vote(
            self.text_a, self.text_b, valid_template, self.config
        )
        
        # Verify results (should use the majority vote - Synonym)
        self.assertEqual(label, "Synonym")
        self.assertEqual(evidence, "Evidence 3")
        self.assertTrue(conflict)
        
        # Verify LLM was called 3 times (including tie-breaker)
        self.assertEqual(mock_call_llm.call_count, 3)
    
    @mock.patch('concord.pipeline._annotate_pair')
    def test_run_pair(self, mock_annotate):
        """Test run_pair function (high-level)."""
        # Setup mock to return a result
        mock_annotate.return_value = {
            "label": "Synonym",
            "similarity": 0.85,
            "evidence": "These are synonyms",
            "conflict": False
        }
        
        # Run pair annotation
        label, sim, evidence = run_pair(self.text_a, self.text_b, self.config_path)
        
        # Verify results
        self.assertEqual(label, "Synonym")
        self.assertEqual(sim, 0.85)
        self.assertEqual(evidence, "These are synonyms")
    
    @mock.patch('concord.pipeline._annotate_pair')
    @mock.patch('concord.pipeline._write')
    def test_run_file(self, mock_write, mock_annotate):
        """Test run_file function."""
        # Setup mock to return a result
        mock_annotate.return_value = {
            "label": "Synonym",
            "similarity": 0.85,
            "evidence": "These are synonyms",
            "conflict": False
        }
        
        # Run file annotation
        output_path = run_file(
            self.csv_path, 
            self.config_path,
            col_a="text_a",
            col_b="text_b"
        )
        
        # Verify _annotate_pair was called for each row
        self.assertEqual(mock_annotate.call_count, 3)
        
        # Verify _write was called for each row
        self.assertEqual(mock_write.call_count, 3)
        
        # Check that the output path ends with .llm.csv
        # The suffix method in pathlib only returns the last extension (.csv)
        # So we need to check the full path string ends with .llm.csv
        self.assertTrue(str(output_path).endswith(".llm.csv"))
    
if __name__ == '__main__':
    unittest.main()