"""
Tests for the prompts module.
"""

import unittest
from unittest import mock

from concord.llm.prompts import (BucketPrompt, _validate_template,
                                 build_annotation_prompt, choose_bucket,
                                 get_prompt_template, list_available_templates,
                                 save_template)


class TestPrompts(unittest.TestCase):
    """Test suite for prompt template functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config
        self.cfg = {"engine": {"mode": "llm"}, "prompt_ver": "v1.0"}

        # Test texts for bucket routing
        self.enzyme_text_a = "Glucose-6-phosphate isomerase (EC 5.3.1.9)"
        self.enzyme_text_b = "Phosphoglucose isomerase enzyme"

        self.phage_text_a = "Bacteriophage T4 tail protein"
        self.phage_text_b = "Phage capsid assembly protein"

        self.general_text_a = "Hypothetical protein"
        self.general_text_b = "Unknown function protein"

        # Reset mock responses
        BucketPrompt.reset_mocks()

    def test_bucket_routing_enzyme(self):
        """Test that enzyme texts route to the enzyme bucket."""
        bucket = choose_bucket(self.enzyme_text_a, self.enzyme_text_b)
        self.assertEqual(bucket, "v1.1-enzyme")

    def test_bucket_routing_phage(self):
        """Test that phage texts route to the phage bucket."""
        bucket = choose_bucket(self.phage_text_a, self.phage_text_b)
        self.assertEqual(bucket, "v1.1-phage")

    def test_bucket_routing_general(self):
        """Test that general texts route to the general bucket."""
        bucket = choose_bucket(self.general_text_a, self.general_text_b)
        self.assertEqual(bucket, "v1.1-general")

    def test_get_prompt_template_explicit_version(self):
        """Test getting prompt template with explicit version."""
        template = get_prompt_template(self.cfg, ver="v1.0")
        self.assertIn("A: {A}", template)
        self.assertIn("B: {B}", template)

    def test_get_prompt_template_from_config(self):
        """Test getting prompt template from config."""
        template = get_prompt_template(self.cfg)
        self.assertIn("A: {A}", template)
        self.assertIn("B: {B}", template)

    def test_bucket_routing_and_templates(self):
        """Test bucket routing and template retrieval separately without complex mocking."""
        # 1. Test bucket routing works correctly
        bucket = choose_bucket(self.enzyme_text_a, self.enzyme_text_b)
        self.assertEqual(bucket, "v1.1-enzyme")

        # 2. Test getting a template with explicit version works
        template = get_prompt_template(self.cfg, ver="v1.0")
        self.assertIn("A: {A}", template)
        self.assertIn("B: {B}", template)
        self.assertIn("Label", template)

    def test_build_annotation_prompt(self):
        """Test filling placeholders in template."""
        template = "A: {A}\nB: {B}\nAnalyze the relationship."
        filled = build_annotation_prompt("Text A", "Text B", template)

        self.assertEqual(filled, "A: Text A\nB: Text B\nAnalyze the relationship.")

    def test_build_annotation_prompt_error(self):
        """Test error handling in build_annotation_prompt."""
        template = "A: {A}\nMissing B placeholder\nAnalyze the relationship."

        with self.assertRaises(ValueError):
            build_annotation_prompt("Text A", "Text B", template)

    def test_validate_template_valid(self):
        """Test template validation with valid template."""
        template = "A: {A}\nB: {B}\nClassify with these labels: Exact, Different"
        self.assertTrue(_validate_template(template))

    def test_validate_template_missing_placeholders(self):
        """Test template validation with missing placeholders."""
        template = "A: {A}\nMissing B placeholder\nClassify with these labels: Exact, Different"
        self.assertFalse(_validate_template(template))

    def test_validate_template_missing_labels(self):
        """Test template validation with no mention of labels."""
        template = "A: {A}\nB: {B}\nCompare these entities."
        self.assertFalse(_validate_template(template))

    @mock.patch(
        "concord.llm.prompts._TEMPLATES",
        {"v1.0": "Template 1", "v1.1-test": "Template 2"},
    )
    def test_list_available_templates(self):
        """Test listing available templates."""
        templates = list_available_templates()
        self.assertEqual(set(templates), {"v1.0", "v1.1-test"})

    @mock.patch("concord.llm.prompts._TEMPLATE_DIR")
    @mock.patch("concord.llm.prompts._validate_template", return_value=True)
    @mock.patch("concord.llm.prompts._TEMPLATES", {})
    def test_save_template(self, mock_validate, mock_dir):
        """Test saving a template."""
        mock_dir.mkdir.return_value = None
        mock_open = mock.mock_open()

        with mock.patch("builtins.open", mock_open):
            result = save_template("v1.2-test", "Template content with {A} and {B}")

        self.assertTrue(result)
        mock_open.assert_called_once()
        mock_open().write.assert_called_once_with("Template content with {A} and {B}")

    @mock.patch("concord.llm.prompts._validate_template", return_value=False)
    def test_save_template_invalid(self, mock_validate):
        """Test saving an invalid template."""
        result = save_template("v1.2-test", "Invalid template")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
