"""
Tests for the prompts module.
"""

import unittest
from unittest import mock

from concord.llm.prompts import (
    build_annotation_prompt,
    get_prompt_template,
    list_available_templates,
    save_template,
)
from concord.utils import validate_template


class TestPrompts(unittest.TestCase):
    """Test suite for prompt template functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock config
        self.cfg = {"engine": {"mode": "zero-shot"}, "prompt_ver": "v1.0"}

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

    def test_get_prompt_template_default(self):
        """Test getting default prompt template when no version specified."""
        cfg_no_version = {"engine": {"mode": "zero-shot"}}
        template = get_prompt_template(cfg_no_version)
        self.assertIn("A: {A}", template)
        self.assertIn("B: {B}", template)

    def test_get_prompt_template_invalid_version(self):
        """Test error handling for invalid template version."""
        with self.assertRaises(ValueError):
            get_prompt_template(self.cfg, ver="invalid_version")

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
        self.assertTrue(validate_template(template, raise_error=False))

    def test_validate_template_missing_placeholders(self):
        """Test template validation with missing placeholders."""
        template = "A: {A}\nMissing B placeholder\nClassify with these labels: Exact, Different"
        self.assertFalse(validate_template(template, raise_error=False))

    def test_validate_template_missing_both_placeholders(self):
        """Test template validation with both placeholders missing."""
        template = (
            "Missing both placeholders\nClassify with these labels: Exact, Different"
        )
        self.assertFalse(validate_template(template, raise_error=False))

    @mock.patch(
        "concord.llm.prompts._TEMPLATES",
        {"v1.0": "Template 1", "v2.0": "Template 2"},
    )
    def test_list_available_templates(self):
        """Test listing available templates."""
        templates = list_available_templates()
        self.assertEqual(set(templates), {"v1.0", "v2.0"})

    @mock.patch(
        "concord.llm.prompts._TEMPLATES",
        {"v1.0": "Template 1", "v2.0": "", "v3.0": "Template 3"},
    )
    def test_list_available_templates_excludes_empty(self):
        """Test that empty templates are excluded from the list."""
        templates = list_available_templates()
        self.assertEqual(set(templates), {"v1.0", "v3.0"})

    @mock.patch("concord.llm.prompts._TEMPLATE_DIR")
    @mock.patch("concord.utils.validate_template", return_value=True)
    @mock.patch("concord.llm.prompts._TEMPLATES", {})
    def test_save_template(self, mock_validate, mock_dir):
        """Test saving a template."""
        mock_dir.mkdir.return_value = None
        mock_open = mock.mock_open()

        with mock.patch("builtins.open", mock_open):
            result = save_template("v2.0-test", "Template content with {A} and {B}")

        self.assertTrue(result)
        mock_open.assert_called_once()
        mock_open().write.assert_called_once_with("Template content with {A} and {B}")

    @mock.patch("concord.utils.validate_template", return_value=False)
    def test_save_template_invalid(self, mock_validate):
        """Test saving an invalid template."""
        result = save_template("v2.0-test", "Invalid template")
        self.assertFalse(result)

    def test_backward_compatibility_bucket_pair_parameter(self):
        """Test that bucket_pair parameter is accepted but ignored for backward compatibility."""
        template = get_prompt_template(
            self.cfg, ver="v1.0", bucket_pair=("text1", "text2")
        )
        self.assertIn("A: {A}", template)
        self.assertIn("B: {B}", template)


if __name__ == "__main__":
    unittest.main()
