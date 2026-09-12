"""
Tests for prompt_utils — placeholder replacement and repetition.
"""

import pytest
from src.processing.prompt_utils import replace_prompt_entities, apply_prompt_repetition


class TestReplacePromptEntities:
    def test_bracket_placeholder(self):
        result = replace_prompt_entities("Hello [name]!", {"name": "Alice"})
        assert result == "Hello Alice!"

    def test_double_brace_placeholder(self):
        result = replace_prompt_entities("Hello {{name}}!", {"name": "Bob"})
        assert result == "Hello Bob!"

    def test_multiple_placeholders(self):
        result = replace_prompt_entities("[e1] vs [e2]", {"e1": "A", "e2": "B"})
        assert result == "A vs B"

    def test_missing_placeholder_left_intact(self):
        result = replace_prompt_entities("[e1] vs [e2]", {"e1": "A"})
        assert "[e2]" in result

    def test_empty_values(self):
        result = replace_prompt_entities("[e1]", {"e1": ""})
        assert result == ""

    def test_no_substitution_needed(self):
        template = "No placeholders here."
        result = replace_prompt_entities(template, {"e1": "X"})
        assert result == template


class TestApplyPromptRepetition:
    def test_zero_repetitions_unchanged(self):
        prompt = "Match these entities."
        assert apply_prompt_repetition(prompt, 0) == prompt

    def test_negative_repetitions_unchanged(self):
        prompt = "Match these entities."
        assert apply_prompt_repetition(prompt, -1) == prompt

    def test_one_repetition_basic(self):
        prompt = "Match."
        result = apply_prompt_repetition(prompt, 1, style="basic")
        assert result == "Match. Match."

    def test_two_repetitions_basic(self):
        prompt = "X"
        result = apply_prompt_repetition(prompt, 2, style="basic")
        parts = result.split(" ")
        assert len(parts) == 3
        assert all(p == "X" for p in parts)

    def test_one_repetition_verbose_has_bridge(self):
        prompt = "Match."
        result = apply_prompt_repetition(prompt, 1, style="verbose")
        assert "Let me repeat" in result
        assert "Match." in result

    def test_repetition_count_matches(self):
        prompt = "Q"
        result = apply_prompt_repetition(prompt, 3, style="basic")
        # original + 3 copies = 4 occurrences
        assert result.count("Q") == 4
