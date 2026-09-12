"""
Tests for EntityMatchingProcessor — ID correctness, column layout, and prompt output.
Uses in-memory CSVs so no real data files are needed.
"""

import io
import pytest
import pandas as pd

from src.data_processing.entity_matching_processor import EntityMatchingProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENTITY1_CSV = """id|name|city
10|Alice|London
20|Bob|Paris
30|Carol|Berlin
"""

ENTITY2_CSV = """id|name|price
100|Widget|9.99
200|Gadget|19.99
300|Doohickey|4.99
"""

PAIRS_CSV = """id1,id2
10,100
20,200
30,300
10,200
"""


def _make_processor(monkeypatch, d1_csv=ENTITY1_CSV, d2_csv=ENTITY2_CSV, pairs_csv=PAIRS_CSV):
    """Build an EntityMatchingProcessor from in-memory CSV strings."""
    from src.evaluator.data_manipulation import load_csv_with_separator_detection

    def fake_load(path, **kwargs):
        data = {
            "d1": d1_csv,
            "d2": d2_csv,
            "pairs": pairs_csv,
        }
        sep = "|" if "|" in data[path] else ","
        return pd.read_csv(io.StringIO(data[path]), sep=sep)

    monkeypatch.setattr(
        "src.data_processing.entity_matching_processor.load_csv_with_separator_detection",
        fake_load,
    )

    return EntityMatchingProcessor("d1", "d2", "pairs")


# ---------------------------------------------------------------------------
# ID correctness
# ---------------------------------------------------------------------------

class TestIDCorrectness:
    def test_datasets_indexed_by_id(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        assert proc.dataset1_df.index.name == "id"
        assert proc.dataset2_df.index.name == "id"

    def test_entity_lookup_by_id_not_position(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        # ID 20 is at row position 1; make sure we get "Bob", not "Alice"
        entity = proc.dataset1_df.loc[20]
        assert entity["name"] == "Bob"

    def test_pair_returns_correct_ids(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(0)
        assert result["id1_index"] == 10
        assert result["id2_index"] == 100

    def test_second_pair_correct_ids(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(1)
        assert result["id1_index"] == 20
        assert result["id2_index"] == 200

    def test_cross_pair_ids(self, monkeypatch):
        # pair index 3: id1=10, id2=200 (cross-pair, not same-row entities)
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(3)
        assert result["id1_index"] == 10
        assert result["id2_index"] == 200

    def test_cross_pair_entities_match_ids(self, monkeypatch):
        # id1=10 is Alice, id2=200 is Gadget — verify entity data matches IDs
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(3)
        assert result["entity1_raw"]["name"] == "Alice"
        assert result["entity2_raw"]["name"] == "Gadget"

    def test_out_of_range_raises(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        with pytest.raises(IndexError):
            proc.get_pair_by_index(999)


# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------

class TestColumnLayout:
    def test_col_names_from_pairs_file(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        assert proc.col1_name == "id1"
        assert proc.col2_name == "id2"

    def test_id_column_not_in_entity_fields(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(0)
        assert "id" not in result["entity1_raw"]
        assert "id" not in result["entity2_raw"]

    def test_pair_result_has_expected_keys(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(0)
        assert "pair_index" in result
        assert "id1_index" in result
        assert "id2_index" in result
        assert "entity1_formatted" in result
        assert "entity2_formatted" in result
        assert "entity1_raw" in result
        assert "entity2_raw" in result

    def test_entity1_raw_has_entity_columns(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(0)
        assert "name" in result["entity1_raw"]
        assert "city" in result["entity1_raw"]

    def test_entity2_raw_has_entity_columns(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        result = proc.get_pair_by_index(0)
        assert "name" in result["entity2_raw"]
        assert "price" in result["entity2_raw"]

    def test_pairs_df_length(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        assert len(proc.pairs_df) == 4


# ---------------------------------------------------------------------------
# Prompt output
# ---------------------------------------------------------------------------

class TestPromptOutput:
    def test_prompt_contains_entity_values(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        template = "Entity A: {entity_1}. Entity B: {entity_2}. Match?"
        result = proc.generate_prompt(0, template=template)
        assert "Alice" in result["prompt"]
        assert "Widget" in result["prompt"]

    def test_prompt_placeholders_replaced(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        template = "Entity A: {entity_1}. Entity B: {entity_2}."
        result = proc.generate_prompt(0, template=template)
        assert "{entity_1}" not in result["prompt"]
        assert "{entity_2}" not in result["prompt"]

    def test_prompt_metadata_has_ids(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        template = "{entity_1} vs {entity_2}"
        result = proc.generate_prompt(0, template=template)
        assert result["metadata"]["id1_index"] == 10
        assert result["metadata"]["id2_index"] == 100

    def test_batch_prompt_count(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        template = "{entity_1} vs {entity_2}"
        prompts = proc.batch_generate_prompts(start_index=0, count=3, template=template)
        assert len(prompts) == 3

    def test_format_entity_excludes_id(self, monkeypatch):
        proc = _make_processor(monkeypatch)
        entity = proc.dataset1_df.loc[10]
        formatted = proc.format_entity(entity)
        assert "Alice" in formatted
        # 'id' should not appear as a field label
        assert "Id:" not in formatted

    def test_format_entity_skips_empty_values(self, monkeypatch):
        # Add entity with empty field
        d1 = "id|name|city\n10|Alice|\n"
        proc = _make_processor(monkeypatch, d1_csv=d1)
        entity = proc.dataset1_df.loc[10]
        formatted = proc.format_entity(entity)
        assert "Alice" in formatted
        assert "City" not in formatted
