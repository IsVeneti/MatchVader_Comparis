"""
Tests for candidate_processor — group loading, ID assignment, match/no-match output columns.
"""

import io
import logging
import pytest
import pandas as pd
from unittest.mock import MagicMock

from src.processing.candidate_processor import load_candidate_groups


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pairs_df(csv_str):
    return pd.read_csv(io.StringIO(csv_str))


PAIRS_D1_TARGET = """id1,id2
10,100
10,200
20,300
30,100
"""

PAIRS_D2_TARGET = """id1,id2
100,10
200,10
300,20
"""


# ---------------------------------------------------------------------------
# load_candidate_groups
# ---------------------------------------------------------------------------

class TestLoadCandidateGroups:
    def test_d1_target_groups_by_id1(self):
        pairs_df = _pairs_df(PAIRS_D1_TARGET)
        groups, target_col, candidate_col = load_candidate_groups(pairs_df, "d1", logger)
        assert target_col == "id1"
        assert candidate_col == "id2"
        assert set(groups.keys()) == {10, 20, 30}

    def test_d1_target_candidates_correct(self):
        pairs_df = _pairs_df(PAIRS_D1_TARGET)
        groups, _, _ = load_candidate_groups(pairs_df, "d1", logger)
        assert set(groups[10]) == {100, 200}
        assert groups[20] == [300]

    def test_d2_target_groups_by_id2(self):
        pairs_df = _pairs_df(PAIRS_D2_TARGET)
        groups, target_col, candidate_col = load_candidate_groups(pairs_df, "d2", logger)
        assert target_col == "id2"
        assert candidate_col == "id1"
        assert set(groups.keys()) == {10, 20}

    def test_d2_target_candidates_correct(self):
        pairs_df = _pairs_df(PAIRS_D2_TARGET)
        groups, _, _ = load_candidate_groups(pairs_df, "d2", logger)
        assert set(groups[10]) == {100, 200}

    def test_single_candidate_per_target(self):
        pairs_df = _pairs_df("id1,id2\n5,99\n")
        groups, _, _ = load_candidate_groups(pairs_df, "d1", logger)
        assert groups[5] == [99]

    def test_group_count_matches_unique_targets(self):
        pairs_df = _pairs_df(PAIRS_D1_TARGET)
        groups, _, _ = load_candidate_groups(pairs_df, "d1", logger)
        assert len(groups) == 3


# ---------------------------------------------------------------------------
# Result DataFrame columns (via process_candidate_selection with mocked LLM)
# ---------------------------------------------------------------------------

class TestCandidateSelectionResultColumns:
    def _run(self, target_side="d1"):
        from src.processing.candidate_processor import process_candidate_selection

        # Minimal pairs_with_ids: target 10 has candidates 100, 200
        pairs_df = _pairs_df("id1,id2\n10,100\n10,200\n")

        d1 = pd.DataFrame({"name": ["Alice"]}, index=pd.Index([10], name="id"))
        d2 = pd.DataFrame({"name": ["Widget", "Gadget"]}, index=pd.Index([100, 200], name="id"))

        processor = MagicMock()
        processor.pairs_df = pairs_df
        processor.dataset1_df = d1
        processor.dataset2_df = d2
        processor.col1_name = "id1"
        processor.col2_name = "id2"

        schema_response = MagicMock()
        schema_response.selected_candidate = 1  # selects first candidate
        schema_response.model_dump.return_value = {"selected_candidate": 1}

        llm = MagicMock()
        llm.generate_structured.return_value = schema_response

        task_config = {"target_side": target_side}

        return process_candidate_selection(
            llm, processor, "[target_entity] [candidates]",
            MagicMock(), task_config, logger,
        )

    def test_result_has_expected_columns(self):
        df = self._run()
        for col in ("success", "target_id", "candidate_id", "match",
                    "selected_candidate", "pair_index"):
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_equals_candidate_count(self):
        df = self._run()
        assert len(df) == 2  # 2 candidates for the 1 target

    def test_selected_candidate_gets_match_1(self):
        df = self._run()
        # selected_candidate=1 → first candidate (id=100) gets match=1
        row = df[df["candidate_id"] == 100].iloc[0]
        assert row["match"] == 1

    def test_non_selected_candidate_gets_match_0(self):
        df = self._run()
        row = df[df["candidate_id"] == 200].iloc[0]
        assert row["match"] == 0

    def test_target_id_correct_for_all_rows(self):
        df = self._run()
        assert (df["target_id"] == 10).all()

    def test_all_rows_succeed(self):
        df = self._run()
        assert df["success"].all()
