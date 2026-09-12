"""
Tests for data_manipulation — merge correctness, column naming, and GT loading.
"""

import io
import pytest
import pandas as pd

from src.evaluator.data_manipulation import (
    load_ground_truth,
    merge_dataframes,
    merge_response_pairs,
    merge_response_pairs_partial,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(csv_str, sep=","):
    return pd.read_csv(io.StringIO(csv_str), sep=sep)


# ---------------------------------------------------------------------------
# load_ground_truth
# ---------------------------------------------------------------------------

class TestLoadGroundTruth:
    def test_adds_match_column(self, monkeypatch, tmp_path):
        gt_file = tmp_path / "gt.csv"
        gt_file.write_text("D1,D2\n1,10\n2,20\n")
        df = load_ground_truth(str(gt_file))
        assert "match" in df.columns
        assert (df["match"] == 1).all()

    def test_match_is_all_ones(self, monkeypatch, tmp_path):
        gt_file = tmp_path / "gt.csv"
        gt_file.write_text("D1,D2\n5,50\n6,60\n7,70\n")
        df = load_ground_truth(str(gt_file))
        assert list(df["match"]) == [1, 1, 1]


# ---------------------------------------------------------------------------
# merge_dataframes
# ---------------------------------------------------------------------------

class TestMergeDataframes:
    def _pairs(self):
        return _df("id1,id2,match\n1,10,1\n2,20,0\n3,30,1\n")

    def _gt(self):
        df = _df("id1,id2\n1,10\n3,30\n")
        df["match"] = 1
        return df

    def test_inner_join_row_count(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="inner")
        assert len(merged) == 2

    def test_left_join_keeps_all_pairs(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="left")
        assert len(merged) == 3

    def test_suffixes_are_pairs_and_gt(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="inner")
        assert "match_pairs" in merged.columns
        assert "match_gt" in merged.columns

    def test_non_gt_pair_gets_zero_gt(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="left")
        non_gt_row = merged[merged["id1"] == 2].iloc[0]
        assert non_gt_row["match_gt"] == 0

    def test_gt_pair_has_correct_gt_value(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="inner")
        row = merged[merged["id1"] == 1].iloc[0]
        assert row["match_gt"] == 1

    def test_invalid_join_type_raises(self):
        with pytest.raises(ValueError):
            merge_dataframes(self._pairs(), self._gt(), join_type="cross")

    def test_output_has_id1_and_id2(self):
        merged = merge_dataframes(self._pairs(), self._gt(), join_type="inner")
        assert "id1" in merged.columns
        assert "id2" in merged.columns


# ---------------------------------------------------------------------------
# merge_response_pairs
# ---------------------------------------------------------------------------

class TestMergeResponsePairs:
    def test_adds_match_column(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,match\n0,1\n1,0\n2,1\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n20,200\n30,300\n")

        df = merge_response_pairs(str(resp), str(pairs))
        assert "match" in df.columns
        assert list(df["match"]) == [1, 0, 1]

    def test_raises_on_missing_match_column(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,response\n0,yes\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n")

        with pytest.raises(ValueError, match="match"):
            merge_response_pairs(str(resp), str(pairs))

    def test_raises_on_row_count_mismatch(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,match\n0,1\n1,0\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n20,200\n30,300\n")

        with pytest.raises(ValueError):
            merge_response_pairs(str(resp), str(pairs))


# ---------------------------------------------------------------------------
# merge_response_pairs_partial
# ---------------------------------------------------------------------------

class TestMergeResponsePairsPartial:
    def test_keeps_only_evaluated_pairs(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,match\n0,1\n2,0\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n20,200\n30,300\n")

        df = merge_response_pairs_partial(str(resp), str(pairs))
        assert len(df) == 2
        assert set(df["id1"]) == {10, 30}

    def test_match_values_aligned_to_pair_index(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,match\n2,1\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n20,200\n30,300\n")

        df = merge_response_pairs_partial(str(resp), str(pairs))
        assert df.iloc[0]["match"] == 1
        assert df.iloc[0]["id1"] == 30

    def test_raises_on_missing_match_column(self, tmp_path):
        resp = tmp_path / "results.csv"
        resp.write_text("pair_index,response\n0,yes\n")
        pairs = tmp_path / "pairs.csv"
        pairs.write_text("id1,id2\n10,100\n")

        with pytest.raises(ValueError, match="match"):
            merge_response_pairs_partial(str(resp), str(pairs))
