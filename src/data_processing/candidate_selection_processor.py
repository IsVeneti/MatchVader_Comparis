import pandas as pd
from pathlib import Path


class CandidateSelectionProcessor:
    """
    Processor for candidate selection from existing pairs file.
    Can group by either d1 or d2 as the target entity.
    """
    
    def __init__(self, d1_path: str, d2_path: str, pairs_path: str, target_side: str = "d2"):
        """
        Args:
            d1_path: Path to d1 entities CSV
            d2_path: Path to d2 entities CSV
            pairs_path: Path to pairs CSV with columns like [d1_index, d2_index]
            target_side: Which side is the target entity - either "d1" or "d2"
                        "d2": d2 is target, d1 entities are candidates (default)
                        "d1": d1 is target, d2 entities are candidates
        """
        self.d1_df = pd.read_csv(d1_path)
        self.d2_df = pd.read_csv(d2_path)
        self.pairs_df = pd.read_csv(pairs_path)
        
        # Validate target_side
        if target_side not in ["d1", "d2"]:
            raise ValueError(f"target_side must be 'd1' or 'd2', got: {target_side}")
        
        self.target_side = target_side
        
        # Detect column names (assuming format like "d1_index" and "d2_index" or similar)
        pairs_cols = self.pairs_df.columns.tolist()
        self.col1_name = pairs_cols[0]  # e.g., "d1_index" or "rest1_index"
        self.col2_name = pairs_cols[1]  # e.g., "d2_index" or "rest2_index"
        
        # Set target and candidate configuration based on target_side
        if target_side == "d2":
            self.target_col = self.col2_name
            self.candidate_col = self.col1_name
            self.target_df = self.d2_df
            self.candidate_df = self.d1_df
        else:  # target_side == "d1"
            self.target_col = self.col1_name
            self.candidate_col = self.col2_name
            self.target_df = self.d1_df
            self.candidate_df = self.d2_df
        
        # Group pairs by target entity
        self.grouped_pairs = self.pairs_df.groupby(self.target_col)
        
        # Get unique target IDs
        self.target_ids = self.pairs_df[self.target_col].unique()
        
        print(f"Loaded {len(self.pairs_df)} pairs")
        print(f"Found {len(self.target_ids)} unique target entities")
        print(f"Target side: {target_side}")
        print(f"  Target column: {self.target_col}")
        print(f"  Candidate column: {self.candidate_col}")
    
    def get_target_count(self) -> int:
        """Return number of unique target entities."""
        return len(self.target_ids)
    
    def get_target_by_index(self, index: int) -> dict:
        """
        Get target entity and all its candidate pairs by index.
        
        Returns:
            dict with:
                - target_id: ID of target entity
                - target_entity: dict of target entity attributes
                - candidates: list of dicts, each with:
                    - candidate_id: ID of candidate
                    - candidate_num: 1-based position
                    - entity: dict of candidate attributes
                    - pair_index: original index in pairs file
        """
        target_id = self.target_ids[index]
        
        # Get target entity
        target_row = self.target_df[self.target_df['id'] == target_id].iloc[0]
        target_entity = target_row.to_dict()
        
        # Get all candidate pairs for this target
        candidate_pairs = self.grouped_pairs.get_group(target_id)
        
        candidates = []
        for idx, (pair_idx, pair_row) in enumerate(candidate_pairs.iterrows(), start=1):
            candidate_id = pair_row[self.candidate_col]
            
            # Get candidate entity
            cand_entity_row = self.candidate_df[self.candidate_df['id'] == candidate_id].iloc[0]
            
            candidates.append({
                'candidate_id': candidate_id,
                'candidate_num': idx,  # 1-based index
                'entity': cand_entity_row.to_dict(),
                'pair_index': pair_idx  # Original index in pairs_df
            })
        
        return {
            'target_id': target_id,
            'target_entity': target_entity,
            'candidates': candidates,
            'candidate_count': len(candidates)
        }