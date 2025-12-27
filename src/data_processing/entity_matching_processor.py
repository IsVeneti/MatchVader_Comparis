import pandas as pd
import json

from src.evaluator.data_manipulation import load_csv_with_separator_detection

class EntityMatchingProcessor:
    """
    A class to process entity matching data from entity datasets
    and prepare them for LLM prompts.
    """
    
    def __init__(self, dataset1_path, dataset2_path, pairs_path):
        """
        Initialize with file paths.
        
        Args:
            dataset1_path: Path to first dataset (e.g., rest1clean.csv or abtclean.csv)
            dataset2_path: Path to second dataset (e.g., rest2clean.csv or buyclean.csv)
            pairs_path: Path to pairs file
        """
        self.dataset1_path = dataset1_path
        self.dataset2_path = dataset2_path
        self.pairs_path = pairs_path
        
        # Load the data
        self._load_data()
    
    def _load_data(self):
        """Load and parse the CSV files."""
        # Load datasets with pipe delimiter
        self.dataset1_df = load_csv_with_separator_detection(self.dataset1_path)
        self.dataset2_df = load_csv_with_separator_detection(self.dataset2_path)
        
        # Load pairs dataset
        self.pairs_df = load_csv_with_separator_detection(self.pairs_path)
        
        # Get column names dynamically from pairs file
        self.col1_name, self.col2_name = self.pairs_df.columns[0], self.pairs_df.columns[1]
        
        # Keep original column names from the CSV files - don't hardcode them!
        # Strip whitespace from column names
        self.dataset1_df.columns = self.dataset1_df.columns.str.strip()
        self.dataset2_df.columns = self.dataset2_df.columns.str.strip()
        
        print(f"Loaded {len(self.dataset1_df)} entities from {self.col1_name}")
        print(f"  Columns: {list(self.dataset1_df.columns)}")
        print(f"Loaded {len(self.dataset2_df)} entities from {self.col2_name}") 
        print(f"  Columns: {list(self.dataset2_df.columns)}")
        print(f"Loaded {len(self.pairs_df)} pairs")
    
    def format_entity(self, entity_data):
        """
        Format an entity for the LLM prompt.
        
        Args:
            entity_data: Row from dataframe containing entity info
            
        Returns:
            Formatted string representation of the entity
        """
        parts = []
        
        # Dynamically format all columns (except 'id' if present)
        for col in entity_data.index:
            if col.lower() == 'id':
                continue  # Skip ID field in the formatted output
            
            value = str(entity_data[col]).strip() if pd.notna(entity_data[col]) else ''
            if value:
                # Capitalize column name for display
                col_display = col.replace('_', ' ').title()
                parts.append(f"{col_display}: {value}")
        
        return ", ".join(parts) if parts else "No information available"
    
    def get_pair_by_index(self, pair_index):
        """
        Get a specific pair by its index in the pairs file.
        
        Args:
            pair_index: Index of the pair in the pairs dataframe
            
        Returns:
            Dictionary with formatted entities and metadata
        """
        if pair_index >= len(self.pairs_df):
            raise IndexError(f"Pair index {pair_index} out of range")
        
        pair = self.pairs_df.iloc[pair_index]
        
        # Use dynamic column names instead of hardcoded ones
        dataset1_idx = pair[self.col1_name]
        dataset2_idx = pair[self.col2_name]
        
        # Get entities by their indices
        entity1 = self.dataset1_df.iloc[dataset1_idx]
        entity2 = self.dataset2_df.iloc[dataset2_idx]
        
        return {
            'pair_index': pair_index,
            f'{self.col1_name}_index': dataset1_idx,
            f'{self.col2_name}_index': dataset2_idx,
            'entity1_raw': entity1.to_dict(),
            'entity2_raw': entity2.to_dict(),
            'entity1_formatted': self.format_entity(entity1),
            'entity2_formatted': self.format_entity(entity2)
        }
    
    def generate_prompt(self, pair_index, template=None):
        """
        Generate a formatted prompt for entity matching.
        
        Args:
            pair_index: Index of the pair to use
            template: Custom template string. Uses default if None.
            
        Returns:
            Formatted prompt string
        """
        if template is None:
            template = ("Are those two entities the same?\n"
                       "Entity 1: {entity_1}, Entity 2: {entity_2}.\n"
                       "Please answer in json, {{\"match\":1}} for match or {{\"match\":0}} for no match.")
        
        pair_data = self.get_pair_by_index(pair_index)
        
        # Format the prompt using .format()
        prompt = template.format(
            entity_1=pair_data['entity1_formatted'],
            entity_2=pair_data['entity2_formatted']
        )
        
        return {
            'prompt': prompt,
            'metadata': {
                'pair_index': pair_index,
                f'{self.col1_name}_index': pair_data[f'{self.col1_name}_index'],
                f'{self.col2_name}_index': pair_data[f'{self.col2_name}_index']
            }
        }
    
    def batch_generate_prompts(self, start_index=0, count=10, template=None):
        """
        Generate multiple prompts for batch processing.
        
        Args:
            start_index: Starting pair index
            count: Number of prompts to generate
            template: Custom template string
            
        Returns:
            List of prompt dictionaries
        """
        prompts = []
        end_index = min(start_index + count, len(self.pairs_df))
        
        for i in range(start_index, end_index):
            try:
                prompt_data = self.generate_prompt(i, template)
                prompts.append(prompt_data)
            except Exception as e:
                print(f"Error generating prompt for pair {i}: {e}")
                continue
        
        return prompts
    
    def _convert_to_json_serializable(self, obj):
        """
        Convert pandas/numpy types to JSON serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif pd.isna(obj):  # pandas NaN
            return None
        else:
            return obj

    def export_formatted_pairs(self, output_path, start_index=0, count=None):
        """
        Export formatted pairs to a JSON file for easy review.
        
        Args:
            output_path: Path to save the JSON file
            start_index: Starting pair index
            count: Number of pairs to export (None for all)
        """
        if count is None:
            count = len(self.pairs_df) - start_index
        
        pairs_data = []
        end_index = min(start_index + count, len(self.pairs_df))
        
        for i in range(start_index, end_index):
            try:
                pair_data = self.get_pair_by_index(i)
                # Convert to JSON serializable format
                pair_data_serializable = self._convert_to_json_serializable(pair_data)
                pairs_data.append(pair_data_serializable)
            except Exception as e:
                print(f"Error processing pair {i}: {e}")
                continue
        
        with open(output_path, 'w') as f:
            json.dump(pairs_data, f, indent=2)
        
        print(f"Exported {len(pairs_data)} formatted pairs to {output_path}")


# Example usage
if __name__ == "__main__":
    # Works with restaurant data
    # processor = EntityMatchingProcessor(
    #     dataset1_path='data/rest1clean.csv',
    #     dataset2_path='data/rest2clean.csv', 
    #     pairs_path='data/d1_pairs.csv'
    # )
    
    # Also works with abt-buy data
    processor = EntityMatchingProcessor(
        dataset1_path='data/d2_data/abtclean.csv',
        dataset2_path='data/d2_data/buyclean.csv', 
        pairs_path='data/d2_data/d2_pairs.csv'
    )
    
    # Generate a single prompt
    prompt_result = processor.generate_prompt(0)
    print("Sample prompt:")
    print(prompt_result['prompt'])
    print(f"Metadata: {prompt_result['metadata']}")
    
    # Generate batch prompts
    batch_prompts = processor.batch_generate_prompts(start_index=0, count=5)
    print(f"\nGenerated {len(batch_prompts)} batch prompts")
    
    # Export formatted pairs for review
    processor.export_formatted_pairs('formatted_pairs_sample.json', count=10)