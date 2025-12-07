from typing_extensions import Literal
from pydantic import BaseModel


class DualPairsSchema(BaseModel):
    """
    Schema for comparing two pairs of entities simultaneously.
    
    The model should output:
    - pair_1_match: 1 if entity_1a and entity_1b match, else 0
    - pair_2_match: 1 if entity_2a and entity_2b match, else 0
    """
    pair_1_match: Literal[0, 1]
    pair_2_match: Literal[0, 1] 