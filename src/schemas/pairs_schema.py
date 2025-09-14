from typing_extensions import Literal
from pydantic import BaseModel


class PairsSchema(BaseModel):
    """
    Schema for binary entity matching.
    
    The model should output:
    - match: 1 if the two entities refer to the same thing, else 0
    """
    match: Literal[0, 1] 