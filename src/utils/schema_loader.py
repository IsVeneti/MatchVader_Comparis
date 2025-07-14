from pydantic import BaseModel
import importlib


def load_schema_class(dotted_path: str) -> type[BaseModel]:
    """
    Dynamically imports a Pydantic model from a dotted path.

    Args:
        dotted_path (str): Import path like "schemas.pairs_schema.PairsSchema"

    Returns:
        A subclass of pydantic.BaseModel

    Raises:
        ImportError: If module or class not found or invalid.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        schema_class = getattr(module, class_name)
        if not issubclass(schema_class, BaseModel):
            raise TypeError("Schema must be a subclass of pydantic.BaseModel")
        return schema_class
    except Exception as e:
        raise ImportError(f"Failed to load schema class from '{dotted_path}': {e}")
