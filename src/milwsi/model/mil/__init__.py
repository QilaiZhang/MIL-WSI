import importlib
from .base_model import build_model


mil_models = [
    "abmil",
    "clam",
    "transmil",
]
mil_modules = [importlib.import_module(f'milwsi.model.mil.{mil_model}') for mil_model in mil_models]
