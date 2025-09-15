from typing import Any
from dataclasses import dataclass

# =============================
# Dependencies (DI-friendly)
# =============================
@dataclass
class ModelDeps:
    clf: Any
    regressors: Any
    preprocessor: Any
    static_data: Any