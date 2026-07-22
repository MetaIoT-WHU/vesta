import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch

# Shared 9-class activity label order (id = index).
# 0 = Default (static); 6 = Circle; 7 = Swipe.
ACTIVITY_CLASS_NAMES: List[str] = [
    "Default",
    "Push hand",
    "Nod",
    "Turn head",
    "Touch",
    "Push twice",
    "Circle",
    "Swipe",
    "Pick up",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
