from __future__ import annotations

from pathlib import Path
from typing import List


def load_prompt_file(path: str | Path) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
