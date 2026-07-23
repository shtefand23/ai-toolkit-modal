"""Patch convrot_quant.py to fix PyTorch 2.6+ infer_schema compatibility."""
import re
from pathlib import Path

TARGET = Path("/root/ai-toolkit/toolkit/util/convrot_quant.py")

if TARGET.exists():
    text = TARGET.read_text(encoding="utf-8")
    # Add typing import if missing
    if "from typing import List" not in text and "import typing" not in text:
        text = "from typing import List\n" + text
    # Fix lowercase list[torch.Tensor] -> List[torch.Tensor]
    text = text.replace("list[torch.Tensor]", "List[torch.Tensor]")
    TARGET.write_text(text, encoding="utf-8")
    print("Patched convrot_quant.py successfully")
else:
    print(f"File not found: {TARGET}")
