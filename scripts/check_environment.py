#!/usr/bin/env python3
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roi_image_edit.cli import main


if __name__ == "__main__":
    sys.argv.insert(1, "check-env")
    main()
