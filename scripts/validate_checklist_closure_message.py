#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from roi_image_edit.checklist_policy import checklist_closure_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that a commit or PR message references closed checklist items."
    )
    parser.add_argument("message_file", type=Path, help="Path to the commit or PR message file.")
    args = parser.parse_args(argv)

    message = args.message_file.read_text(encoding="utf-8")
    report = checklist_closure_report(message)
    if report.passed:
        print(
            "checklist closure reference ok: "
            + ", ".join(str(item) for item in report.item_numbers)
        )
        return 0
    print(report.reason, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
