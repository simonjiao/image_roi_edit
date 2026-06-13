from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = REPO_ROOT / "docs" / "workflow_checklist.md"
PATH_RE = re.compile(r"`((?:tests|scripts|docs|src)/[^`]+?)`")


class ChecklistIntegrityTest(unittest.TestCase):
    def checked_lines(self) -> list[tuple[int, str]]:
        return [
            (line_no, line)
            for line_no, line in enumerate(CHECKLIST.read_text(encoding="utf-8").splitlines(), start=1)
            if line.startswith("- [x]") or line.startswith("- [X]")
        ]

    def test_every_checked_item_has_current_evidence_marker(self) -> None:
        missing = [
            f"{line_no}:{line}"
            for line_no, line in self.checked_lines()
            if "证据：" not in line
        ]

        self.assertEqual(missing, [])

    def test_checked_item_evidence_paths_exist(self) -> None:
        missing_paths: list[str] = []
        for line_no, line in self.checked_lines():
            for ref in PATH_RE.findall(line):
                path_text = ref.split("::", 1)[0]
                if "*" in path_text:
                    if not list(REPO_ROOT.glob(path_text)):
                        missing_paths.append(f"{line_no}:{ref}")
                    continue
                candidate = REPO_ROOT / path_text
                if not candidate.exists():
                    missing_paths.append(f"{line_no}:{ref}")

        self.assertEqual(missing_paths, [])

    def test_checked_items_use_current_verification_command_or_stable_artifact(self) -> None:
        weak_evidence: list[str] = []
        stable_markers = (
            ".venv/bin/python -m unittest discover -s tests",
            "scripts/",
            "docs/",
            "src/",
            "output/",
            "result.json",
            "progress.jsonl",
        )
        for line_no, line in self.checked_lines():
            evidence = line.split("证据：", 1)[1]
            if not any(marker in evidence for marker in stable_markers):
                weak_evidence.append(f"{line_no}:{line}")

        self.assertEqual(weak_evidence, [])

    def test_completion_gate_matches_open_item_count(self) -> None:
        lines = CHECKLIST.read_text(encoding="utf-8").splitlines()
        open_lines = [
            f"{line_no}:{line}"
            for line_no, line in enumerate(lines, start=1)
            if line.startswith("- [ ]")
        ]
        completion_lines = [
            line
            for line in lines
            if "设计目标转换情况" in line and "全部关闭" in line
        ]
        self.assertEqual(len(completion_lines), 1)
        completion_checked = completion_lines[0].startswith("- [x]") or completion_lines[0].startswith("- [X]")
        if completion_checked:
            self.assertEqual(open_lines, [])
        else:
            self.assertNotEqual(open_lines, [])


if __name__ == "__main__":
    unittest.main()
