from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from roi_image_edit.checklist_policy import checklist_closure_report


REPO_ROOT = Path(__file__).resolve().parents[1]


class ChecklistCommitPolicyTest(unittest.TestCase):
    def test_message_with_closed_checklist_items_passes(self) -> None:
        report = checklist_closure_report(
            "Guard public completion claims\n\n"
            "Closes checklist items 226 and 263: completion claims are tested."
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.item_numbers, (226, 263))
        self.assertEqual(report.reason, "references_closed_checklist_items")

    def test_generic_process_message_fails(self) -> None:
        report = checklist_closure_report("优化流程\n\nRefactor local validation.")

        self.assertFalse(report.passed)
        self.assertEqual(report.item_numbers, ())
        self.assertEqual(report.reason, "missing_closed_checklist_item_reference")

    def test_validate_message_script_fails_without_checklist_reference(self) -> None:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as message_file:
            message_file.write("Optimize flow\n\nNo checklist closure reference.")
            message_path = Path(message_file.name)
        self.addCleanup(message_path.unlink)

        result = subprocess.run(
            [
                str(REPO_ROOT / ".venv" / "bin" / "python"),
                str(REPO_ROOT / "scripts" / "validate_checklist_closure_message.py"),
                str(message_path),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing_closed_checklist_item_reference", result.stderr)

    def test_plain_feature_commit_is_not_a_checklist_closure_message(self) -> None:
        report = checklist_closure_report("Add Web canvas zoom controls\n\nFeature-only change.")

        self.assertFalse(report.passed)
        self.assertEqual(report.reason, "missing_closed_checklist_item_reference")


if __name__ == "__main__":
    unittest.main()
