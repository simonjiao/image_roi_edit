from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = REPO_ROOT / "docs" / "workflow_checklist.md"


class CompletionClaimsTest(unittest.TestCase):
    def assert_no_unverified_completion_claim(self, path: Path, phrases: tuple[str, ...]) -> None:
        guarded_context_markers = (
            "do not",
            "not describe",
            "cannot",
            "must not",
            "unless",
            "until",
            "while",
            "forbidden",
            "禁止",
            "不能",
            "不可",
            "不再声称",
            "除非",
            "未满足",
            "未验证",
            "禁止当前完成态措辞",
        )
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            lowered = line.lower()
            for phrase in phrases:
                if phrase.lower() not in lowered:
                    continue
                if any(marker in lowered for marker in guarded_context_markers):
                    continue
                self.fail(
                    f"{path.relative_to(REPO_ROOT)}:{line_no} has an unverified "
                    f"completion claim: {phrase!r}"
                )

    def test_readme_declares_hardening_status_while_checklist_is_open(self) -> None:
        checklist = CHECKLIST.read_text(encoding="utf-8")
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

        if "- [ ]" in checklist:
            self.assertIn("Workflow hardening status is tracked", readme)
        else:
            self.assertNotIn("Workflow hardening status is tracked", readme)
        self.assertIn("workflow_checklist.md", readme)

    def test_public_docs_and_prompts_do_not_claim_unverified_completion(self) -> None:
        checked_paths = [
            REPO_ROOT / "README.md",
            *(REPO_ROOT / "docs").glob("*.md"),
            *(REPO_ROOT / "src" / "roi_image_edit" / "prompts").glob("*.txt"),
        ]
        forbidden_phrases = (
            "local flow is complete",
            "local-flow hardening is complete",
            "workflow is complete",
            "workflow hardening is complete",
            "all checklist items are closed",
            "本地流程完善完成",
            "本地流程已经完成",
            "workflow 已完善",
            "workflow 已经完成",
            "所有 checklist 项全部关闭",
            "所有设计目标已完成",
        )

        for path in checked_paths:
            with self.subTest(path=str(path.relative_to(REPO_ROOT))):
                self.assert_no_unverified_completion_claim(path, forbidden_phrases)


if __name__ == "__main__":
    unittest.main()
