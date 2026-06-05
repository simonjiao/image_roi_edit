from __future__ import annotations

from pathlib import Path
import unittest

from roi_image_edit.cli import build_process_summary
from roi_image_edit.roi_locator import parse_instruction_details


class InstructionParsingTest(unittest.TestCase):
    def assert_instruction_fields(
        self,
        instruction: str,
        *,
        field: str | None,
        old_value: str,
        new_value: str,
        source_explicit: bool,
    ) -> dict:
        details = parse_instruction_details(instruction)
        self.assertEqual(details["field"], field)
        self.assertEqual(details["field_key"], field)
        self.assertEqual(details["old_value"], old_value)
        self.assertEqual(details["source_text"], old_value)
        self.assertEqual(details["new_value"], new_value)
        self.assertEqual(details["target_text"], new_value)
        self.assertEqual(details["source_explicit"], source_explicit)
        self.assertIsInstance(details["confidence"], float)
        self.assertIsNone(details["failure_reason"])
        return details

    def test_parse_name_date_age_and_manual_roi_instruction_fields(self) -> None:
        name = self.assert_instruction_fields(
            "姓名赵真真修改为陈芸",
            field="name",
            old_value="赵真真",
            new_value="陈芸",
            source_explicit=True,
        )
        self.assertGreaterEqual(name["confidence"], 0.9)

        date = self.assert_instruction_fields(
            "日期2024-01-01改为2025-02-03",
            field="date",
            old_value="2024-01-01",
            new_value="2025-02-03",
            source_explicit=True,
        )
        self.assertGreaterEqual(date["confidence"], 0.9)

        age = self.assert_instruction_fields(
            "年龄18改为19",
            field="age",
            old_value="18",
            new_value="19",
            source_explicit=True,
        )
        self.assertGreaterEqual(age["confidence"], 0.9)

        manual = self.assert_instruction_fields(
            "陈芸",
            field=None,
            old_value="",
            new_value="陈芸",
            source_explicit=False,
        )
        self.assertGreaterEqual(manual["confidence"], 0.5)

    def test_empty_instruction_reports_failure_reason(self) -> None:
        details = parse_instruction_details("")
        self.assertEqual(details["field"], None)
        self.assertEqual(details["old_value"], "")
        self.assertEqual(details["new_value"], "")
        self.assertEqual(details["confidence"], 0.0)
        self.assertEqual(details["failure_reason"], "empty_instruction")

    def test_process_cli_json_summary_includes_instruction_details(self) -> None:
        instruction_details = parse_instruction_details("姓名甲修改为乙")
        summary = build_process_summary(
            {"runDir": "output/web/run1"},
            {
                "ok": True,
                "accepted": False,
                "applied": False,
                "instructionDetails": instruction_details,
                "artifacts": {"applied": "output/web/run1/applied.png"},
                "regions": [],
            },
            Path("output/result.png"),
        )
        self.assertEqual(summary["instruction_details"]["field"], "name")
        self.assertEqual(summary["instruction_details"]["old_value"], "甲")
        self.assertEqual(summary["instruction_details"]["new_value"], "乙")
        self.assertIn("confidence", summary["instruction_details"])
        self.assertIn("failure_reason", summary["instruction_details"])


if __name__ == "__main__":
    unittest.main()
