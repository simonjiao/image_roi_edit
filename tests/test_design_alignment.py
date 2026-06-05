from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STAGE_ORDER = (
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup",
)
OLD_CONCERNS = (
    "slot_alignment",
    "font_structure",
    "pose_geometry",
    "stroke_body",
    "tone_gray",
    "edge_quality",
    "photo_texture",
)


class DesignAlignmentTest(unittest.TestCase):
    def test_staged_design_declares_five_stage_gate_and_old_concern_mapping(self) -> None:
        doc = (ROOT / "docs" / "staged_roi_pipeline_design.md").read_text(encoding="utf-8")
        self.assertIn("当前代码和 checklist 的阶段门禁只有 5 个", doc)
        self.assertIn("不能再作为 `stage_id`、`blocking_stage` 或公开 gate 输出", doc)
        for stage_id in EXPECTED_STAGE_ORDER:
            self.assertIn(f'"{stage_id}"', doc)
        for concern in OLD_CONCERNS:
            self.assertIn(f"| `{concern}` |", doc)

    def test_staged_design_report_json_example_is_valid(self) -> None:
        doc = (ROOT / "docs" / "staged_roi_pipeline_design.md").read_text(encoding="utf-8")
        match = re.search(r"报告中新增：\n\n```json\n(?P<body>.*?)\n```", doc, re.DOTALL)
        self.assertIsNotNone(match)
        body = match.group("body") if match else ""
        payload = json.loads(body)
        self.assertEqual(tuple(payload["stage_order"]), EXPECTED_STAGE_ORDER)
        self.assertEqual(payload["blocking_stage"], "text_shape")
        self.assertTrue(payload["blocking_stage_blocks_next"])

    def test_checklist_keeps_design_sync_as_a_closeable_goal(self) -> None:
        checklist = (ROOT / "docs" / "local_flow_hardening_checklist.md").read_text(encoding="utf-8")
        self.assertIn("staged_roi_pipeline_design.md", checklist)
        self.assertIn("旧 7 阶段设计必须说明与当前 5 stage 结构的关系", checklist)


if __name__ == "__main__":
    unittest.main()
