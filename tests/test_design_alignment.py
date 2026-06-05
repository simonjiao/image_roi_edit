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
        self.assertIn("旧 7 类诊断关注点必须说明与当前 5 stage 结构的关系", checklist)

    def test_text_shape_gap_table_syncs_status_to_checklist_items(self) -> None:
        doc = (ROOT / "docs" / "text_shape_joint_optimization_design.md").read_text(encoding="utf-8")
        match = re.search(r"## 现有流程差距\n(?P<body>.*?)\n## 分层联合优化设计", doc, re.DOTALL)
        self.assertIsNotNone(match)
        body = match.group("body") if match else ""
        rows = [
            line
            for line in body.splitlines()
            if line.startswith("| ") and not line.startswith("| ---") and "目标能力" not in line
        ]

        self.assertEqual(len(rows), 11)
        for row in rows:
            self.assertRegex(row, r"\| (已覆盖|部分覆盖|未完成)")
            self.assertIn("local_flow_hardening_checklist.md", row)
            if "部分覆盖" in row or "未完成" in row:
                self.assertIn("未完成：", row)
        expected_status = {
            "方向和目标字段联合选择": "部分覆盖",
            "搜索 ROI 与编辑 ROI 分离": "部分覆盖",
            "旧槽位完整性门禁": "部分覆盖",
            "同字数 CJK 放置": "部分覆盖",
            "单字形态变化检测": "部分覆盖",
            "字体形态搜索": "已覆盖",
            "姿态继承": "部分覆盖",
            "黑灰门禁": "已覆盖",
            "照片质感": "已覆盖",
            "背景处理": "未完成",
            "视觉模型": "已覆盖",
        }
        for ability, status in expected_status.items():
            matching = [row for row in rows if f"| {ability} |" in row]
            self.assertEqual(len(matching), 1, ability)
            self.assertIn(f"| {status}", matching[0])
        self.assertIn("[145-150](local_flow_hardening_checklist.md#k-背景处理拆分)", body)
        self.assertIn("[95-96](local_flow_hardening_checklist.md#f-放置策略选择)", body)


if __name__ == "__main__":
    unittest.main()
