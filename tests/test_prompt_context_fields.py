from __future__ import annotations

import unittest

from roi_image_edit.iterative_pipeline import RenderPlan, TextRun, vision_task_context
from roi_image_edit.region_processing import processing_prompt_context
from roi_image_edit.roi_locator import parse_instruction_details


def field_plan() -> RenderPlan:
    return RenderPlan(
        target_text="25岁",
        source_text="24岁",
        search_roi=(10, 20, 90, 42),
        target_roi=(42, 22, 78, 40),
        slot_boxes=(TextRun(42, 22, 70, 40, 120),),
        protected_boxes=((12, 22, 36, 40), (37, 24, 40, 38)),
        source_reference_box=(42, 22, 78, 40),
        style_reference_box=None,
        style_reference_text=None,
        draw_mode="auto",
        field_key="age",
        field_label_text="年龄",
        field_separator_text="：",
        protected_texts=("年龄", "："),
    )


class PromptContextFieldsTest(unittest.TestCase):
    def test_instruction_details_include_field_label_separator_and_protected_texts(self) -> None:
        details = parse_instruction_details("年龄：24岁修改为25岁")

        self.assertEqual(details["field_key"], "age")
        self.assertEqual(details["field_label_text"], "年龄")
        self.assertEqual(details["field_separator_text"], "：")
        self.assertEqual(details["protected_texts"], ["年龄", "："])

    def test_processing_prompt_context_embeds_dynamic_field_context(self) -> None:
        context = processing_prompt_context(field_plan())

        self.assertIn("field_key: age", context)
        self.assertIn("field_label_text: 年龄", context)
        self.assertIn("field_separator_text: ：", context)
        self.assertIn("protected_texts: ['年龄', '：']", context)
        self.assertIn("source_text 所在字段值区域", context)
        self.assertNotIn("姓名区域", context)
        self.assertNotIn("名：", context)

    def test_iterative_vision_context_embeds_dynamic_field_context(self) -> None:
        context = vision_task_context(field_plan())

        self.assertIn("field_key: age", context)
        self.assertIn("field_label_text: 年龄", context)
        self.assertIn("field_separator_text: ：", context)
        self.assertIn("protected_texts: ['年龄', '：']", context)
        self.assertNotIn("固定姓名", context)
        self.assertNotIn("名：", context)


if __name__ == "__main__":
    unittest.main()
