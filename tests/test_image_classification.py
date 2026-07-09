from __future__ import annotations

import unittest

from PIL import Image, ImageDraw

from roi_image_edit.image_classification import (
    classify_image_workflow,
    classify_region_roi_policy,
    with_region_roi_policy,
)
from roi_image_edit.roi_locator import parse_instruction_details


def text_like_image(size: tuple[int, int], *, lines: int = 1, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    for line in range(lines):
        y = 18 + line * 22
        draw.rectangle([12, y, 44, y + 9], fill=(30, 30, 30))
        draw.rectangle([58, y, 84, y + 9], fill=(30, 30, 30))
        draw.rectangle([98, y, 132, y + 9], fill=(30, 30, 30))
    return image


class ImageClassificationTest(unittest.TestCase):
    def test_photo_form_field_cjk_class_is_stable_for_same_kind(self) -> None:
        instruction = parse_instruction_details("姓名甲乙修改为丙丁")
        first = classify_image_workflow(text_like_image((260, 220), lines=2), instruction_details=instruction)
        second = classify_image_workflow(text_like_image((300, 220), lines=2), instruction_details=instruction)

        self.assertEqual(first["class_key"], "photo_document.form_field_value_replace.cjk")
        self.assertEqual(second["class_key"], first["class_key"])
        self.assertEqual(first["internal_profile"], "photo_scan")
        self.assertEqual(first["profile_source"], "classification")

    def test_different_scene_classes_do_not_collapse(self) -> None:
        photo = classify_image_workflow(
            text_like_image((260, 220), lines=2, color=(238, 236, 232)),
            instruction_details=parse_instruction_details("姓名甲乙修改为丙丁"),
        )
        clean_number = classify_image_workflow(
            text_like_image((320, 220), lines=1),
            instruction_details=parse_instruction_details("编号12修改为34"),
        )
        low_res = classify_image_workflow(
            text_like_image((120, 70), lines=1),
            instruction_details=parse_instruction_details("文字甲乙修改为丙丁"),
        )
        dense = classify_image_workflow(
            text_like_image((320, 240), lines=5),
            instruction_details=parse_instruction_details("甲乙修改为丙丁"),
        )

        class_keys = {photo["class_key"], clean_number["class_key"], low_res["class_key"], dense["class_key"]}
        self.assertEqual(len(class_keys), 4)
        self.assertEqual(clean_number["class_key"], "clean_digital.numeric_or_date_replace")
        self.assertEqual(clean_number["internal_profile"], "clean_digital")
        self.assertEqual(clean_number["prompt_pack"], "clean_numeric_or_date_replace")
        self.assertEqual(clean_number["parameter_family"], "clean_digital_no_photo_texture")
        self.assertTrue(low_res["class_key"].startswith("low_res_thumbnail."))
        self.assertTrue(low_res["prompt_pack"].startswith("low_res_"))
        self.assertEqual(low_res["parameter_family"], "low_res_magnified_conservative")
        self.assertTrue(dense["class_key"].startswith("photo_document.dense_paragraph_replace."))

    def test_manual_roi_policy_is_part_of_classification_and_region_plan(self) -> None:
        image = text_like_image((240, 140), lines=1)
        manual_anchor = classify_image_workflow(
            image,
            instruction_details=parse_instruction_details("姓名甲乙修改为丙丁"),
            regions=[{"id": "r1", "rect": {"x": 0, "y": 0, "w": 160, "h": 60}}],
        )
        manual_exact = classify_image_workflow(
            image,
            instruction_details=parse_instruction_details("新文字"),
            regions=[{"id": "r1", "rect": {"x": 12, "y": 18, "w": 40, "h": 20}}],
        )

        self.assertEqual(manual_anchor["roi_policy"], "manual_anchor")
        self.assertEqual(manual_anchor["internal_profile"], "photo_scan")
        self.assertEqual(manual_exact["roi_policy"], "manual_exact")
        self.assertEqual(manual_exact["internal_profile"], "manual_roi_quick")

        region_policy = classify_region_roi_policy(
            image_classification=manual_anchor,
            search_roi=(0, 0, 160, 60),
            edit_roi=(20, 18, 68, 34),
            source_text="甲乙",
        )
        region_classification = with_region_roi_policy(manual_anchor, roi_policy=region_policy)
        self.assertEqual(region_policy, "manual_anchor")
        self.assertEqual(region_classification["profile_source"], "classification")

    def test_anchored_text_removal_gets_dedicated_class(self) -> None:
        result = classify_image_workflow(
            text_like_image((360, 260), lines=3, color=(238, 236, 232)),
            instruction_details=parse_instruction_details("将图片中的提示区下面的甲乙丙丁四个字抹除"),
        )

        self.assertEqual(result["operation"], "remove_text")
        self.assertEqual(result["scenario"], "anchored_text_removal")
        self.assertEqual(result["class_key"], "photo_document.anchored_text_removal.cjk")
        self.assertEqual(result["length_change"], "removed")
        self.assertEqual(result["prompt_pack"], "anchored_text_removal")
        self.assertEqual(result["parameter_family"], "photo_document_text_removal")

    def test_text_redaction_gets_dedicated_class(self) -> None:
        result = classify_image_workflow(
            text_like_image((360, 260), lines=3, color=(238, 236, 232)),
            instruction_details=parse_instruction_details("将 Tim 打码"),
            regions=[{"id": "r1", "rect": {"x": 220, "y": 32, "w": 42, "h": 24}}],
        )

        self.assertEqual(result["operation"], "redact_text")
        self.assertEqual(result["scenario"], "text_redaction")
        self.assertEqual(result["class_key"], "photo_document.text_redaction.latin")
        self.assertEqual(result["length_change"], "redacted")
        self.assertEqual(result["prompt_pack"], "text_redaction")
        self.assertEqual(result["parameter_family"], "photo_document_text_redaction")

    def test_amount_replacement_gets_dedicated_class(self) -> None:
        result = classify_image_workflow(
            text_like_image((591, 1280), lines=8),
            instruction_details=parse_instruction_details("金额+9764修改为+12749"),
            regions=[{"id": "amount", "rect": {"x": 428, "y": 306, "w": 84, "h": 26}}],
        )

        self.assertEqual(result["operation"], "replace_text")
        self.assertEqual(result["scenario"], "amount_value_replace")
        self.assertEqual(result["script"], "numeric_or_date")
        self.assertEqual(result["class_key"], "photo_document.amount_value_replace.numeric_or_date")
        self.assertEqual(result["prompt_pack"], "amount_value_replace")
        self.assertEqual(result["parameter_family"], "clean_digital_amount_value_replace")
        self.assertEqual(result["internal_profile"], "clean_digital")


if __name__ == "__main__":
    unittest.main()
