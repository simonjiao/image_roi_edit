from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from roi_image_edit.iterative_pipeline import VisionClient
from roi_image_edit.vision_audit import write_vision_prompt_audit


class VisionPromptAuditTest(unittest.TestCase):
    def test_write_vision_prompt_audit_records_prompt_text_hashes_images_and_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "candidate.png"
            image_path.write_bytes(b"not-a-real-png-but-auditable")
            audit_path = tmp_path / "candidate_prompt_audit.json"

            audit = write_vision_prompt_audit(
                audit_path,
                prompt_name="candidate_rank_prompt.txt",
                system_prompt="system prompt",
                user_prompt=(
                    "field_key field_label_text field_separator_text protected_texts "
                    "source_text target_text search_roi target_roi slot_boxes protected_boxes "
                    "text_angle_degrees stage_context_by_candidate blocking_stage "
                    "stage_status pass_with_deferred deferred_issues deferred_to_stage "
                    "allowed_patch_keys blocked_patch_keys stage_filter_contract"
                ),
                image_paths=[image_path],
                model="mock-vision",
                response_json={"pass": True, "best_candidate": "c1"},
                elapsed_seconds=1.25,
            )

            saved = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, audit)
            self.assertEqual(saved["prompt_name"], "candidate_rank_prompt.txt")
            self.assertTrue(saved["request"]["input_presence"]["complete"])
            self.assertEqual(saved["request"]["images"][0]["path"], str(image_path))
            self.assertTrue(saved["request"]["images"][0]["sha256"])
            self.assertEqual(saved["response"]["top_level_keys"], ["best_candidate", "pass"])
            self.assertEqual(saved["transport"]["cache_hit"], False)
            self.assertEqual(saved["transport"]["elapsed_seconds"], 1.25)
            user_prompt_path = Path(saved["request"]["prompt_text_artifacts"]["user_prompt"])
            system_prompt_path = Path(saved["request"]["prompt_text_artifacts"]["system_prompt"])
            self.assertIn("stage_context_by_candidate", user_prompt_path.read_text(encoding="utf-8"))
            self.assertEqual(system_prompt_path.read_text(encoding="utf-8"), "system prompt")

    def test_vision_client_call_json_writes_audit_for_successful_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "context.png"
            image_path.write_bytes(b"image-bytes")
            audit_path = tmp_path / "final_prompt_audit.json"

            client = VisionClient.__new__(VisionClient)
            client.model = "mock-vision"
            client.base_url = "https://mock.example/v1"
            client.cache_enabled = False
            client._post_with_retry = lambda _payload: {"pass": True, "final_decision": "deliver"}  # type: ignore[method-assign]

            response = client.call_json(
                system_prompt="system prompt",
                user_prompt=(
                    "field_key field_label_text field_separator_text protected_texts "
                    "source_text target_text search_roi target_roi slot_boxes protected_boxes "
                    "text_angle_degrees stage_context blocking_stage stage_status "
                    "pass_with_deferred deferred_issues deferred_to_stage allowed_patch_keys "
                    "blocked_patch_keys profile_constraints final prompt"
                ),
                image_paths=[image_path],
                prompt_name="final_acceptance_prompt.txt",
                audit_path=audit_path,
            )

            self.assertEqual(response["final_decision"], "deliver")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["response"]["top_level_keys"], ["final_decision", "pass"])
            self.assertEqual(audit["model"], "mock-vision")
            self.assertEqual(audit["transport"]["fallback_without_response_format_or_temperature"], False)
            self.assertEqual(audit["transport"]["cache_hit"], False)
            self.assertIsInstance(audit["transport"]["elapsed_seconds"], float)
            self.assertTrue(Path(audit["request"]["prompt_text_artifacts"]["user_prompt"]).exists())

    def test_vision_client_call_json_reuses_identical_cached_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "context.png"
            image_path.write_bytes(b"image-bytes")
            first_audit_path = tmp_path / "first_prompt_audit.json"
            second_audit_path = tmp_path / "second_prompt_audit.json"
            post_count = {"value": 0}

            client = VisionClient.__new__(VisionClient)
            client.model = "mock-vision"
            client.base_url = "https://mock.example/v1"
            client.cache_enabled = True
            client.cache_dir = tmp_path / "vision-cache"

            def post_once(_payload: dict) -> dict:
                post_count["value"] += 1
                return {"pass": True, "final_decision": "deliver", "call": post_count["value"]}

            client._post_with_retry = post_once  # type: ignore[method-assign]
            kwargs = {
                "system_prompt": "system prompt",
                "user_prompt": (
                    "field_key field_label_text field_separator_text protected_texts "
                    "source_text target_text search_roi target_roi slot_boxes protected_boxes "
                    "text_angle_degrees stage_context blocking_stage stage_status "
                    "pass_with_deferred deferred_issues deferred_to_stage allowed_patch_keys "
                    "blocked_patch_keys profile_constraints final prompt"
                ),
                "image_paths": [image_path],
                "prompt_name": "final_acceptance_prompt.txt",
            }

            first = client.call_json(**kwargs, audit_path=first_audit_path)
            second = client.call_json(**kwargs, audit_path=second_audit_path)

            self.assertEqual(first, second)
            self.assertEqual(post_count["value"], 1)
            first_audit = json.loads(first_audit_path.read_text(encoding="utf-8"))
            second_audit = json.loads(second_audit_path.read_text(encoding="utf-8"))
            self.assertFalse(first_audit["transport"]["cache_hit"])
            self.assertTrue(second_audit["transport"]["cache_hit"])
            self.assertEqual(first_audit["transport"]["cache_key"], second_audit["transport"]["cache_key"])


if __name__ == "__main__":
    unittest.main()
