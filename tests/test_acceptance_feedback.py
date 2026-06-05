from __future__ import annotations

import unittest

from roi_image_edit.acceptance_feedback import acceptance_reports_background_patch
from roi_image_edit.local_validation import constraint_reason


class AcceptanceFeedbackTest(unittest.TestCase):
    def test_background_patch_feedback_is_shared_by_validation_and_patchers(self) -> None:
        acceptance = {
            "visual_findings": {"background": "patch_visible"},
            "must_fix": [{"issue": "背景有补丁感和涂抹残影"}],
        }

        self.assertTrue(acceptance_reports_background_patch(acceptance))
        self.assertEqual(
            constraint_reason({"pass": True, "pipeline_profile": "photo_scan"}, acceptance),
            "vision_background_patch_feedback_caps",
        )


if __name__ == "__main__":
    unittest.main()
