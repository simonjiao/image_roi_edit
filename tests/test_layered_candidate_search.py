from __future__ import annotations

import inspect
import unittest

import roi_image_edit.processing_service as processing_service
from roi_image_edit.revision_solver import layered_candidate_search_report


def grid_report(stage_id: str, raw: int, retained: int, pruned: int) -> dict:
    return {
        "enabled": True,
        "stage_id": stage_id,
        "optimization_step": stage_id,
        "candidate_count": retained,
        "budget": {
            "raw_candidate_budget": raw,
            "retained_count": retained,
            "pruned_count": pruned,
            "within_budget": True,
        },
    }


class LayeredCandidateSearchTest(unittest.TestCase):
    def test_layered_report_records_stage_budgets_without_cross_stage_cartesian_total(self) -> None:
        report = layered_candidate_search_report(
            grid_report("text_shape", 432, 48, 384),
            grid_report("ink_gray_balance", 128, 16, 112),
            grid_report("photo_texture", 108, 6, 102),
        )

        self.assertEqual(report["strategy"], "layered_stage_search")
        self.assertFalse(report["cross_stage_cartesian_search"])
        self.assertEqual(
            report["stage_order"],
            ["text_shape", "ink_gray_balance", "photo_texture"],
        )
        self.assertEqual(report["raw_candidate_budget_by_stage"]["text_shape"], 432)
        self.assertEqual(report["raw_candidate_budget_by_stage"]["ink_gray_balance"], 128)
        self.assertEqual(report["raw_candidate_budget_by_stage"]["photo_texture"], 108)
        self.assertEqual(report["retained_count_by_stage"]["text_shape"], 48)
        self.assertEqual(report["pruned_count_by_stage"]["photo_texture"], 102)
        self.assertNotIn("total_raw_candidate_budget", report)
        self.assertNotIn("cross_stage_cartesian_budget", report)
        self.assertNotIn("single_full_combination_total", report)

    def test_processing_service_writes_layered_report_in_revision_rounds(self) -> None:
        source = inspect.getsource(processing_service.run_region_vision_checks)
        self.assertIn("layered_candidate_search_report", source)
        self.assertIn('"layered_candidate_search": layered_search_report', source)


if __name__ == "__main__":
    unittest.main()
