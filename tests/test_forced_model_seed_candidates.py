from __future__ import annotations

import unittest

from roi_image_edit.forced_model_seeds import forced_model_seed_audit


class ForcedModelSeedCandidatesTest(unittest.TestCase):

    def test_convertible_model_suggestion_creates_forced_seed_candidate(self) -> None:
        model_records = [
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 1,
                "patch": {"opacity_delta": -0.02},
                "conversion_status": "converted",
            }
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[{"opacity_delta": -0.02}],
            seen_params=set(),
            stage_filter_report={"rejected_patches": []},
        )
        self.assertEqual(report["forced_seed_count"], 1)
        self.assertEqual(len(report["forced_seeds"]), 1)
        seed = report["forced_seeds"][0]
        self.assertTrue(seed["converted"])
        self.assertFalse(seed.get("deduped", False))
        self.assertEqual(seed["converted_patch"], {"opacity_delta": -0.02})

    def test_unconvertible_suggestion_is_audited_not_seeded(self) -> None:
        model_records = [
            {
                "source": "final_acceptance",
                "kind": "parameter_suggestion",
                "index": 1,
                "conversion_status": "unconvertible",
                "conversion_reason": "unsupported parameter",
            }
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[],
            seen_params=set(),
            stage_filter_report={"rejected_patches": []},
        )
        self.assertEqual(report["forced_seed_count"], 0)
        audited = report["audited_suggestions"]
        self.assertEqual(len(audited), 1)
        self.assertFalse(audited[0]["converted"])
        self.assertFalse(audited[0]["selectable"])
        self.assertIn("unsupported parameter", audited[0]["rejection_reason"])

    def test_filtered_model_seed_is_audited_with_rejection(self) -> None:
        model_records = [
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 1,
                "patch": {"mask_threshold_delta": 5},
                "conversion_status": "converted",
            }
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[],
            seen_params=set(),
            stage_filter_report={
                "rejected_patches": [{"mask_threshold_delta": 5}],
            },
        )
        self.assertEqual(report["forced_seed_count"], 0)
        audited = report["audited_suggestions"]
        self.assertEqual(len(audited), 1)
        self.assertIn("rejected by stage", audited[0]["rejection_reason"])

    def test_deduped_model_seed_is_audited(self) -> None:
        model_records = [
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 1,
                "patch": {"opacity_delta": -0.02},
                "conversion_status": "converted",
            },
            {
                "source": "final_acceptance",
                "kind": "parameter_suggestion",
                "index": 2,
                "patch": {"opacity_delta": -0.02},
                "conversion_status": "converted",
            },
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[{"opacity_delta": -0.02}],
            seen_params=set(),
            stage_filter_report={"rejected_patches": []},
        )
        self.assertGreaterEqual(report["forced_seed_count"], 1)
        audited = report["audited_suggestions"]
        deduped = [a for a in audited if a.get("deduped")]
        self.assertGreaterEqual(len(deduped), 0)

    def test_forced_seed_audit_records_all_fields(self) -> None:
        model_records = [
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 1,
                "patch": {"opacity_delta": -0.02},
                "conversion_status": "converted",
            }
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[{"opacity_delta": -0.02}],
            seen_params=set(),
            stage_filter_report={"rejected_patches": []},
        )
        seed = report["forced_seeds"][0]
        self.assertIn("source", seed)
        self.assertIn("kind", seed)
        self.assertIn("raw_suggestion", seed)
        self.assertIn("converted_patch", seed)
        self.assertIn("converted", seed)

    def test_non_dict_suggestion_is_audited_unconvertible(self) -> None:
        model_records = [
            {
                "source": "tuning",
                "kind": "parameter_suggestion",
                "index": 1,
                "conversion_status": "unconvertible",
                "conversion_reason": "parameter suggestion is not an object",
            }
        ]
        report = forced_model_seed_audit(
            model_records,
            [],
            set(),
            {"rejected_patches": []},
        )
        self.assertEqual(report["forced_seed_count"], 0)
        audited = report["audited_suggestions"]
        self.assertEqual(len(audited), 1)
        self.assertFalse(audited[0]["rendered"])

    def test_mixed_convertible_and_unconvertible_suggestions(self) -> None:
        model_records = [
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 1,
                "patch": {"opacity_delta": -0.02},
                "conversion_status": "converted",
            },
            {
                "source": "candidate_rank",
                "kind": "parameter_suggestion",
                "index": 2,
                "conversion_status": "unconvertible",
                "conversion_reason": "missing delta",
            },
            {
                "source": "final_acceptance",
                "kind": "suggested_patch",
                "patch": {"blur_delta": 0.04},
                "conversion_status": "converted",
            },
        ]
        report = forced_model_seed_audit(
            model_records,
            allowed_patches=[{"opacity_delta": -0.02}, {"blur_delta": 0.04}],
            seen_params=set(),
            stage_filter_report={"rejected_patches": []},
        )
        self.assertEqual(report["forced_seed_count"], 2)
        self.assertGreaterEqual(len(report["audited_suggestions"]), 1)
        unconvertible = [a for a in report["audited_suggestions"] if not a.get("converted")]
        self.assertEqual(len(unconvertible), 1)


if __name__ == "__main__":
    unittest.main()
