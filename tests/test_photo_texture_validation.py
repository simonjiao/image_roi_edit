from __future__ import annotations

import unittest

from roi_image_edit.local_validation import (
    PHOTO_TEXTURE_ISSUE_TYPES,
    local_photo_texture_issues,
    photo_texture_axes_report,
    stage_issue_severity,
)


def photo_report(
    *,
    blur: float = 0.1,
    edge_breakup: float = 0.0,
    photo_noise: float = 0.0,
    jpeg_weight: float = 0.0,
    edge_ratio: float = 1.0,
    residual_ratio: float = 1.0,
    old_residual: float = 3.0,
    old_edge: float = 32.0,
) -> dict:
    return {
        "pass": True,
        "pipeline_profile": "photo_scan",
        "photo_texture_metrics": {
            "enabled": True,
            "edge_laplacian_ratio": edge_ratio,
            "residual_ratio": residual_ratio,
            "old_residual_mean": old_residual,
            "old_edge_laplacian_mean": old_edge,
            "params": {
                "blur": blur,
                "photo_warp": 0.0,
                "edge_breakup": edge_breakup,
                "photo_noise": photo_noise,
                "jpeg_weight": jpeg_weight,
            },
        },
    }


class PhotoTextureValidationTest(unittest.TestCase):
    def issue_types(self, report: dict) -> set[str]:
        return {str(issue.get("type")) for issue in local_photo_texture_issues(report)}

    def test_photo_texture_issue_enum_covers_sharp_clean_blurry_and_missing_breakup(self) -> None:
        self.assertEqual(
            PHOTO_TEXTURE_ISSUE_TYPES,
            {
                "photo_texture_not_applied",
                "photo_texture_too_sharp",
                "photo_texture_too_clean",
                "photo_texture_too_blurry",
                "photo_texture_edge_breakup_missing",
            },
        )
        self.assertIn(
            "photo_texture_too_sharp",
            self.issue_types(photo_report(blur=0.1, edge_ratio=2.2, edge_breakup=0.02)),
        )
        self.assertIn(
            "photo_texture_too_clean",
            self.issue_types(photo_report(residual_ratio=0.2, photo_noise=0.0, jpeg_weight=0.0, edge_breakup=0.02)),
        )
        self.assertIn(
            "photo_texture_too_blurry",
            self.issue_types(photo_report(blur=0.8, edge_ratio=0.12, edge_breakup=0.02)),
        )
        self.assertIn(
            "photo_texture_edge_breakup_missing",
            self.issue_types(photo_report(edge_breakup=0.0, photo_noise=0.01)),
        )
        self.assertIn(
            "photo_texture_not_applied",
            self.issue_types(photo_report(blur=0.0, edge_breakup=0.0, photo_noise=0.0, jpeg_weight=0.0)),
        )

    def test_photo_texture_missing_breakup_contributes_stage_severity(self) -> None:
        report = photo_report(edge_breakup=0.0, photo_noise=0.01)
        report["local_photo_texture_issues"] = local_photo_texture_issues(report)
        self.assertGreater(stage_issue_severity(report, "photo_texture"), 0.0)

    def test_photo_texture_axes_report_records_texture_not_just_blur(self) -> None:
        metrics = photo_report(
            blur=0.2,
            edge_breakup=0.012,
            photo_noise=0.02,
            jpeg_weight=0.08,
            edge_ratio=1.4,
            residual_ratio=0.76,
        )["photo_texture_metrics"]
        axes = photo_texture_axes_report(metrics)
        self.assertEqual(axes["objective"], "match_source_photo_or_scan_texture")
        self.assertEqual(axes["sharpness"]["edge_laplacian_ratio"], 1.4)
        self.assertEqual(axes["breakup"]["edge_breakup"], 0.012)
        self.assertEqual(axes["noise"]["photo_noise"], 0.02)
        self.assertEqual(axes["compression"]["jpeg_weight"], 0.08)


if __name__ == "__main__":
    unittest.main()
