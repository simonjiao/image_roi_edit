from __future__ import annotations

from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATHS = (
    PROJECT_ROOT / "src" / "roi_image_edit",
)
FORBIDDEN_RUNTIME_FRAGMENTS = (
    "赵芳",
    "陈慧",
    "赵真真",
    "陈芸",
    "563177",
    "562177",
    ".pic.jpg",
    "本图",
    "这个字",
    "目标字符必须右倾",
    "目标字符必须左倾",
    "名：",
    "姓名区域",
    "名前",
    "名后",
    'default="名"',
    'or "名"',
)
FORBIDDEN_PROMPT_FRAGMENTS = (
    "左侧“",
    "打印宋体",
    "仿宋",
    "宋体感",
    "0.76",
    "0.72",
    '"24"',
    '"25"',
    '"26"',
    '"27"',
    '"28"',
)


class NoHardcodedSpecialCasesTest(unittest.TestCase):
    def test_runtime_code_and_prompts_do_not_encode_specific_names_images_or_target_char_rules(self) -> None:
        violations: list[str] = []
        for root in RUNTIME_PATHS:
            for path in root.rglob("*"):
                if path.suffix not in {".py", ".txt", ".md"}:
                    continue
                text = path.read_text(encoding="utf-8")
                for fragment in FORBIDDEN_RUNTIME_FRAGMENTS:
                    if fragment in text:
                        violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {fragment}")

        self.assertEqual(violations, [])

    def test_packaged_prompts_do_not_anchor_to_single_field_or_fixed_candidate_values(self) -> None:
        violations: list[str] = []
        prompt_root = PROJECT_ROOT / "src" / "roi_image_edit" / "prompts"
        for path in prompt_root.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            for fragment in FORBIDDEN_PROMPT_FRAGMENTS:
                if fragment in text:
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} contains {fragment}")

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
