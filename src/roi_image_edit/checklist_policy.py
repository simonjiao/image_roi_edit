from __future__ import annotations

from dataclasses import dataclass
import re


_CHECKLIST_ITEM_RE = re.compile(r"checklist\s+items?\D+((?:\d+\D*)+)", re.IGNORECASE)
_CHINESE_CHECKLIST_ITEM_RE = re.compile(r"(?:关闭|引用|关联)\s*checklist\s*项\D*((?:\d+\D*)+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class ChecklistClosureReport:
    passed: bool
    item_numbers: tuple[int, ...]
    reason: str


def checklist_closure_report(message: str) -> ChecklistClosureReport:
    """Validate that a commit or PR description names closed checklist items."""
    item_numbers: list[int] = []
    for pattern in (_CHECKLIST_ITEM_RE, _CHINESE_CHECKLIST_ITEM_RE):
        for match in pattern.finditer(message):
            item_numbers.extend(int(value) for value in _NUMBER_RE.findall(match.group(1)))

    unique_items = tuple(sorted(set(item_numbers)))
    if unique_items:
        return ChecklistClosureReport(
            passed=True,
            item_numbers=unique_items,
            reason="references_closed_checklist_items",
        )
    return ChecklistClosureReport(
        passed=False,
        item_numbers=(),
        reason="missing_closed_checklist_item_reference",
    )
