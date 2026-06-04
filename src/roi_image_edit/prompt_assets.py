from __future__ import annotations

from importlib import resources
from typing import Iterable


PROMPT_NAMES = (
    "master_prompt.txt",
    "candidate_rank_prompt.txt",
    "tuning_prompt.txt",
    "font_size_prompt.txt",
    "darkness_blur_prompt.txt",
    "final_acceptance_prompt.txt",
)


def prompt_resource(name: str):
    return resources.files("roi_image_edit").joinpath("prompts", name)


def prompt_exists(name: str) -> bool:
    return prompt_resource(name).is_file()


def missing_prompt_names(names: Iterable[str] = PROMPT_NAMES) -> list[str]:
    return [name for name in names if not prompt_exists(name)]


def require_prompts(names: Iterable[str] = PROMPT_NAMES) -> None:
    missing = missing_prompt_names(names)
    if missing:
        raise FileNotFoundError(
            "package prompt assets are missing: " + ", ".join(missing)
        )


def load_prompt(name: str) -> str:
    require_prompts((name,))
    return prompt_resource(name).read_text(encoding="utf-8")
