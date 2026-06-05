# Prompt Assets

This package directory contains the runtime prompt assets used by CLI and web
processing. These files are edited as project assets and are loaded from this
package resource directory only.

## Files

- `master_prompt.txt`: shared visual-evaluation system prompt.
- `candidate_rank_prompt.txt`: candidate ranking prompt.
- `tuning_prompt.txt`: single-round tuning diagnosis prompt.
- `font_size_prompt.txt`: font and size diagnosis prompt.
- `darkness_blur_prompt.txt`: darkness and blur diagnosis prompt.
- `final_acceptance_prompt.txt`: final acceptance prompt.

## Stage Participation

Visual prompts are not local stages. They evaluate locally generated top
candidates and return JSON suggestions; local hard checks and ordered stage
gates remain authoritative.

| Prompt | Role |
| --- | --- |
| `master_prompt.txt` | System prompt shared by visual ranking, tuning, and final acceptance calls. |
| `candidate_rank_prompt.txt` | Ranks locally generated top candidates with `stage_context_by_candidate`; suggestions must stay within each candidate's allowed patch keys. |
| `tuning_prompt.txt` | CLI iterative path diagnosis prompt; returns small JSON parameter suggestions for the current blocking stage. |
| `final_acceptance_prompt.txt` | Final visual acceptance prompt; must respect local hard check, strict gate, stage context, and local stage filter. |
| `font_size_prompt.txt` | Packaged specialist prompt asset for font and size diagnosis. |
| `darkness_blur_prompt.txt` | Packaged specialist prompt asset for darkness and blur diagnosis. |
