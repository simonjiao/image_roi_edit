# ROI image text replacement

中文版说明见 [`README.zh-CN.md`](README.zh-CN.md).

License: [`GPL-3.0-or-later`](LICENSE).

## Workflow

1. Use the packaged prompt assets and project pipeline code.
2. Generate local PIL/OpenCV ROI candidates.
3. Write hard-check reports for size, ROI-external pixels, border pixels, and
   protected text boxes.
4. Ask the configured vision model to rank candidates using the configured
   prompt.
5. Apply only small parameter patches.
6. Iterate up to `--max-iterations`.
7. Run final hard-check plus the final-acceptance prompt.

The implementation rules and accumulated review criteria are documented in
[`docs/roi_text_replacement_rules.md`](docs/roi_text_replacement_rules.md).
When changing the ROI pipeline, update that file alongside the code if the
workflow, hard gates, iteration strategy, or visual acceptance criteria change.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Prompt text files are packaged under
[`src/roi_image_edit/prompts/`](src/roi_image_edit/prompts/). CLI, Web, and
environment checks load prompts from that package resource directory only.

## CLI

Check dependencies, packaged prompt files, API config, and font availability:

```bash
.venv/bin/python scripts/roi_image_edit_cli.py check-env
```

Install the fonts that can be installed automatically:

```bash
.venv/bin/python scripts/roi_image_edit_cli.py install-fonts
```

Run the pipeline:

```bash
.venv/bin/python scripts/roi_image_edit_cli.py run \
  --metadata /path/to/metadata.json \
  --vision auto \
  --acceptance-mode strict \
  --max-iterations 8 \
  --max-candidates 12
```

Scan all loadable fonts under the default project, user, and system font
directories before ranking:

```bash
.venv/bin/python scripts/roi_image_edit_cli.py run \
  --metadata /path/to/metadata.json \
  --vision auto \
  --font-source scan \
  --font-candidate-pool-size 12 \
  --max-candidates 48 \
  --sheet-scale 6 \
  --sheet-cols 4
```

Start the local web UI:

```bash
.venv/bin/python scripts/roi_image_edit_web.py --host 127.0.0.1 --port 8787
```

The web page supports multiple uploaded images. Draw one or more rectangles on
the left image, enter a replacement instruction such as `旧文字替换为新文字`, then
click `处理全部`. The right pane shows the edited image. Use `>>>` to open the
candidate drawer; it shows up to five local candidate previews for the image.
Each web run is saved under `output/web/<run_id>/` with `request.json`,
`result.json`, the original image, and the final image. When a rectangle is
larger than the text itself, the web pipeline first shrinks the edit target to
the detected source-text components inside the rectangle. Web processing also
runs the vision candidate-ranking and final-acceptance prompts; per-region
visual artifacts are written under `output/web/<run_id>/regions/<region_id>/`.

Progress is printed to stderr during the run and also written to
`output/<run_id>/progress.jsonl`, so another terminal can follow it with:

```bash
tail -f output/<run_id>/progress.jsonl
```

In strict mode, the pipeline checks both grayscale coverage and font style. The
font style gate renders the original old text ROI, when source text is known,
with each local candidate font and uses the protected label `名` only as a
secondary style reference. It also reports a font category and penalizes modern sans fonts
such as Microsoft YaHei when Song/Ming or CJK serif candidates are available.
The strict gate also checks core stroke darkness and gray-edge similarity with
`--max-core-mean-gray-delta`, `--max-edge-mean-gray-delta`, and
`--min-dark-pixel-ratio`, so a candidate that is the right height but visibly
too light or too hard-edged is rejected before visual-model acceptance. The
default minimum dark-pixel coverage is now `0.88` to avoid candidates that have
similar mean gray but too little stroke coverage. Directional lightness limits,
`--max-core-lighten-delta` and `--max-edge-lighten-delta`, prevent a visually
acceptable but still pale candidate from replacing a closer local metric match.
The strict gate also checks per-character center placement and center distance
with `--max-char-center-dx` and `--max-char-center-distance-delta`, so one
character cannot drift sideways until the spacing no longer matches the source
text; the default per-character horizontal center limit is 2px for these small
fixtures. The visual model can only override the local fallback within
`--max-model-local-score-delta` and cannot pick a candidate whose old-text font
style ratio is worse than the local fallback by more than
`--max-model-font-style-ratio-delta`. The script still renders and hard-checks
all generated candidates, but `--vision-candidate-limit` sends only the
locally ranked top candidates to the visual candidate-ranking prompt by
default. This keeps the stage prompt small enough for OpenAI-compatible
vision gateways while preserving the full local hard-check report. The local
scorer also applies a small
blur preference with `--blur-score-weight` after `--blur-score-free-margin`, so
a candidate that is otherwise comparable but less fuzzy can beat an overly soft
render. `--ink-gain` darkens glyph alpha before blur, `--core-ink-gain`
darkens only the high-alpha stroke center after blur, `--core-darken-strength`
darkens already-rendered high-confidence core pixels without expanding low-alpha
antialias edges, `--alpha-contrast` tightens the alpha transition after blur,
and `--stroke-opacity` adds a fractional outer stroke. The local scorer compares
the real black core (`<55`/`<70`), the deep-gray core band (`55-70`), the
medium-gray stroke interior (`70-90`), the broader dark body (`<90`/`<120`),
and the 120-165 outer gray band. The hard report also includes per-character
gray bands so a replacement cannot hide one weak target character behind
acceptable whole-ROI totals. Too many `55-70` or `70-90` pixels make the strokes look gray even when
the total `<70` count is acceptable, while too many 120-165 pixels read as a
hazy outline. It now prefers converting gray stroke interiors into true black
core pixels instead of gaining darkness from gray edge pixels, while the hard
gates still keep total dark-pixel area bounded. Model tuning patches are
rejected when they pass visually but worsen these local true-black and gray-band
metrics. This is only a scoring pass; the final image is still edited only
inside the target ROI.

## Fonts

The script checks fonts in the same order the source document recommends for
candidate generation. Project-local Windows fonts can be placed under `fonts/`:

- `fonts/simsun.ttc`
- `fonts/simfang.ttf`
- `fonts/msyh.ttc`

Font binaries under `fonts/` are local runtime assets and are ignored by Git.
Keep licensed fonts on the local machine; do not commit proprietary font files
unless their redistribution license is confirmed.

On this macOS environment, `PingFangUI.ttc` is registered by the system, but
Pillow reports it as `unknown file format`, so the pipeline excludes it from the
usable render order until a Pillow-loadable PingFang file is provided.

By default `run` uses `--font-ranking style`, which reorders the usable fonts by
the old-text ROI style score for the current image when source text is known.
Use `--font-ranking document` to keep the document order. Use `--source-text`
when metadata does not provide the old text.

Use `--font-source scan` to scan `fonts/`, `~/Library/Fonts`, `/Library/Fonts`,
`/System/Library/Fonts`, and `/System/Library/Fonts/Supplemental`. Scanned fonts
must be Pillow-loadable and must render every required character from the target
text, source text, and style reference text. Fonts that fall back to a missing
glyph box for any required character are written to `rejected_font_candidates` in
`summary.json`. Increase `--font-candidate-pool-size` when using scan mode so
more ranked fonts can enter candidate generation.

## Run the fixture pipeline

```bash
.venv/bin/python scripts/run_iterative_roi.py \
  --metadata /path/to/metadata.json \
  --vision auto \
  --max-iterations 8 \
  --max-candidates 12
```

The script reads API settings from:

```text
.env
```

Use `--vision off` to run only local candidate generation and hard validation.

## Outputs

Each run writes a timestamped folder under `output/`, including:

- `original_crop.png`
- `iteration_*/contact_sheet.png`
- `iteration_*/hard_check_report.json`
- `iteration_*/visual_eval_candidate_rank.json`
- `iteration_*/visual_eval_tuning.json`
- `final_crop.png`
- `final_full.png`
- `final_acceptance.json`
- `final_acceptance_strict.json`
- `progress.jsonl`
- `summary.json`
