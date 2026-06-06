# Section U Implementation Progress

## Overview

所有 Section U 项（U.1 - U.12）已全部实现，checklist 中再无 `[ ]` 未完成项。

## U.1-U.3: Near-threshold overblack micro tuning

**Files:**
- `src/roi_image_edit/revision_solver.py` — 核心实现
- `tests/test_ink_gray_micro_tuning.py` — 新增测试
- `tests/test_ink_gray_candidate_grid.py` — 微调候选保留测试

**Changes:**
- 新增 `_core_overblack_near_threshold()` — 检测 `roi_core_too_black` / `changed_char_core_too_black` 近阈值条件
- 新增 `_core_overblack_micro_variants()` — 生成对称过黑微调候选（降低 opacity/core_ink_gain/core_darken_strength）
- 重构 `ink_gray_near_threshold_micro_tuning()` — 先检查 core_light，再检查 overblack，统一返回
- 重构 `ink_gray_micro_tuning_candidates()` — 根据 candidate_family 分发到对应变体生成函数
- 新增 `_build_micro_tuning_report()` — 稳定报告 schema 包含 enabled/family/stage_id/metric/candidate_ids
- 微调候选独立于 axis-priority top-N 保留，prepend 到候选列表前面
- `overblack_micro_tuning_report` 添加到 ink_gray_candidate_grid 报告

## U.4-U.5: Forced model seed candidates

**Files:**
- `src/roi_image_edit/forced_model_seeds.py` — 新增模块
- `tests/test_forced_model_seed_candidates.py` — 新增测试

**Changes:**
- `forced_model_seed_audit()` — 将模型 suggested_patch/parameter_suggestions 转换为 forced seed 候选
- 不可转换的建议记录 `rejection_reason`
- 被 stage filter 拒绝的建议记录 `filter_rejection_reason`
- 重复建议标记 `deduped_to_key`
- 集成到 `region_processing.py` 的 revision loop，添加 "forced_model_seed" origin

## U.6-U.7: Candidate rejection table and progress diagnostics

**Files:**
- `src/roi_image_edit/region_processing.py` — `build_candidate_rejection_table()`
- `tests/test_revision_stop_diagnostics.py` — 新增测试

**Changes:**
- `build_candidate_rejection_table()` — 为 `no_selectable_revision_candidate` 生成完整拒绝表
- 每个候选记录 `candidate_id`/`origin`/`primary_stage`/`optimization_step`/`strict_pass`/`stage_pass`/`selectable`/`rejection_reason`
- Progress/result schema 现在包含 `micro_tuning_count`/`forced_model_seed_count`/`controlled_escape_count`/`candidate_rejection_count`

## U.8-U.9: Controlled cross-stage escape candidates

**Files:**
- `src/roi_image_edit/revision_solver.py` — `controlled_escape_candidate_grid()`
- `tests/test_controlled_escape_candidates.py` — 新增测试

**Changes:**
- 跨阶段逃逸只在当前阶段近阈值、hard boundary 通过、前序阶段不回退时启用
- `CONTROLLED_ESCAPE_LIMIT = 4` 限制逃逸候选数
- `controlled_escape=true`/`primary_stage`/`secondary_stage`/`allowed_secondary_delta_bounds` 标记
- `cross_stage_cartesian_disabled=true` 保持不变

## U.10-U.11: Complexity-normalized ink thresholds

**Files:**
- `src/roi_image_edit/complexity_normalization.py` — 新增模块
- `tests/test_complexity_normalized_ink_thresholds.py` — 新增测试

**Changes:**
- `text_complexity()` — CJK 字符复杂度估算
- `complexity_normalized_ink_limits()` — 计算字数/复杂度归一化后的黑灰阈值
- 归一化只影响黑灰质量阈值，不改变 `hard_boundary`/`protected_text`/`slot_quality`/`outside_roi`/`old_text_residual`

## U.12: Regression fixture Case E

**Files:**
- `tests/fixtures/regression_cases/case_e_cjk_longer_text_near_threshold_ink.json`

**Changes:**
- 通用 CJK 2→3 字近阈值过黑回归 fixture
- 不写死具体姓名或目标字
- 覆盖 micro tuning report/family、forced seed audit、candidate rejection table、complexity normalization

## Verification

```bash
.venv/bin/python -m unittest discover -s tests  # 263 个测试全部通过
```
