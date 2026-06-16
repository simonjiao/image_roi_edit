# Staged ROI Text Replacement Pipeline Design

本文档描述下一版 ROI 文字替换流程的阶段化设计。目标不是继续给现有逻辑叠补丁，而是把“发现问题、阻塞后续阶段、生成候选、选择参数、验收交付”拆成可验证、可重排、可扩展的阶段系统。

## 设计目标

1. 不再让颜色、灰边、照片质感修复掩盖字体、形态、粗细问题。
2. 不再依赖视觉模型的 `ok` 覆盖本地硬指标。
3. 不再通过补丁叠补丁修复单张图，而是用明确阶段控制参数搜索方向。
4. 支持照片/扫描件、干净数字图、低分辨率缩略图、手动 ROI 等输入图自动分类，并由分类结果选择内部执行策略。
5. 每个阶段都能单独验证、单独回归、单独解释失败原因。

## 非目标

1. 不在第一版重写所有渲染算法。
2. 不取消现有视觉模型评估；视觉模型仍用于排序和最终验收。
3. 不把某张图、某个字、某个字体的特殊调参写死成通用规则。
4. 不允许“先交付，再靠用户肉眼指出问题”作为流程成功标准。

## 当前问题

现有流程的问题不是没有规则，而是规则混在同一层执行：

1. `text_shape` 中的真实笔画体量没过时，系统可能先开始清灰边。
2. 字体结构或局部姿态还没稳定时，视觉模型可能建议调黑度或模糊。
3. 旧字槽位、同一行邻字、视觉模型三种参照冲突时，没有统一仲裁顺序。
4. `revision_patches_for_round` 里多个补丁族共享同一候选评分，容易来回摆动。
5. 通过验收后难以解释：到底是字体过了、粗细过了，还是只是整体视觉模型给了 pass。

最近某个姓名替换任务的失败路径就是例子：

1. 用户指出其中一个目标字符不够粗。
2. 流程先修了灰雾和外圈浅灰。
3. 灰边变干净后，该目标字符仍然真实笔画体量不足。
4. 视觉模型一度给 `stroke_weight=ok`，但本地阶段顺序没有把粗细作为前置阻塞。

## 核心原则

### 全局硬约束不可重排

这些约束永远在所有内部策略前执行，失败时直接停止或返回 rejected candidate：

1. 输出尺寸必须等于原图尺寸。
2. ROI 外像素必须不变。
3. 图片边缘必须不变。
4. protected text 必须不变，例如动态上下文里的 `field_label_text`、`field_separator_text`、`protected_texts`、字段标签和后续未修改文字。
5. 目标 ROI 必须覆盖完整旧字，不能残留旧字笔画。
6. 新字不能覆盖旧字后面的未修改内容。

这些不是可组合阶段，而是安全边界。

### 可组合阶段必须有阻塞能力

每个可组合阶段都必须能回答四个问题：

1. 当前阶段是否通过。
2. 如果失败，是否阻塞后续阶段。
3. 本阶段允许调整哪些参数。
4. 本阶段禁止哪些参数抢先主导。

例如 `text_shape` 中的笔画体量失败时：

1. 允许主搜索调整 `font_size`、位置/字槽和局部字形宽度；`stroke_opacity` 只作为次级 `stroke_body_shape`，`core_ink_gain`、`core_darken_strength` 归入 `ink_gray_balance`。
2. 禁止 `blur`、`photo_noise`、`jpeg_quality` 抢先主导。
3. 禁止只靠外层 `120-165` 浅灰边假装变粗。

### 视觉模型只能辅助阶段判断

视觉模型可以指出“字体不对”“太细”“灰边多”“背景补丁明显”，但不能跳过本地阶段门槛。

如果视觉模型给 `pass`，但本地 `blocking_stage != null`，最终仍必须 `revise`。

如果本地 `blocking_stage == null` 或五个本地阶段全部通过，但视觉最终验收仍返回
`revise` / `marginal` 并给出具体视觉问题，流程进入 `vision_disagreement` 状态。
该状态不是第六个公开 stage，必须把视觉问题映射回现有五个 stage 之一，形成受限
`vision_target`，再由本地候选生成、过滤、评分和拒绝诊断处理。

### 临界失败和模型建议不能只停在说明层

视觉模型可以给出连续参数建议，但任何可转成本地 patch 或目标参数的建议都必须成为可审计的本地候选。
如果建议被过滤、约束、去重或渲染后失败，运行产物必须记录失败原因；不能只把建议留在
`final_acceptance_iterXX.json` 文本里。模型建议的落地分两层：单条参数建议生成
`forced_model_seed` 或拒绝记录；重复出现的视觉问题生成 `vision_target`，参与下一轮候选评分。

当本地失败只剩一个接近阈值的质量指标，例如 `ink_gray_balance` 下的
`roi_core_too_black`、`changed_char_core_too_black` 或方向相反的近阈值偏浅问题时，
流程必须进入 deterministic micro-search，而不是依赖增加 `max_revision_rounds`。
临界微调只能围绕当前阻塞阶段的允许参数做更小步长搜索，并保留每个候选的 stage severity
before/after、strict gate、prior stage regression 和 rejection reason。

受控跨阶段逃逸只能作为临界失败的例外机制：硬边界必须通过，前序阶段必须已通过或不回退，
当前阶段必须接近通过，并且逃逸候选必须显式标记 `controlled_escape=true`、`primary_stage`、
`secondary_stage` 和允许的微小参数范围。该机制不能重新引入全量跨阶段笛卡尔搜索。

## Stage 定义

### Stage 数据结构

建议引入 `StageSpec`：

```python
@dataclass(frozen=True)
class StageSpec:
    id: str
    display_name: str
    blocks_next: bool
    detect: Callable[[Report], StageResult]
    optimization_steps: tuple[str, ...]
    allowed_patch_keys: frozenset[str]
    blocked_patch_keys: frozenset[str]
```

阶段结果：

```python
@dataclass(frozen=True)
class StageResult:
    stage_id: str
    display_name: str
    passed: bool
    blocks_next: bool
    severity: str
    issues: list[dict[str, Any]]
    reason: str
    allowed_patch_keys: list[str]
    blocked_patch_keys: list[str]
```

报告中新增：

```json
{
  "classification": {
    "class_key": "photo_document.form_field_value_replace.cjk",
    "image_type": "photo_document",
    "scenario": "form_field_value_replace",
    "script": "cjk",
    "length_change": "same_length",
    "roi_input": "auto",
    "roi_policy": "auto",
    "profile_source": "classification",
    "internal_profile": "photo_scan"
  },
  "class_key": "photo_document.form_field_value_replace.cjk",
  "roi_policy": "auto",
  "profile_source": "classification",
  "internal_profile": "photo_scan",
  "stage_order": [
    "hard_boundary",
    "text_shape",
    "ink_gray_balance",
    "photo_texture",
    "background_cleanup"
  ],
  "stage_status": {
    "hard_boundary": {
      "pass": true,
      "blocks_next": true,
      "issues": []
    },
    "text_shape": {
      "pass": false,
      "blocks_next": true,
      "issues": [
        {
          "type": "changed_char_stroke_body_too_thin",
          "target_char": "target_char_n",
          "neighbor_char": "reference_char"
        }
      ]
    },
    "ink_gray_balance": {
      "pass": true,
      "blocks_next": true,
      "issues": []
    },
    "photo_texture": {
      "pass": true,
      "blocks_next": true,
      "issues": []
    },
    "background_cleanup": {
      "pass": true,
      "blocks_next": true,
      "issues": []
    }
  },
  "blocking_stage": "text_shape",
  "blocking_stage_blocks_next": true
}
```

### 当前 stage 语义

当前公开阶段门禁只有 5 个：`hard_boundary`、`text_shape`、`ink_gray_balance`、`photo_texture`、`background_cleanup`。
旧设计中的 `slot_alignment`、`font_structure`、`pose_geometry`、`stroke_body`、`tone_gray` 和 `edge_quality`
不能再作为 `stage_id`、`blocking_stage` 或公开 gate 输出，也不能作为运行时映射报告输出。
这些词只能作为内部 issue type、局部 detector 名称或阶段内 Optimization Step 的语义来源。

## 自动分类和内部策略

阶段顺序不由前端或用户可选 profile 决定。每张输入图必须先自动分类，再由分类结果选择内部策略、prompt pack 和参数范围。
分类输出至少包含 `image_type`、`scenario`、`script`、`length_change`、`roi_input`、`class_key`、`confidence` 和 evidence。
同一批图片逐图独立分类，同类图片可以复用策略，但不能共享上一张图的分类、失败原因、候选或模型建议。

### `photo_document.form_field_value_replace.*`

用于拍照件、扫描件、表单字段值替换场景。

特点：

1. 姿态和真实笔画粗细排在颜色和照片质感前。
2. `photo_texture` 只在 `text_shape` 和 `ink_gray_balance` 通过后执行。
3. 视觉模型不能在 `text_shape` 失败时直接 deliver。

### `clean_digital.numeric_or_date_replace`

用于截图、清晰数字图片、日期或编号替换、无明显拍照失真场景。

特点：

1. 内部策略可使用 `clean_digital`。
2. 不启用照片噪声或 `photo_warp` 作为主修复方向。
3. 边缘应更干净。

### `low_res_thumbnail.*`

用于极小图、缩略图、小尺寸压缩图。

特点：

1. 字体结构和笔画体量比精细灰阶更重要。
2. 允许更粗略的边缘指标。
3. 视觉验收要看放大上下文。

### `manual_exact` 和 `manual_anchor`

用户手动画框不是 profile。流程必须先判断矩形是精确 edit ROI 还是 search/anchor ROI。

特点：

1. `manual_exact` 可以作为实际编辑意图，但仍要做旧值槽位和 protected text 检查。
2. `manual_anchor` 只能作为 search/anchor ROI，后端必须重新定位旧值槽位并生成独立 `edit_roi` 或 `expanded_edit_roi`。
3. 只有 `manual_exact` 且确实无法定位旧值槽位时，才允许保守 fallback，并降低自动验收置信度。

## 阶段仲裁

每轮只允许一个 blocking stage 主导补丁选择。

算法：

```python
def determine_blocking_stage(report, internal_strategy):
    for stage_id in internal_strategy.stage_order:
        result = run_stage_detector(stage_id, report)
        if not result.passed and result.blocks_next:
            return stage_id
    return None
```

候选补丁：

```python
def revision_patches_for_round(params, acceptance, report, internal_strategy):
    stage = report["blocking_stage"]
    if stage is None:
        return final_acceptance_patches(acceptance)
    return STAGE_PATCHERS[stage](params, acceptance, report)
```

禁止逻辑：

```python
def filter_patch_by_stage(patch, stage_spec):
    if any(key in patch for key in stage_spec.blocked_patch_keys):
        return False
    if not any(key in patch for key in stage_spec.allowed_patch_keys):
        return False
    return True
```

## 补丁族重构

现有补丁族应迁移成 stage patchers：

| Stage | Patch Family | 当前可复用逻辑 |
| --- | --- | --- |
| `hard_boundary` | `hard_boundary_patches` | ROI adjustment, slot quality, protected guard diagnostics |
| `text_shape` | `text_shape_patches` | font candidate ranking, font size grid, char offsets, baseline, shear, stroke-body recovery |
| `ink_gray_balance` | `ink_gray_balance_patches` | black core reduction, opacity/core tuning, outer gray control |
| `background_cleanup` | `background_cleanup_patches` | old residue, ghost/shadow, seam and texture cleanup |
| `photo_texture` | `photo_texture_patches` | noise/JPEG/texture blending |

不允许继续出现一个补丁同时服务多个阶段，除非它声明：

1. 主阶段。
2. 次级影响。
3. 为什么不会破坏前置阶段。

### 迁移边界

阶段化实现不是在现有 `revision_patches_for_round` 里继续增加分支，而是逐步把旧入口缩成兼容层：

1. 每个 Phase 合入时必须说明替换了哪个旧路径。
2. 新增 patch 必须放进某个 stage patcher，不能继续散落在全局候选生成函数里。
3. 旧函数只能调用新的 stage dispatcher，不能反过来让 stage dispatcher 调旧的全局混合补丁。
4. 临时双轨只允许用于验证同一输入的新旧结果差异，不能作为长期交付路径。
5. 如果某个失败只能靠增加一次性参数解决，先补 detector 或 stage 定义，再补 patcher。

每个阶段 PR 或提交必须包含：

1. 本阶段新增或迁移的 detector。
2. 本阶段新增或迁移的 patcher。
3. 对应的 allowed/blocked 参数声明。
4. 至少一个失败用例和一个通过用例。
5. `progress.jsonl` 或 `result.json` 中可追踪的 stage evidence。

## 视觉模型 Prompt 调整

视觉模型 prompt 应增加阶段顺序约束：

1. 先判断当前 `blocking_stage` 是否真实存在。
2. 当 `blocking_stage` 存在时，只能针对当前 `blocking_stage` 给建议。
3. 当本地 `blocking_stage` 不存在但视觉仍拒绝交付时，必须输出可映射到五个公开 stage 的 `vision_target_stage` 和依据。
4. 不能建议当前阶段或映射目标阶段禁止的参数。
5. 如果它认为前置阶段已通过，必须说明依据。
6. 如果建议 deliver，但本地 `blocking_stage` 不为空，本地仍改为 revise。

Prompt 输入应包含：

```json
{
  "classification": {
    "class_key": "photo_document.form_field_value_replace.cjk",
    "roi_input": "manual",
    "roi_policy": "manual_anchor",
    "profile_source": "classification",
    "internal_profile": "photo_scan"
  },
  "blocking_stage": "text_shape",
  "stage_status": {...},
  "allowed_patch_keys": ["font_size_delta", "stroke_opacity_delta"],
  "blocked_patch_keys": ["blur", "photo_noise", "jpeg_quality"]
}
```

## 进度和 UI

CLI 和 Web 每轮必须显示：

```text
round 3
class_key: photo_document.form_field_value_replace.cjk
internal_profile: photo_scan
roi_policy: manual_anchor
blocking_stage: text_shape
reason: changed char core density below same-row neighbor
allowed_params: font_size_delta, text_dx_delta, text_dy_delta, char_offsets_delta, stroke_opacity_delta
blocked_params: blur, photo_noise, jpeg_quality
selected_optimization_step: stroke_body_search
```

Web 候选 drawer 应展示：

1. 每轮 `blocking_stage`。
2. 当前阶段失败原因。
3. 本轮候选图。
4. 哪些候选因前置阶段失败被拒绝。
5. 最终图是否 `accepted=true`。

## 分阶段实施计划

### Phase 1: 只加阶段报告，不改变行为

目标：

1. 增加 `classification`、`class_key`、`internal_profile` 和 `profile_source=classification`。
2. 增加 `stage_status`。
3. 增加 `blocking_stage`。
4. 仍使用现有补丁逻辑。

验收：

1. CLI 输出 `blocking_stage`。
2. `result.json` 包含完整阶段状态。
3. 当前回归图不改变最终产物。
4. 字段旧文字替换为新文字时，能报告 `text_shape`、`ink_gray_balance` 或 `photo_texture` 下的真实阻塞原因。

### Phase 2: 按 blocking stage 派发补丁

目标：

1. `revision_patches_for_round` 改为 stage dispatcher。
2. 每个 stage 只允许自己的参数族。
3. 视觉模型 patch 必须经过 stage filter。

验收：

1. `text_shape` 失败时，不会生成 `photo_noise`/`jpeg_quality` 主导补丁。
2. `ink_gray_balance` 失败时，不会改字体和基线。
3. 字体结构失败时，不会靠 blur 通过。

### Phase 3: 拆分补丁族

目标：

1. 建立 `stage_patchers.py` 或同等模块。
2. 将当前补丁函数迁移到对应 stage。
3. 删除跨阶段混用补丁。

验收：

1. 每个 patcher 有单元测试。
2. 每个 patcher 声明 allowed/blocked keys。
3. patcher 输出不能包含未声明参数。

### Phase 4: 自动分类和内部策略支持

目标：

1. Web/CLI 在 ROI 规划和候选生成前自动分类每张输入图。
2. 分类结果选择内部策略、prompt pack 和参数范围，并把执行名记录为 `internal_profile`。
3. Profile 不作为 Web 前端公开选择项；调试入口如保留必须标记为 non-Web override。

验收：

1. 同一类图片归并到稳定 `class_key`，不同类图片不会落入同一处理场景。
2. 混合批次逐图独立记录 `classification`、`internal_profile`、prompt context 和 candidate filters。
3. `clean_digital.numeric_or_date_replace` 自动选择内部 `clean_digital` 策略，并禁用照片噪声和 photo warp。

### Phase 5: Prompt 和 UI 对齐

目标：

1. 视觉 prompt 输入 stage 信息。
2. 视觉输出必须引用当前 stage。
3. Web 展示阶段进度。

验收：

1. 用户能看到为什么没有进入下一阶段。
2. rejected candidate 能说明阻塞阶段。
3. deliver 时所有 stage 均通过。

## 回归用例

### Case A: `字段旧文字修改为新文字`

目标：

1. 笔画体量必须作为 `text_shape` 问题先通过，不能绕到 `background_cleanup` 或 `photo_texture` 交付。
2. 如果某个目标字符相对同一行参考字符仍偏细，不能 deliver。
3. 清灰边不能牺牲真实笔画宽度。

检查：

1. `blocking_stage` 是否为 `text_shape`，且 issue 或 Optimization Step 指向笔画体量问题。
2. 目标字符与对应旧槽位或同一行参考字符的核心密度差是否在阈值内。
3. `120-165` 外圈灰边是否不过量。
4. 视觉模型 pass 时，本地 stage 是否全部 pass。

### Case B: 字数减少

目标：

1. 多余旧槽位必须清理。
2. 后续未修改文字不能移动。
3. 旧字残留属于 `hard_boundary`/slot quality 或 `background_cleanup`，不属于 `photo_texture`。

### Case C: 字数增加

目标：

1. 新字不能覆盖后续字段。
2. 用户原始框选不足时自动扩展 `edit_roi`，不能因空间不足或 `right_boundary` 在候选生成前阻断。
3. 字体和字距先过，再调颜色。
4. `right_boundary` 和 protected distance 只作为扩框诊断和最终 hard report 验收依据。

### Case D: 干净数字图

目标：

1. 自动分类为 `clean_digital.numeric_or_date_replace` 并选择内部 `clean_digital` 策略。
2. 不应增加照片噪声。
3. 边缘应干净。

## 代码落点

建议新增模块：

```text
src/roi_image_edit/stages.py
src/roi_image_edit/image_classification.py
src/roi_image_edit/stage_profiles.py
src/roi_image_edit/stage_patchers.py
```

职责：

1. `stages.py`: `StageSpec`, `StageResult`, detector mapping。
2. `image_classification.py`: 每张输入图的 `classification`、`class_key`、`roi_policy` 和证据报告。
3. `stage_profiles.py`: 由分类结果选择的内部策略定义和加载；不能作为 Web 前端选择器。
4. `stage_patchers.py`: 每个 stage 的补丁生成。

现有文件迁移：

1. `local_validation.py` 中的 `local_*_issues` 继续拆成 detector 的底层函数。
2. `revision_solver.py` 中的 `revision_patches_for_round` 继续缩减为 dispatcher。
3. `processing_service.py` 在 ROI 规划和候选生成前调用分类，并逐图隔离内部策略、prompt pack 和候选过滤器。
4. `iterative_pipeline.py` 的视觉 prompt 上下文加入 classification、ROI plan 和 stage 信息。
5. `result.json` 和 `progress.jsonl` 记录 classification、ROI plan、内部策略来源和 stage 状态。
6. `web_app.py` 只保留 HTTP/API/job 状态和本地 Web 服务启动，不承载阶段策略或图像处理逻辑。

## 成功标准

阶段化重构完成后，任何一次交付都必须满足：

1. 全局硬约束通过。
2. `blocking_stage == null`。
3. 分类推导出的内部策略中所有启用 stage 均通过。
4. 视觉模型最终 `pass=true` 且 `final_decision=deliver`。
5. 如果用户指出某阶段问题，例如“不够粗”，该问题能映射到明确 stage，而不是继续靠灰边、黑度或照片质感补丁试错。

## 反模式

以下做法禁止进入新实现：

1. 一个问题失败后直接把所有补丁族混合评分。
2. 视觉模型说 `ok` 就覆盖本地阶段失败。
3. 字体没过时调 blur。
4. 粗细没过时先清灰边。
5. 灰边过多时继续加 photo_noise。
6. 某张图的右倾/左倾写成通用规则。
7. 为单个字写特殊规则，例如“某目标字符必须右倾”。
8. 用更多迭代次数替代阶段判定。
