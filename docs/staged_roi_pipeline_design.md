# Staged ROI Text Replacement Pipeline Design

本文档描述下一版 ROI 文字替换流程的阶段化设计。目标不是继续给现有逻辑叠补丁，而是把“发现问题、阻塞后续阶段、生成候选、选择参数、验收交付”拆成可验证、可重排、可扩展的阶段系统。

## 设计目标

1. 不再让颜色、灰边、照片质感修复掩盖字体、形态、粗细问题。
2. 不再依赖视觉模型的 `ok` 覆盖本地硬指标。
3. 不再通过补丁叠补丁修复单张图，而是用明确阶段控制参数搜索方向。
4. 支持照片/扫描件、干净数字图、低分辨率缩略图、手动 ROI 快速处理等不同 profile。
5. 每个阶段都能单独验证、单独回归、单独解释失败原因。

## 非目标

1. 不在第一版重写所有渲染算法。
2. 不取消现有视觉模型评估；视觉模型仍用于排序和最终验收。
3. 不把某张图、某个字、某个字体的特殊调参写死成通用规则。
4. 不允许“先交付，再靠用户肉眼指出问题”作为流程成功标准。

## 当前问题

现有流程的问题不是没有规则，而是规则混在同一层执行：

1. `stroke_body` 没过时，`edge_quality` 可能先开始清灰边。
2. `font_structure` 或 `pose_geometry` 还没稳定时，视觉模型可能建议调黑度或模糊。
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

这些约束永远在所有 profile 前执行，失败时直接停止或返回 rejected candidate：

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

例如 `stroke_body` 失败时：

1. 允许调整 `stroke_opacity`、`font_size`、`core_ink_gain`、`core_darken_strength`、局部字形宽度。
2. 禁止 `blur`、`photo_noise`、`jpeg_quality` 抢先主导。
3. 禁止只靠外层 `120-165` 浅灰边假装变粗。

### 视觉模型只能辅助阶段判断

视觉模型可以指出“字体不对”“太细”“灰边多”“背景补丁明显”，但不能跳过本地阶段门槛。

如果视觉模型给 `pass`，但本地 `blocking_stage != null`，最终仍必须 `revise`。

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
  "pipeline_profile": "photo_scan",
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

### 当前 gate 与旧 7 类关注点关系

当前代码和 checklist 的阶段门禁只有 5 个：`hard_boundary`、`text_shape`、`ink_gray_balance`、`photo_texture`、`background_cleanup`。
下面的 `slot_alignment`、`font_structure`、`pose_geometry`、`stroke_body`、`tone_gray`、`edge_quality`、`photo_texture`
是旧设计中的 7 类关注点，不能再作为 `stage_id`、`blocking_stage` 或公开 gate 输出。
它们必须映射到当前 5 stage 下的 Optimization Step 或局部 detector。

| 旧 7 类关注点 | 当前 5 stage 中的归属 |
| --- | --- |
| `slot_alignment` | `hard_boundary` 的 ROI/slot 安全条件；`text_shape.slot_alignment_search` |
| `font_structure` | `text_shape.font_style_search`、`text_shape.font_size_search` |
| `pose_geometry` | `text_shape.pose_shear_search` |
| `stroke_body` | `text_shape.stroke_body_search` |
| `tone_gray` | `ink_gray_balance.core_black_search`、`ink_gray_balance.mid_gray_body_search`、`ink_gray_balance.opacity_search` |
| `edge_quality` | `ink_gray_balance.outer_gray_control`、`photo_texture.edge_breakup_match` |
| `photo_texture` | `photo_texture.blur_match`、`photo_texture.noise_texture_match`、`photo_texture.jpeg_texture_match` |

### 旧 7 类关注点列表

#### `slot_alignment`

职责：

1. 字符槽位。
2. 基线。
3. 字距。
4. 字符中心。
5. 是否覆盖旧字完整范围。

主要输入：

1. `slot_boxes`
2. `char_alignment_metrics`
3. `extra_source_slot_cleanup_metrics`
4. `protected_boxes`

失败时允许：

1. `char_offsets`
2. `text_dx`
3. `text_dy`
4. `target_roi` 收缩或扩展，但不能越过 protected text。

失败时禁止：

1. `blur`
2. `photo_noise`
3. `core_ink_gain`
4. `edge_breakup`

#### `font_structure`

职责：

1. 字体类别是否接近。
2. 字形结构是否接近。
3. 字号是否合理。
4. 字宽/字高比例是否接近。

主要输入：

1. `font_style_gate`
2. `build_font_style_reference`
3. 候选字体可渲染字符检查

失败时允许：

1. `font_name`
2. `font_path`
3. `font_size`
4. 字体候选池

失败时禁止：

1. 用 `opacity` 修字体。
2. 用 `blur` 模糊字体结构差异。
3. 用 `photo_noise` 掩盖字体错误。

#### `pose_geometry`

职责：

1. 局部倾斜。
2. 拍照扭曲。
3. 字符姿态继承。
4. 与旧槽位和相邻未修改字的方向一致性。

主要输入：

1. `char_pose_metrics`
2. `source_slot_shear`
3. `neighbor_shear`
4. `reference_shear`
5. `applied_shear`

失败时允许：

1. 局部 shear。
2. 小幅 `photo_warp`。
3. per-character pose 参数。
4. 必要时微调 `char_offsets`。

失败时禁止：

1. 通过加黑掩盖姿态不对。
2. 通过模糊掩盖倾斜不一致。
3. 固化某张图的“右倾”或“左倾”为通用规则。

#### `stroke_body`

职责：

1. 真实笔画几何粗细。
2. 核心暗部密度。
3. `<70` 深色密度。
4. 与同一行保留字的笔画体量一致性。
5. 复杂目标字的合理核心增量。

主要输入：

1. `char_gray_band_metrics`
2. `local_stroke_body_issues`
3. 同一行保留邻字密度
4. 字形复杂度比

通过条件示例：

1. 新字核心密度不能明显低于邻字。
2. `<70` 深色密度不能明显低于邻字。
3. `120-165` 浅灰边不能替代真实笔画。
4. 复杂字允许核心像素增量高于旧字，但必须受邻字上限约束。

失败时允许：

1. `stroke_opacity`
2. `font_size`
3. `core_ink_gain`
4. `core_darken_strength`
5. `alpha_contrast` 仅在用于收紧笔画边缘时允许

失败时禁止：

1. 增加 `blur` 来撑厚。
2. 增加 `photo_noise` 来撑厚。
3. 增加 `edge_breakup` 来掩盖细。
4. 只加 `core_ink_gain` 但不增加笔画体量。

#### `tone_gray`

职责：

1. 黑度。
2. 灰阶分布。
3. 真黑核心是否过多。
4. 核心是否偏浅。

主要输入：

1. `local_ink_balance_issues`
2. `strict_visual_metrics`
3. `<55`, `<70`, `70-90`, `90-120`, `120-165`

失败时允许：

1. `opacity`
2. `ink_gain`
3. `core_ink_gain`
4. `core_darken_strength`
5. `core_darken_threshold`

失败时禁止：

1. 换字体。
2. 改槽位。
3. 改姿态。
4. 增加照片噪声。

#### `edge_quality`

职责：

1. 外层灰边。
2. 抗锯齿边缘。
3. 锯齿/断裂感。
4. 是否有底部灰雾或外圈浅灰 halo。

主要输入：

1. `local_outer_gray_halo_issues`
2. `char_gray_band_metrics`
3. 邻字 `120-165` 外层灰边占比和密度

失败时允许：

1. `alpha_contrast`
2. 小幅 `blur`
3. `edge_breakup`
4. `stroke_opacity` 小幅减少

失败时禁止：

1. 改字体。
2. 改基线。
3. 用过量 `photo_noise` 制造灰雾。
4. 在 `stroke_body` 未过时抢先清灰边。

#### `photo_texture`

职责：

1. 背景纹理。
2. 局部噪声。
3. JPEG/压缩质感。
4. 拍照/扫描整体融合。

主要输入：

1. 背景补丁指标。
2. ROI 内局部纹理残差。
3. 视觉模型背景判断。

失败时允许：

1. `photo_noise`
2. `jpeg_quality`
3. 小幅 `photo_warp`
4. 背景修补参数

失败时禁止：

1. 修字体。
2. 修粗细。
3. 修基线。
4. 修 ROI 选择。

## Pipeline Profile

阶段顺序不写死在代码里，使用 profile。

### `photo_scan`

用于拍照件、扫描件、低清纸面图。

```text
slot_alignment
font_structure
pose_geometry
stroke_body
tone_gray
edge_quality
photo_texture
final_acceptance
```

特点：

1. 姿态和真实笔画粗细排在颜色和照片质感前。
2. `photo_texture` 最后执行。
3. 视觉模型不能在 `stroke_body` 失败时直接 deliver。

### `clean_digital`

用于截图、清晰数字图片、无明显拍照失真场景。

```text
slot_alignment
font_structure
stroke_body
tone_gray
edge_quality
final_acceptance
```

特点：

1. 不启用 `photo_texture`。
2. 不鼓励 `photo_warp`。
3. 边缘应更干净。

### `low_res_thumbnail`

用于极小图、缩略图、小尺寸压缩图。

```text
slot_alignment
font_structure
stroke_body
edge_quality
tone_gray
final_acceptance
```

特点：

1. 字体结构和笔画体量比精细灰阶更重要。
2. 允许更粗略的边缘指标。
3. 视觉验收要看放大图。

### `manual_roi_quick`

用于用户明确画框且只需要快速候选。

```text
slot_alignment
font_structure
stroke_body
final_acceptance
```

特点：

1. 只做最少阶段。
2. 未通过时保留 rejected candidate。
3. 不自动做复杂照片质感。

## 阶段仲裁

每轮只允许一个 blocking stage 主导补丁选择。

算法：

```python
def determine_blocking_stage(report, profile):
    for stage_id in profile.stage_order:
        result = run_stage_detector(stage_id, report)
        if not result.passed and result.blocks_next:
            return stage_id
    return None
```

候选补丁：

```python
def revision_patches_for_round(params, acceptance, report, profile):
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
| `slot_alignment` | `slot_alignment_patches` | char offsets, target ROI adjustment |
| `font_structure` | `font_structure_patches` | font candidate ranking, font size grid |
| `pose_geometry` | `pose_geometry_patches` | shear, pose inheritance, photo warp limited |
| `stroke_body` | `stroke_body_patches` | stroke/core density recovery |
| `tone_gray` | `tone_gray_patches` | black core reduction, opacity/core tuning |
| `edge_quality` | `edge_quality_patches` | outer halo cleanup, alpha contrast |
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
2. 只能针对当前 `blocking_stage` 给建议。
3. 不能建议当前阶段禁止的参数。
4. 如果它认为前置阶段已通过，必须说明依据。
5. 如果建议 deliver，但本地 `blocking_stage` 不为空，本地仍改为 revise。

Prompt 输入应包含：

```json
{
  "pipeline_profile": "photo_scan",
  "blocking_stage": "stroke_body",
  "stage_status": {...},
  "allowed_patch_keys": ["stroke_opacity", "core_ink_gain"],
  "blocked_patch_keys": ["blur", "photo_noise", "jpeg_quality"]
}
```

## 进度和 UI

CLI 和 Web 每轮必须显示：

```text
round 3
profile: photo_scan
blocking_stage: stroke_body
reason: changed char core density below same-row neighbor
allowed_params: stroke_opacity, font_size, core_ink_gain, core_darken_strength
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

1. 增加 `pipeline_profile`。
2. 增加 `stage_status`。
3. 增加 `blocking_stage`。
4. 仍使用现有补丁逻辑。

验收：

1. CLI 输出 `blocking_stage`。
2. `result.json` 包含完整阶段状态。
3. 当前回归图不改变最终产物。
4. 字段旧文字替换为新文字时，能报告 `stroke_body` 或 `edge_quality` 的真实阻塞原因。

### Phase 2: 按 blocking stage 派发补丁

目标：

1. `revision_patches_for_round` 改为 stage dispatcher。
2. 每个 stage 只允许自己的参数族。
3. 视觉模型 patch 必须经过 stage filter。

验收：

1. `stroke_body` 失败时，不会生成 `photo_noise`/`jpeg_quality` 主导补丁。
2. `edge_quality` 失败时，不会改字体和基线。
3. `font_structure` 失败时，不会靠 blur 通过。

### Phase 3: 拆分补丁族

目标：

1. 建立 `stage_patchers.py` 或同等模块。
2. 将当前补丁函数迁移到对应 stage。
3. 删除跨阶段混用补丁。

验收：

1. 每个 patcher 有单元测试。
2. 每个 patcher 声明 allowed/blocked keys。
3. patcher 输出不能包含未声明参数。

### Phase 4: Profile 支持

目标：

1. CLI 增加 `--profile photo_scan|clean_digital|low_res_thumbnail|manual_roi_quick`。
2. Web 每个任务记录 profile。
3. 自动 profile 仅作为建议，用户指定优先。

验收：

1. 同一张图可用不同 profile 运行。
2. `photo_scan` 包含 pose 和 texture。
3. `clean_digital` 不启用 photo texture。

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

1. `stroke_body` 必须在 `edge_quality` 前通过。
2. 如果某个目标字符相对同一行参考字符仍偏细，不能 deliver。
3. 清灰边不能牺牲真实笔画宽度。

检查：

1. `blocking_stage` 是否为 `stroke_body`。
2. 目标字符与对应旧槽位或同一行参考字符的核心密度差是否在阈值内。
3. `120-165` 外圈灰边是否不过量。
4. 视觉模型 pass 时，本地 stage 是否全部 pass。

### Case B: 字数减少

目标：

1. 多余旧槽位必须清理。
2. 后续未修改文字不能移动。
3. 旧字残留属于 `slot_alignment` 或 global hard gate，不属于 `photo_texture`。

### Case C: 字数增加

目标：

1. 新字不能覆盖后续字段。
2. ROI 可扩展，但必须受 protected text 限制。
3. 字体和字距先过，再调颜色。

### Case D: 干净数字图

目标：

1. profile 使用 `clean_digital`。
2. 不应增加照片噪声。
3. 边缘应干净。

## 代码落点

建议新增模块：

```text
src/roi_image_edit/stages.py
src/roi_image_edit/stage_profiles.py
src/roi_image_edit/stage_patchers.py
```

职责：

1. `stages.py`: `StageSpec`, `StageResult`, detector mapping。
2. `stage_profiles.py`: profile 定义和加载。
3. `stage_patchers.py`: 每个 stage 的补丁生成。

现有文件迁移：

1. `local_validation.py` 中的 `local_*_issues` 继续拆成 detector 的底层函数。
2. `revision_solver.py` 中的 `revision_patches_for_round` 继续缩减为 dispatcher。
3. `iterative_pipeline.py` 的视觉 prompt 上下文加入 stage 信息。
4. `result.json` 和 `progress.jsonl` 记录 stage 状态。
5. `web_app.py` 只保留 HTTP/API/job 状态和本地 Web 服务启动，不承载阶段策略或图像处理逻辑。

## 成功标准

阶段化重构完成后，任何一次交付都必须满足：

1. 全局硬约束通过。
2. `blocking_stage == null`。
3. profile 中所有启用 stage 均通过。
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
