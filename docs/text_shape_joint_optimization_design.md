# Text Shape Gates and Layered Joint Optimization Design

本文档沉淀当前 ROI 文字替换流程的下一步实施设计，重点覆盖三件事：

1. 目标流程的实施策略。
2. 现有流程与目标门禁的差距。
3. 字体形态联合优化如何控制搜索量，并落到代码中。

本文档不是单张图片的调参记录。任何具体图片、具体文字或具体字体结论都不能直接固化为通用规则；通用规则必须来自旧槽位、邻字、背景和本地指标。

## 三层结构

必须严格区分三层概念，不能把它们混成同一条 stage 链：

### 1. 前置安全流程

这些步骤发生在阶段门禁之前或属于 `hard_boundary` 的输入条件。失败时不能进入候选生成。

```text
方向校正
-> 字段和旧值 ROI 定位
-> 旧槽位完整性门禁
-> protected text 边界确认
```

### 2. 阶段门禁

阶段门禁只有以下五个，顺序必须和 `src/roi_image_edit/stage_policy.py` 中的 `STAGE_ORDER` 保持一致：

```text
hard_boundary
-> text_shape
-> ink_gray_balance
-> photo_texture
-> background_cleanup
```

这些是本地 gate。它们负责判断当前候选是否通过、哪个阶段阻塞、后续能否继续。

### 3. 阶段内优化

阶段内优化不是新 stage，而是某个阻塞阶段内部允许执行的 Optimization Step，即 solver/search/patch step。

```text
text_shape 内部：
  放置策略选择
  字体形态联合搜索
  字号、字槽、基线、stroke body、局部 shear 搜索

ink_gray_balance 内部：
  opacity、core gain、core darken、alpha contrast、outer gray 搜索

photo_texture 内部：
  blur、edge breakup、noise、compression、residual 搜索

background_cleanup 内部：
  最终背景融合、ghost/shadow、补丁纹理修复
```

视觉模型终检也不是本地 stage。它只能评估本地 top candidates、返回 JSON 建议，并且不能覆盖本地阶段门禁。

这个结构的核心约束是：不能用颜色、模糊、噪声、压缩或背景修补掩盖字体形态、位置、粗细、基线和姿态问题。

## 阶段表

| Stage | 目的 | 作用 | 阶段内 Optimization Steps | 视觉模型参与和 prompt |
| --- | --- | --- | --- | --- |
| `hard_boundary` | 保证这是在原图 ROI 内修改，而不是改坏整图或无关文字。 | 检查尺寸、ROI 外像素、边缘、protected text；方向/字段/旧槽位不可靠时阻塞候选生成。protected text guard 不区分左、右、上、下，任何未修改文字与目标 ROI、旧字清理范围或实际改动像素交叠都必须失败。 | `orientation_check`、`field_roi_selection`、`slot_quality_gate`、`protected_text_guard`、`hard_check`。 | 不依赖视觉模型裁决。`candidate_rank_prompt.txt` 和 `final_acceptance_prompt.txt` 会看到 hard report，但不能覆盖该阶段失败。 |
| `text_shape` | 先把文字形态放对、放像、放稳。 | 阻塞字体、字号、槽位、行基线、字距、笔画身体、局部姿态错误；行基线必须同时考虑旧文字原位置和同一行未修改文字，不能只按 ROI 中心放置；禁止黑灰、模糊、背景补丁抢先掩盖形态问题。 | `placement_strategy`、`shape_change_detection`、`font_style_search`、`font_size_search`、`slot_alignment_search`、`row_baseline_check`、`stroke_body_search`、`pose_shear_search`、`shape_reset`。 | `candidate_rank_prompt.txt` 可在本地 top candidates 中比较字体和形态；prompt 输入包含 `stage_context`；`final_acceptance_prompt.txt` 最终验收必须尊重本地 `text_shape` gate，模型建议会被本地 stage filter 过滤。 |
| `ink_gray_balance` | 让新字黑灰比例接近旧字和邻字。 | 分开控制真黑核心、中灰笔画身体、外灰边，避免“太黑/太淡/太硬/太灰”混成一个方向。 | `core_black_search`、`mid_gray_body_search`、`outer_gray_control`、`opacity_search`、`core_gain_search`、`alpha_contrast_search`。 | `candidate_rank_prompt.txt` 可比较候选黑灰观感；`tuning_prompt.txt` 可建议 opacity/core/contrast 小步变化；`final_acceptance_prompt.txt` 不能接受本地黑灰 gate 失败的候选。 |
| `photo_texture` | 匹配照片/扫描件的模糊、断裂、噪声和压缩质感。 | 在形态和黑灰过关后，修复过清晰、过干净、过糊、边缘无断裂等照片质感问题。 | `blur_match`、`edge_breakup_match`、`noise_texture_match`、`jpeg_texture_match`、`residual_retexture`、`alpha_degradation_search`。 | `candidate_rank_prompt.txt` 可比较 top candidates 的照片感；`tuning_prompt.txt` 可建议 blur/noise/compression；`final_acceptance_prompt.txt` 做最终自然度验收。 |
| `background_cleanup` | 让最终候选周围背景自然，且没有旧字残留。 | 验收旧字残影、白影、暗影、平滑涂抹、背景纹理断裂和 ROI 边缘接缝。前置旧槽位清除失败不能拖到此阶段补救。 | `old_slot_cleanup_check`、`ghost_residual_repair`、`shadow_residual_repair`、`background_texture_repair`、`seam_gradient_repair`、`final_blend_check`。 | `candidate_rank_prompt.txt` 可指出候选补丁感；`final_acceptance_prompt.txt` 必须检查背景自然度和残影。视觉模型建议只能生成 JSON patch，不能放过本地 background gate。 |

所有视觉 prompt 都以 `master_prompt.txt` 作为 system prompt。Web 路径当前加载 `candidate_rank_prompt.txt` 和 `final_acceptance_prompt.txt`；CLI 迭代路径还会加载 `tuning_prompt.txt`。`font_size_prompt.txt` 和 `darkness_blur_prompt.txt` 是保留的专项诊断 prompt 资产，不能替代阶段门禁。

视觉 prompt 的字段上下文必须由运行时动态注入：`field_key`、`field_label_text`、`field_separator_text`、`protected_texts` 和 `protected_boxes`。静态 prompt 不允许写死某个字段标签、默认参考字、固定字号枚举或固定候选参数值；字段标签和标点只来自指令解析、自动 ROI evidence 或后续 OCR/检测结果。

## 实施策略

### 1. 方向、字段和旧值 ROI

自动流程必须先解析用户指令，得到字段、旧值和新值。之后再尝试方向校正和字段定位。

要求：

- 自动方向选择不能只看整页文字方向评分，还要看目标字段和旧值定位质量。
- 字段搜索 ROI 和实际编辑 ROI 分离。
- 搜索 ROI 可以较宽，用于找字段锚点、旧值和后续保护文本。
- 编辑 ROI 必须收缩到旧值槽位和必要空白，不能把整行当成修改目标。
- 找不到字段或旧值时立即失败，并保留 rejected 产物；不能静默输出原图或无效结果。

### 2. 旧槽位完整性门禁

候选生成前必须确认旧值槽位质量。旧槽位不完整时，不允许进入渲染候选。

必须检查：

- 旧值字符数与槽位数匹配。
- 每个旧槽位覆盖完整笔画、灰边、底部和倾斜外溢。
- 槽位没有把字段标签、冒号或后续未修改文本混进去。
- 旧值最后一个字不能被误判成 protected text。
- 字数减少时，多余旧槽位必须进入前置清除区域。
- 字数增加时，编辑区域右边界必须受后续保护文本限制。

旧槽位门禁失败的表现不是 `text_shape`，而是更前置的 ROI/slot 问题。它必须阻塞候选生成，否则后续字体、黑度和背景修补都会在错误目标上工作。

### 3. 放置策略选择

放置策略不应固定为单一方式。应根据字数关系、旧槽位质量和单字形态变化选择。

| 场景 | 推荐策略 | 主要约束 |
| --- | --- | --- |
| 同字数 CJK，字形变化小 | 槽位左上边界贴齐 | 限制中心误差、字距、基线 |
| 同字数 CJK，字形变化大 | 槽位中心优先 | 限制左边界、基线、字距 |
| 字数减少 | 目标字按旧值整体跨度排布 | 清理多余旧槽位 |
| 字数增加 | 左边界锚定，保留旧字槽位并从旧值最右侧追加新增字 | 不覆盖后续 protected text；不得为了容纳新增字压缩旧字槽位 |
| 数字、日期、编号 | 左对齐和基线优先 | 保持数字节奏和字段宽度 |
| 手动 ROI 且无旧值 | 保守居中或左对齐 fallback | 必须降低自动验收置信度 |

无论使用逐字绘制、span 绘制还是中心绘制，`row_baseline_metrics` 都必须输出候选整体 bbox、旧槽位 bbox、同一行 protected boxes、参考中心线和上下边界偏移。参考线以旧槽位为主，同一行未修改文字只作为上下文，防止 ROI 画得偏高或偏低时把新字整体拖离原文字所在行。

同字数替换时，当前实现偏向按旧槽位左上边界绘制。这个策略适合同字体、同结构、同宽高的字符，但对结构差异明显的单字替换不够稳。目标实现应增加 `placement_strategy`：

```text
top_left_anchor
center_primary
left_anchor_span
baseline_numeric
manual_fallback
```

报告中必须写出实际使用的策略，以及为什么选择它。

### 4. 单字形态变化判定

“单字变化较大”不能靠语义判断，也不能用人工写死的字表。应按旧槽位真实图像和新字候选渲染形态比较。

建议指标：

- `bbox_width_delta_ratio`：新旧字形宽度差。
- `bbox_height_delta_ratio`：新旧字形高度差。
- `centroid_dx/dy`：左上贴齐后，新字质心相对旧字质心的偏移。
- `ink_area_ratio`：新字有效墨迹面积与旧字墨迹面积比。
- `row_projection_distance`：横向投影轮廓差异。
- `col_projection_distance`：纵向投影轮廓差异。
- `margin_distribution_delta`：上下左右边距分布差异。

触发条件示例：

```text
shape_change_large =
  centroid_error > slot_height * 0.08
  or abs(width_delta_ratio) > 0.14
  or abs(height_delta_ratio) > 0.10
  or projection_distance > dynamic_projection_limit
  or ink_area_ratio outside dynamic_ink_area_range
```

阈值应来自旧槽位高度、邻字稳定性和字体候选分布，而不是固定经验值。固定数字只能作为第一版保守起点，并必须写入报告。

### 5. 字体形态联合搜索

字体形态联合搜索只处理形态阶段，不处理最终黑灰和照片质感。

形态阶段联合搜索维度：

- 字体候选。
- 字号。
- 放置策略。
- `text_dx/text_dy`。
- 单字 `char_offsets`。
- 轻描边或笔画身体参数。
- 局部 shear/姿态继承。

本阶段排序指标：

- 字高、字宽、字距、基线。
- 候选整体行基线相对旧槽位和同一行未修改文字的偏移。
- 单字中心与旧槽位中心误差。
- 左边界和右边界误差。
- 笔画面积和复杂度修正后的体量。
- 姿态继承误差。
- protected text 距离。
- 字体风格分数。

形态没通过时，禁止 blur、noise、JPEG、背景融合成为主要修复方向。

### 6. 黑灰比例搜索

黑灰阶段只在形态候选前几名上执行。

要分开判断：

- `<55` 真黑核心。
- `<70` 深色核心。
- `70-120` 中灰笔画身体。
- `120-165` 外灰边。

规则：

- 核心太黑时，优先降低 `opacity`、`core_ink_gain`、`core_darken_strength`。
- 核心不足但灰边多时，不能继续加 blur 或扩大灰边，应恢复核心密度并收紧外灰。
- 只剩近阈值核心偏亮时，启用 `core_only_micro_recovery`：微步恢复核心黑度，候选必须排在普通 ink grid 前面，且不能改变字体、字号、槽位、行基线、模糊、mask 或背景参数。
- 旧字和邻字指标冲突时，优先使用同一行邻字作为风格上限，但必须在报告中写出仲裁。
- 黑灰阶段不能改变已经通过的字体、槽位和基线，除非重新回到形态阶段。
- 如果候选已经通过 `ink_gray_balance`，但后续被 `photo_texture` 或 `background_cleanup` 阻塞，选择器必须把它视为阶段推进候选并进入下一轮；不能因为最终仍未通过而把它丢弃为不可选择候选。

### 7. 照片质感搜索

照片质感阶段在形态和黑灰通过后执行。

可调参数：

- 小幅 blur。
- edge breakup。
- 局部噪声。
- 压缩质感。
- 轻微 alpha 退化。
- 局部残差回填。

要求：

- 目标是匹配原图拍照/扫描质感，不是把字弄糊。
- 不能让照片质感破坏黑灰和形态指标。
- 文字过清晰、过干净、过糊、边缘无断裂都应作为 photo_texture 问题报告。

### 8. 背景处理拆分

背景处理必须拆成前置清除和后置融合。

前置清除：

- 删除旧值槽位内旧字核心和灰边。
- 灰边残留用候选局部背景动态阈值判断，避免把拍照灰底或目标字抗锯齿低 alpha 边缘误判为旧残留。
- 旧字核心残留仍按硬阈值阻塞。
- 字数减少时清理多余旧槽位。
- 清除失败必须阻塞候选生成或阻塞最终验收。

后置融合：

- 修复最终候选周围补丁感、发白、发暗、平滑涂抹和纹理断裂。
- 只能围绕最终文字形态做局部融合。
- 不能用后置融合掩盖旧槽位没清干净。

## 现有流程差距

本表只同步当前状态，权威关闭条件在
[`workflow_checklist.md`](workflow_checklist.md)。
状态含义：

- `已覆盖`：checklist 中对应能力已经有 `[x]`、证据和测试。
- `部分覆盖`：已有代码和测试，但仍有未关闭 checklist 项。
- `未完成`：对应能力仍以 checklist `[ ]` 为主，不能声明完成。

| 目标能力 | 当前同步状态 | Checklist 对应项 |
| --- | --- | --- |
| 方向和目标字段联合选择 | 部分覆盖：指令解析、方向质量、找不到即失败已覆盖；自动分类和 manual anchor ROI 仍未关闭。 | 已覆盖：[方向、字段和旧值 ROI](workflow_checklist.md#d-方向字段和旧值-roi)；未完成：[自动分类、场景和内部 profile](workflow_checklist.md#n-自动分类场景和内部-profile)。 |
| 搜索 ROI 与编辑 ROI 分离 | 部分覆盖：search/edit ROI 分离和标注图已覆盖；manual anchor 和 expanded edit ROI 仍未关闭。 | 已覆盖：[方向、字段和旧值 ROI](workflow_checklist.md#d-方向字段和旧值-roi)；未完成：[全局硬约束](workflow_checklist.md#c-全局硬约束)、[放置策略选择](workflow_checklist.md#f-放置策略选择)。 |
| 旧槽位完整性门禁 | 部分覆盖：旧槽位质量已折叠为一个完整报告条目；字数增加时右边界仍需降级为扩框诊断。 | 已覆盖：[旧槽位完整性门禁](workflow_checklist.md#e-旧槽位完整性门禁)；未完成：[全局硬约束](workflow_checklist.md#c-全局硬约束)。 |
| 同字数 CJK 放置 | 部分覆盖：同字数小/大变化已有策略；分类驱动 placement matrix 仍未关闭。 | 已覆盖：[放置策略选择](workflow_checklist.md#f-放置策略选择)；未完成：[自动分类、场景和内部 profile](workflow_checklist.md#n-自动分类场景和内部-profile)。 |
| 单字形态变化检测 | 已覆盖：几何、投影、边距、墨迹、动态阈值和禁用语义字表已有测试。 | 已覆盖：[单字形态变化检测](workflow_checklist.md#g-单字形态变化检测)。 |
| 字体形态搜索 | 已覆盖：`text_shape` grid、形态评分组成、形态先行约束和形态剪枝原因均已有测试。 | 已覆盖：[字体形态联合搜索](workflow_checklist.md#h-字体形态联合搜索)、[分层联合优化和搜索预算](workflow_checklist.md#l-分层联合优化和搜索预算)。 |
| 姿态继承 | 已覆盖：pose scoring 和形态剪枝原因已覆盖，旧 7 类 concern 映射已从运行时和主 checklist 删除。 | 已覆盖：[单字形态变化检测](workflow_checklist.md#g-单字形态变化检测)、[字体形态联合搜索](workflow_checklist.md#h-字体形态联合搜索)。 |
| 黑灰门禁 | 已覆盖：分层执行、四段灰度、核心过黑/过浅、同一行邻字仲裁、形态参数保护和黑灰剪枝原因均已有测试。 | 已覆盖：[黑灰比例搜索](workflow_checklist.md#i-黑灰比例搜索)、[分层联合优化和搜索预算](workflow_checklist.md#l-分层联合优化和搜索预算)。 |
| 照片质感 | 已覆盖：执行顺序、允许参数、照片质感指标、前置阶段回退检查、issue types 和剪枝原因均已有测试。 | 已覆盖：[照片质感搜索](workflow_checklist.md#j-照片质感搜索)、[分层联合优化和搜索预算](workflow_checklist.md#l-分层联合优化和搜索预算)。 |
| 背景处理 | 已覆盖：前置清除、后置融合、旧槽位残留和局部融合优先级已有测试。 | 已覆盖：[背景处理拆分](workflow_checklist.md#k-背景处理拆分)。 |
| 视觉模型 | 已覆盖：只看本地 top candidates、prompt stage context、本地 stage filter、deliver 覆盖阻止和不可转化建议记录均已有测试。 | 已覆盖：[三层流程边界](workflow_checklist.md#a-三层流程边界)、[视觉模型 prompt 和本地仲裁](workflow_checklist.md#o-视觉模型-prompt-和本地仲裁)、[反模式门禁](workflow_checklist.md#r-反模式门禁)。 |

## 分层联合优化设计

联合优化不能做全量笛卡尔积。必须使用分层联合、动态剪枝和少量视觉评估。

### 不允许的搜索方式

```text
font_count
* font_size_count
* dx_count
* dy_count
* stroke_count
* opacity_count
* blur_count
* shear_count
* texture_count
```

这种全量组合会让搜索量膨胀到不可控，也会让阶段原因无法解释。

### 推荐搜索方式

```text
Stage A: shape search
  字体 / 字号 / 放置策略 / dx dy / char_offsets / stroke body / shear
  -> 本地形态指标剪枝
  -> 保留 top 20-50

Stage B: ink-gray search
  opacity / core gain / core darken / alpha contrast / outer gray
  -> 本地黑灰指标剪枝
  -> 保留 top 8-20

Stage C: photo texture search
  blur / edge breakup / noise / compression / residual
  -> 本地照片质感和背景指标剪枝
  -> 保留 top 3-8

Stage D: vision final check
  视觉模型只看 top 3-8
  -> 返回 JSON 建议
  -> 本地决定是否小步调参或 reject
```

### 搜索预算

第一版建议预算：

| 阶段 | 本地候选量 | 视觉候选量 |
| --- | --- | --- |
| shape | 300-1500 | 0 |
| ink-gray | 100-800 | 0 |
| photo texture | 30-200 | 0 |
| final visual | 3-8 | 3-8 |

ROI 很小时，本地渲染和 NumPy 指标可以承受几百到一两千候选。视觉模型不应参与大规模搜索。

### 临界失败微调

分层搜索不能因为避免全量笛卡尔积而漏掉临界小步调整。若当前候选满足以下条件，应进入当前
stage 的 micro-search：

1. `hard_boundary.pass=true`。
2. 当前 `blocking_stage` 是本地阶段，不是单纯视觉主观拒绝。
3. 失败 issue 数量很少，且至少一个主要 issue 的 `gap = actual - limit` 已接近阈值。
4. 前序阶段已通过，或本轮候选能证明前序阶段不回退。

第一版必须先覆盖两个方向：

| 场景 | 触发 issue | 候选族 | 禁止变化 |
| --- | --- | --- | --- |
| 过黑近阈值 | `roi_core_too_black`、`changed_char_core_too_black`、`excess_black_core` | `near_threshold_overblack_micro_tuning`：微降 `opacity`、`core_ink_gain`、`core_darken_strength`、`alpha_contrast`，少量组合。 | 字体、字号、位置、字槽、mask、背景、照片噪声。 |
| 偏浅近阈值 | `core_mean_gray_too_light`、`core_lighten_too_high` | `near_threshold_core_light_micro_tuning`：微增 core 相关参数，保持形态。 | 字体、字号、位置、mask、背景。 |

micro-search 的候选不能被常规 top-N 轴优先剪枝吞掉。报告必须分开写：

```json
{
  "micro_tuning": {
    "enabled": true,
    "family": "near_threshold_overblack_micro_tuning",
    "stage_id": "ink_gray_balance",
    "metric": "roi_core_too_black.lt55_delta",
    "actual": 117.0,
    "limit": 113.687,
    "gap": 3.313,
    "candidate_count": 12,
    "candidate_ids": ["c000_m01", "c000_m02"]
  }
}
```

### 模型建议强制落地

视觉模型建议不是 stage policy，但可转换建议必须成为本地候选来源之一。

执行规则：

1. `suggested_patch` 和 `parameter_suggestions` 先走现有 stage/profile filter。
2. 通过 filter 的建议必须生成 `forced_model_seed` 候选，不得只参与 patch dispatch 文本记录。
3. 如果建议被 constraint 改写，记录 `raw_patch`、`constrained_patch` 和改写原因。
4. 如果建议被去重，记录 `deduped_to_candidate_id`。
5. 如果建议生成候选但候选不可选，记录不可选原因。

建议落地产物建议写入：

```json
{
  "forced_model_candidates": [
    {
      "source": "final_acceptance_basis_round_7",
      "raw_suggestion": {"parameter": "opacity", "to": 0.67},
      "converted_patch": {"opacity_delta": -0.03},
      "candidate_id": "c000_fm08_01",
      "rendered": true,
      "selectable": false,
      "rejection_reason": "prior_stage_regression:text_shape.center_dx"
    }
  ]
}
```

### 受控跨阶段逃逸

默认仍禁止跨阶段混合调参。只有在临界失败时，允许一个显式标记的 escape 族：

1. `controlled_escape=true`。
2. `primary_stage` 必须仍是当前 blocking stage。
3. `secondary_stage` 只能是紧邻相关阶段，例如 `ink_gray_balance` 临界失败时的小幅
   `photo_texture.blur_match` 或 `text_shape` 的 ±1px 位置回验。
4. 所有 secondary 参数必须有硬上限，例如 `text_dx/text_dy/char_offsets <= 1px`、
   `blur_delta <= 0.02`、`photo_noise_delta <= 0.004`。
5. escape 候选必须通过 prior-stage regression；只要硬边界、旧槽位、protected text、
   text shape hard issue 或背景残留回退，就必须拒绝。

该机制的目标是解除“只差最后几个像素但常规网格无法命中”的死锁，不是恢复全阶段联合搜索。

### 剪枝规则

形态阶段先剪掉：

- 字高差超限。
- 单字中心偏离旧槽位过大。
- 基线偏移超限。
- 字距破坏。
- protected text 距离不足。
- 字体风格分数过差。
- 单字笔画体量明显偏离。
- 姿态继承方向错误或幅度过大。

黑灰阶段先剪掉：

- `<55` 真黑核心超动态上限。
- `<70` 深色核心明显不足。
- `120-165` 外灰边占比过高。
- 中灰笔画身体不足。
- 复杂度修正后仍过黑或过淡。

照片质感阶段先剪掉：

- 文字边缘过锐。
- 文字过糊。
- 边缘无断裂感。
- 背景平滑补丁。
- 白影、暗影或旧字残留。
- ROI 边缘亮度梯度断裂。

## 第一版实施切片

### Slice 1: 旧槽位硬门禁

- 增加 `slot_quality_gate`。
- 输出 `slot_quality_report`。
- 候选生成前检查旧值槽位数、槽位完整性、底部/灰边覆盖、protected text 冲突。
- 失败时直接返回 rejected，不进入字体搜索。

### Slice 2: 放置策略选择器

- 增加 `placement_strategy` 字段。
- 实现 `top_left_anchor` 和 `center_primary`。
- 同字数 CJK 默认先计算单字形态变化。
- 形态变化大时使用中心优先，左边界和基线作为约束。

### Slice 3: 单字形态变化检测器

- 对每个 changed char 生成旧槽位画像和新字候选画像。
- 计算 bbox、质心、投影、墨迹面积和边距分布差异。
- 把 `shape_change_large` 写入报告和候选排序。

### Slice 4: 形态联合搜索

- 在 `text_shape` 阻塞时生成 shape candidate grid。
- shape grid 只包含字体、字号、放置、offset、stroke body、shear。
- 本地评分后保留 top candidates。
- 禁止 photo texture 参数抢先修复。

### Slice 5: 分层候选产物

每个失败任务必须保留：

- 自动方向选择报告。
- 搜索 ROI 和编辑 ROI 标注图。
- `slot_quality_report`。
- shape top candidates。
- ink-gray top candidates。
- photo texture top candidates。
- final visual candidates。
- rejected final candidate。

### Slice 6: 临界失败、模型 forced seed 和拒绝诊断

- 增加 `near_threshold_overblack_micro_tuning`，与现有偏浅 micro tuning 对称。
- micro tuning 候选不受常规 ink-gray top-N 轴优先剪枝吞掉，必须单独计数和审计。
- 可转换的模型 `suggested_patch` / `parameter_suggestions` 必须生成 forced seed candidate。
- 每轮 revision 必须输出 `candidate_rejection_table`，覆盖所有被渲染但不可选的候选。
- `no_selectable_revision_candidate` 不能只写 stop reason；必须说明每个候选为什么不可选。
- 增加 `controlled_escape` 候选族，但只能用于接近通过的当前阻塞阶段，并必须通过 prior-stage regression。
- 字数增加或目标字复杂度明显增加时，黑灰阈值报告必须包含复杂度归一化字段，不能只看全 ROI `<55` 增量。

## Done Definition

该设计完成不能只看某一张图效果。必须满足：

- 自动 ROI 失败时立即报错或保留 rejected candidate。
- 旧槽位不完整时不生成最终候选。
- 同字数 CJK 能在左上贴齐和中心优先之间自动选择。
- 字形变化大时，候选报告能说明为什么触发中心优先。
- `text_shape` 存在 hard-blocking issues 时，流程不会先调照片质感或背景融合；如果只剩受黑灰影响的 deferred issues，则进入声明的后续阶段并在下一轮回验形态。
- 视觉模型只评估本地 top candidates，不能覆盖本地硬门禁。
- 失败也有足够中间产物供用户检查。
- 近阈值过黑和近阈值偏浅都能进入 micro-search，并能证明该搜索没有改变字体、字号、位置、mask 或背景。
- 每个可转换的视觉模型参数建议都能追溯到本地 forced seed candidate、去重记录或拒绝记录。
- `no_selectable_revision_candidate` 必须伴随完整候选拒绝表，不能只靠停止原因解释失败。
- 受控跨阶段逃逸只能在临界失败条件下启用，并且必须记录 primary/secondary stage、参数上限和 prior-stage regression。
- 字数或复杂度增加的替换必须记录复杂度归一化阈值，且不能放宽 hard boundary。
