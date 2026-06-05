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

阶段门禁只有以下五个，顺序必须和代码中的 `STAGE_ORDER` 保持一致：

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
| `hard_boundary` | 保证这是在原图 ROI 内修改，而不是改坏整图或无关文字。 | 检查尺寸、ROI 外像素、边缘、protected text；方向/字段/旧槽位不可靠时阻塞候选生成。 | `orientation_check`、`field_roi_selection`、`slot_quality_gate`、`protected_text_guard`、`hard_check`。 | 不依赖视觉模型裁决。`candidate_rank_prompt.txt` 和 `final_acceptance_prompt.txt` 会看到 hard report，但不能覆盖该阶段失败。 |
| `text_shape` | 先把文字形态放对、放像、放稳。 | 阻塞字体、字号、槽位、基线、字距、笔画身体、局部姿态错误；禁止黑灰、模糊、背景补丁抢先掩盖形态问题。 | `placement_strategy`、`shape_change_detection`、`font_style_search`、`font_size_search`、`slot_alignment_search`、`stroke_body_search`、`pose_shear_search`、`shape_reset`。 | `candidate_rank_prompt.txt` 可在本地 top candidates 中比较字体和形态；CLI 的 `tuning_prompt.txt` 可输出 JSON 调参建议；`final_acceptance_prompt.txt` 最终验收必须尊重本地 `text_shape` gate。 |
| `ink_gray_balance` | 让新字黑灰比例接近旧字和邻字。 | 分开控制真黑核心、中灰笔画身体、外灰边，避免“太黑/太淡/太硬/太灰”混成一个方向。 | `core_black_search`、`mid_gray_body_search`、`outer_gray_control`、`opacity_search`、`core_gain_search`、`alpha_contrast_search`。 | `candidate_rank_prompt.txt` 可比较候选黑灰观感；`tuning_prompt.txt` 可建议 opacity/core/contrast 小步变化；`final_acceptance_prompt.txt` 不能接受本地黑灰 gate 失败的候选。 |
| `photo_texture` | 匹配照片/扫描件的模糊、断裂、噪声和压缩质感。 | 在形态和黑灰过关后，修复过清晰、过干净、过糊、边缘无断裂等照片质感问题。 | `blur_match`、`edge_breakup_match`、`noise_texture_match`、`jpeg_texture_match`、`residual_retexture`、`alpha_degradation_search`。 | `candidate_rank_prompt.txt` 可比较 top candidates 的照片感；`tuning_prompt.txt` 可建议 blur/noise/compression；`final_acceptance_prompt.txt` 做最终自然度验收。 |
| `background_cleanup` | 让最终候选周围背景自然，且没有旧字残留。 | 验收旧字残影、白影、暗影、平滑涂抹、背景纹理断裂和 ROI 边缘接缝。前置旧槽位清除失败不能拖到此阶段补救。 | `old_slot_cleanup_check`、`ghost_residual_repair`、`shadow_residual_repair`、`background_texture_repair`、`seam_gradient_repair`、`final_blend_check`。 | `candidate_rank_prompt.txt` 可指出候选补丁感；`final_acceptance_prompt.txt` 必须检查背景自然度和残影。视觉模型建议只能生成 JSON patch，不能放过本地 background gate。 |

所有视觉 prompt 都以 `master_prompt.txt` 作为 system prompt。Web 路径当前加载 `candidate_rank_prompt.txt` 和 `final_acceptance_prompt.txt`；CLI 迭代路径还会加载 `tuning_prompt.txt`。`font_size_prompt.txt` 和 `darkness_blur_prompt.txt` 是保留的专项诊断 prompt 资产，不能替代阶段门禁。

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
| 字数增加 | 左边界锚定，向右扩展 | 不覆盖后续 protected text |
| 数字、日期、编号 | 左对齐和基线优先 | 保持数字节奏和字段宽度 |
| 手动 ROI 且无旧值 | 保守居中或左对齐 fallback | 必须降低自动验收置信度 |

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
- 旧字和邻字指标冲突时，优先使用同一行邻字作为风格上限，但必须在报告中写出仲裁。
- 黑灰阶段不能改变已经通过的字体、槽位和基线，除非重新回到形态阶段。

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
- 字数减少时清理多余旧槽位。
- 清除失败必须阻塞候选生成或阻塞最终验收。

后置融合：

- 修复最终候选周围补丁感、发白、发暗、平滑涂抹和纹理断裂。
- 只能围绕最终文字形态做局部融合。
- 不能用后置融合掩盖旧槽位没清干净。

## 现有流程差距

| 目标能力 | 现有状态 | 差距 |
| --- | --- | --- |
| 方向和目标字段联合选择 | 已能尝试多方向并按 ROI 质量排序 | 仍需把字段置信、旧值槽位质量写成统一报告 |
| 搜索 ROI 与编辑 ROI 分离 | 已开始分离 | 需要形成所有字段通用路径，而不是只覆盖近期失败场景 |
| 旧槽位完整性门禁 | 已能修复部分漏槽位和底部裁切 | 尚未成为候选生成前的统一硬门禁 |
| 同字数 CJK 放置 | 主要按旧槽位左上边界贴齐 | 缺少 `center_primary` 和策略选择器 |
| 单字形态变化检测 | 还没有独立检测器 | 需要 bbox、质心、投影、墨迹面积联合判断 |
| 字体形态搜索 | 已有字体、字号、offset、shear 等候选 | 还不是严格的形态先行联合搜索和剪枝 |
| 姿态继承 | 已有局部 shear 估计 | 需要纳入形态搜索，并与中心/基线共同评分 |
| 黑灰门禁 | 已有核心黑、中灰、外灰指标 | 需要更严格地禁止形态未过时先调黑灰 |
| 照片质感 | 已有 blur、noise、texture 相关参数 | 需要变成形态和黑灰之后的独立阶段 |
| 背景处理 | 旧字清除和背景融合仍有混合 | 需要前置清除和后置融合分离 |
| 视觉模型 | 已用于排序和最终验收 | 只能看本地 top candidates，不能覆盖本地门禁 |

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

## Done Definition

该设计完成不能只看某一张图效果。必须满足：

- 自动 ROI 失败时立即报错或保留 rejected candidate。
- 旧槽位不完整时不生成最终候选。
- 同字数 CJK 能在左上贴齐和中心优先之间自动选择。
- 字形变化大时，候选报告能说明为什么触发中心优先。
- `text_shape` 未通过时，流程不会先调照片质感或背景融合。
- 视觉模型只评估本地 top candidates，不能覆盖本地硬门禁。
- 失败也有足够中间产物供用户检查。
