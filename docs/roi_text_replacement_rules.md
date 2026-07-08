# ROI Text Replacement Rules

本文档记录本项目在对话和回归中沉淀下来的处理规则。它不是聊天纪要，而是配合代码一起工作的工程规则：后续修改 `src/roi_image_edit/processing_service.py`、`src/roi_image_edit/iterative_pipeline.py`、`src/roi_image_edit/stage_policy.py`、CLI 或 Web 流程时，应先对照这些规则。

ROI 替换 workflow 的不可折中实施 checklist 见
[`docs/workflow_checklist.md`](workflow_checklist.md)。
文字形态门禁、放置策略、现有差距和分层联合优化设计见
[`docs/text_shape_joint_optimization_design.md`](text_shape_joint_optimization_design.md)。
当本文档中的现有经验值或旧规则与该 checklist 冲突时，以 checklist 中的原图参照、阶段求解和动态门禁要求为准。

## 目标边界

1. 这是“在原图上修改 ROI”，不是重新生成整张图。
2. 输出图片尺寸必须与原图一致。
3. ROI 外像素必须保持不变。
4. 图片边缘必须保持不变。
5. 受保护文字，例如本次动态上下文中的 `field_label_text`、`field_separator_text`、`protected_texts`、日期/年龄/编号等字段标签，不能被修改。
6. 如果自动定位不到要改的旧文字或字段，应立即报错或保留 rejected candidate 产物，不能静默输出看似成功的原图。
7. 每张输入图都必须先自动分类，再按分类结果选择场景、内部执行策略、prompt pack 和参数范围。
8. Profile 只能作为后端内部策略，由分类结果推导；Web 前端不能把 profile 暴露为用户需要理解或选择的控件。
9. 字数可能变化，但新文字不能覆盖旧文字后面的未修改内容；需要时可自动扩大实际编辑区域。字数增加不能因为用户原始框选空间不足或右侧间距不足，在候选生成前直接失败；最终仍必须由 hard report 验证 protected text、ROI 外像素和图像边缘未被改变。

## 基本流程

固定流程如下：

1. 对每张输入图自动分类，输出 `image_type`、`scenario`、`script`、`length_change`、`roi_input`、`class_key`、`confidence` 和 evidence。
2. 解析用户指令，识别字段、旧文字和新文字，并把字数变化写入分类/场景上下文。
3. 根据分类结果选择内部策略、prompt pack 和参数范围，并把执行名记录为 `internal_profile`；同一批图片逐图独立分类，不能共享上一张图的失败原因、候选或模型建议。
4. 对照片方向做必要校正，再进行场景化 ROI 规划。用户手动画框时，先判断它是 `manual_exact` 还是 `manual_anchor`；`manual_anchor` 只能作为 search/anchor ROI，不能直接当作最终编辑范围。
5. 定位旧文字槽位并生成实际 `edit_roi`；字数增加时可生成 `expanded_edit_roi`，不能用缩字、压字距或候选前右边界阻断来解决空间不足。
6. 从旧文字提取槽位、灰度、字体风格、基线、间距和姿态参考。
7. 本地生成候选图，只在实际编辑 ROI 内修补背景和重绘目标文字。
8. 先执行形态阶段：字体、字号、字槽、基线、笔画粗细、局部倾斜/扭曲。
9. 形态阶段通过后，才执行墨色和灰阶阶段：真黑核心、中间灰阶笔画、外层灰边。
10. 墨色阶段通过后，才执行照片质感阶段：轻微模糊、断裂、噪声、压缩质感。
11. 最后执行背景融合和残影修复阶段。
12. 输出本地硬校验报告、分类报告、ROI plan、stage evidence、`progress.jsonl` 和 `result.json`。
13. 视觉模型按阶段 prompt 做候选排序和最终验收。
14. 视觉模型只能返回 JSON 参数建议，不能直接覆盖本地硬校验和阶段门禁。
15. 脚本根据 JSON 和本地指标做小步调参。
16. 最多迭代 N 轮，默认 8 轮。
17. 最终必须同时通过硬校验、阶段门禁和视觉验收；未通过时仍保留 rejected candidate、候选对比图和报告。

## 阶段门禁顺序

候选排序和最终验收必须使用同一阶段顺序：

1. `hard_boundary`：尺寸不变、ROI 外不变、边缘不变、受保护文字不变。
2. `text_shape`：字体结构、字号、高度、单字槽位、字距、基线、笔画粗细/身体、局部倾斜和拍照姿态。
3. `ink_gray_balance`：`<55` 真黑核心、`55-120` 中深笔画、`120-165` 外灰边。
4. `photo_texture`：扫描/拍照模糊、边缘断裂、噪声和压缩质感。
5. `background_cleanup`：旧字残留、涂抹、发白、背景纹理断裂。

旧槽位清除不是最后才开始的 stage。完整旧槽位、旧字核心/灰边覆盖、多余旧槽位清理属于候选生成前的安全前提；`background_cleanup` 主要负责最终候选周围的背景融合、残影、涂抹和纹理断裂验收。

## 阶段和优化步骤

阶段是本地 gate，优化步骤是阶段内部的候选生成、搜索或参数补丁。二者不能混用。
阶段顺序、阶段名称和 Optimization Step 策略定义在 `src/roi_image_edit/stage_policy.py`；Web 入口只能导入这些策略，不应在 `web_app.py` 中重新定义。
Web 入口只负责 HTTP/API/job 状态；处理编排集中在 `src/roi_image_edit/processing_service.py`。ROI 定位属于 `src/roi_image_edit/roi_locator.py`，本地验收和候选评分属于 `src/roi_image_edit/local_validation.py`，修订求解器属于 `src/roi_image_edit/revision_solver.py`。

| Stage | 目的和作用 | 主要 Optimization Steps | 视觉 prompt |
| --- | --- | --- | --- |
| `hard_boundary` | 保证尺寸、ROI 外、边缘和 protected text 不变；方向、字段 ROI、旧槽位不可靠时阻塞。protected text guard 是全方向约束，目标 ROI、旧字清理范围和实际改动像素都不能覆盖任何未修改文字。 | `orientation_check`、`field_roi_selection`、`slot_quality_gate`、`protected_text_guard`、`hard_check` | 视觉模型只能读取 hard report；不能覆盖失败。 |
| `text_shape` | 先修字体、字号、槽位、行基线、字距、笔画身体和局部姿态。行基线由旧文字槽位主导，并用同一行未修改文字作上下文约束，不能只按 ROI 中心放置。 | `placement_strategy`、`shape_change_detection`、`font_style_search`、`font_size_search`、`slot_alignment_search`、`row_baseline_check`、`stroke_body_search`、`pose_shear_search`、`shape_reset` | `candidate_rank_prompt.txt` 排序 top candidates；`tuning_prompt.txt` 给 JSON 建议；`final_acceptance_prompt.txt` 终检。 |
| `ink_gray_balance` | 分开控制真黑核心、中灰笔画身体和外灰边。 | `core_black_search`、`mid_gray_body_search`、`outer_gray_control`、`opacity_search`、`core_gain_search`、`alpha_contrast_search` | 同上，但建议必须限制在黑灰相关参数。 |
| `photo_texture` | 匹配照片/扫描的模糊、断裂、噪声和压缩质感。 | `blur_match`、`edge_breakup_match`、`noise_texture_match`、`jpeg_texture_match`、`residual_retexture` | 同上，但只能在形态和黑灰通过后主导。 |
| `background_cleanup` | 验收旧字残影、涂抹、发白、发暗、背景纹理断裂和接缝。 | `old_slot_cleanup_check`、`ghost_residual_repair`、`shadow_residual_repair`、`background_texture_repair`、`seam_gradient_repair` | `candidate_rank_prompt.txt` 可指出补丁感；`final_acceptance_prompt.txt` 终检自然度。 |

所有视觉 prompt 使用 `master_prompt.txt` 作为 system prompt。Web 路径当前使用 `candidate_rank_prompt.txt` 和 `final_acceptance_prompt.txt`；CLI 迭代路径还会使用 `tuning_prompt.txt`。
视觉 prompt 的输入必须包含当前候选或候选集合的 `stage_context`。模型可以指出其它阶段
的视觉问题，但 `suggested_patch` 和 `parameter_suggestions` 必须受当前
`allowed_patch_keys` 约束；本地 `stage_filter` 会拒绝越过阶段边界的建议，并把冲突写入
`revision_rounds.model_suggestion_filter.attempt_records` 和
`revision_rounds.model_conflicts`。
不能转成本地 patch 的视觉建议不能静默丢弃，必须在 attempt record 中记录
`rejection_reason`。
视觉 response 必须包含 `stage_assessment`，声明本地 `blocking_stage` 是否存在、
建议目标阶段和判断依据；本地会把这份契约写入
`revision_rounds.model_stage_response_contracts`，用于识别跨阶段建议。

动态 prompt 必须嵌入本次任务真实字段上下文：`field_key`、`field_label_text`、
`field_separator_text`、`protected_texts` 和 `protected_boxes`。静态 prompt 模板不能写死
某个字段标签、某个默认参考字、固定字号枚举或固定 opacity/blur 建议值。字段标签和
标点只来自指令解析、自动 ROI evidence 或后续 OCR/检测结果；如果上下文为空，模型不能
自行补全成固定字段。

如果 `text_shape` 存在 hard-blocking issues，后续调参只能先处理字体、字号、描边/笔画身体、字槽偏移和姿态继承；不能先通过降黑、加模糊、加噪声或背景修补来掩盖形态问题。

如果 `text_shape.pass=true` 但 `deferred_issues` 非空，说明这些诊断主要受黑灰阶段影响，例如黑芯过量导致中灰笔画身体不足，或轻微近阈值字符中心偏移会在黑灰调整后重新评估。此时不能让 `text_shape` 永久阻塞后续阶段；应进入 `deferred_to_stage`，通常是 `ink_gray_balance`。后续阶段候选仍必须保证字体结构、字号、槽位、基线、字距和严重位置指标不回退。

`text_shape` 的判断不能因为同时存在黑芯过量而被隐藏。若一个候选既偏黑又存在字体结构、字号、槽位、基线、字距或严重姿态问题，应继续把阻塞阶段记为 `text_shape`。若剩余问题主要是笔画身体/中灰层不足、邻字核心密度不一致，且本地已判定黑芯过量，则应把这些问题记录为 `text_shape.deferred_issues`，允许 `ink_gray_balance` 成为当前阻塞阶段。

当 `text_shape` 阻塞时，求解流程必须重新生成形态候选：主搜索键只包含字体排名、字号、字槽偏移、基线、放置策略和局部姿态继承；`stroke_opacity` 只能作为次级 `stroke_body_shape` 轴。放置策略必须按场景限制枚举，并按 `font × placement_strategy` 分桶配额保留候选，不能被全局 top-N 截断吞掉。字体排名必须记录旧 ROI、邻字/标签和 protected text 的本地可解释 `style_profile`、参考质量和权重；弱参考只能降权，不能强行锁错字体族；数字/日期节奏必须单独记录。OCR/MMOCR/PaddleOCR 只辅助文字真值和槽位，不参与字体风格裁决。不能只在当前候选上累加 `core_ink_gain`、`ink_gain`、`alpha_contrast`、`blur`、`photo_warp` 或照片噪声。

当 `text_shape` 阻塞且本地已经判定黑芯严重过量时，允许生成单独标记的
`ink_guard` 候选。`ink_guard` 不是把降黑并入 `text_shape`，而是在形态仍为主阶段时执行横向保护：
只能调整 `opacity`、`ink_gain`、`alpha_contrast`、
`core_ink_gain`、`core_darken_strength`、`core_darken_threshold` 和
`core_darken_target_gray`，不得改字体、字号、`stroke_opacity`、位置、字槽、mask、背景或照片质感。
`ink_guard` 候选只有在 `text_shape` 严重度不回退且 `ink_gray_balance` 严重度下降时才可进入下一轮。

`photo_texture` 不能无条件通过。形态和墨色通过后，仍需用本地指标比较原图与候选在修改文字附近的边缘拉普拉斯、高频残差、模糊、边缘断裂、噪声和 JPEG 压缩权重；若文字过锐、过干净或过糊，应继续在照片质感阶段调参。

## 字段和 ROI 规则

1. 支持姓名、日期、年龄等常见字段的自动 ROI。
2. 用户手动画框时，框可以包含少量空白，但必须先分类为 `manual_exact` 或 `manual_anchor`；`manual_anchor` 只能作为 search/anchor ROI，后端仍需重新定位旧值槽位并生成独立 `edit_roi`。
3. 自动 ROI 应先在用户框或自动字段附近找旧文字深色组件，再将实际编辑范围收缩到旧文字槽位和必要空白。
4. 如果旧文字某个笔画超出初始框，应把目标掩码覆盖到完整旧字槽位，避免旧字残留。
5. 对于相同字数替换，优先按字符槽位重绘；对于字数减少，需要清理多余旧槽位；对于字数增加，已有旧字槽位不能被压缩，新增字应从旧值最右侧位置继续追加，并且不能覆盖后续文本。
6. 字数增加时，`right_boundary` 和 protected distance 是扩框诊断和最终验收依据，不是候选生成前的空间不足阻断器；必须记录扩展方向、扩展幅度和最终 protected diff。
7. 当文本等长、只有少数字符变化、且目标字符的原图字形已存在于同一 ROI 内时，可使用本地图内字形复用策略 `source_glyph_reuse`：先用字符组件槽位定位旧值，擦除变化字符，再从同一 ROI 内复制目标字符字形贴回。该策略是处理方式/patcher，不是新的图片分类；必须记录 `source_glyph_reuse_report`，并验证 diff 只发生在目标字符擦除/粘贴框内，不能改变 protected text、邻近未修改字符或 ROI 外像素。

## 字体规则

1. 不固定使用某一个字体。字体选择应由旧文字 ROI 的风格评分、候选字体可用性和视觉验收共同决定。
2. 不预设字体类别优先级；每次应根据旧文字 ROI 的结构、笔画粗细、边缘形态、候选可用性和视觉验收排序。
3. 项目本地 `fonts/`、用户字体目录和系统字体都可以参与候选，但字体必须能真实渲染所有源文字和目标文字。
4. 字体差异不能只交给视觉模型判断；本地 font style gate 必须保留。
5. 当视觉模型指出字体相似度为 `slightly_off` 或 `wrong_style`，后续轮次应尝试更接近旧文字 ROI 风格的字体或字体参数，而不是只调颜色。

## 黑度、粗细和灰边规则

1. 粗细、黑度、清晰度是三个不同问题，不能混为一个方向。
2. `too_dark` 不等于 `too_bold`，`too_light` 不等于 `too_thin`。
3. 对照片件，原文字通常不是纯黑矢量字，而是有深黑核心、灰色边缘、断裂和拍摄模糊。
4. 本地必须分别统计：
   - `<55` 真黑核心；
   - `<70` 深色核心；
   - `70-90` 中间灰阶；
   - `90-120` 暗灰过渡；
   - `120-165` 外层灰边。
5. 不能用大量灰边假装笔画变粗，也不能只把核心压黑导致横画成块。
6. 如果本地 `local_ink_balance_issues` 指出 `changed_char_core_too_black` 或 `roi_core_too_black`，下一轮必须优先降 opacity、ink gain、core ink 或 core darken，而不是走“加墨补灰边”。
7. 如果核心深色不足但灰边很多，应优先恢复核心黑度，而不是继续加模糊。
8. 如果核心黑度合格但视觉仍说太锐，应优先增加照片质感、轻微 blur、noise、JPEG 退化或边缘破碎，而不是继续降黑。
9. 当同一行存在未修改的中文邻字时，应额外比较新字与最近邻字的槽位面积归一化 `<55`/`<70` 核心密度和 `120-165` 外层灰边占比。若目标字复杂度相近但核心密度明显低、外层灰边明显高，说明新字靠灰雾撑厚，不能通过验收。
10. 当旧字槽位指标与同一行保留邻字指标冲突时，例如相对旧字显得核心增量大、但相对邻字核心仍偏低，应优先采用邻字风格参照；不能在“加实核心”和“降黑”之间来回摆动。
11. 当核心密度已经接近邻字，但 `120-165` 外层浅灰占比和密度仍明显高于邻字时，应判定为外圈灰雾/底部灰边过多；下一轮应压低 blur、photo_noise、edge_breakup 和描边外扩，同时用核心增益维持主笔画。
12. 如果 `local_ink_balance_issues` 已经结合邻字参照判定为空，后续 fallback 不应再按旧字槽位把同一候选改判为过黑。
13. 当目标字比旧字复杂，且同一行邻字风格门槛已通过时，旧槽位的 `70-120` 中间灰阶缺口不能单独作为硬拒绝；否则会把清理外圈灰雾后的候选误判成笔画太窄。
14. 当本地阶段已通过但视觉模型仍指出 `too_dark` / `too_bold`，后续补丁不能把 `opacity`、`stroke_opacity`、`core_ink_gain` 或 `core_darken_strength` 拉回更黑方向；应优先小幅降低 `opacity`/`alpha_contrast` 或轻微软化，直到本地门禁和视觉验收同时通过。
15. 当唯一或主要失败项是 `roi_core_too_black` / `changed_char_core_too_black` 且超限很小，应进入过黑临界微调，而不是继续使用常规 ink-gray 网格的粗步长。
    - 临界条件必须写入报告，例如 `near_threshold=true`、`metric`、`actual`、`limit`、`gap`、`gap_ratio`。
    - 微调候选应覆盖 `opacity -0.003/-0.006/-0.010/-0.020`、`core_ink_gain -0.003/-0.006/-0.010/-0.020`、`core_darken_strength -0.003/-0.006/-0.010`、`alpha_contrast -0.003/-0.006` 及少量二元组合。
    - 如果微调会让 `text_shape`、protected text、旧字残留或背景指标回退，必须拒绝并记录原因。
16. 当目标文字字符数或渲染复杂度高于旧文字时，黑芯和中灰阈值必须使用复杂度归一化说明；不能把全 ROI `<55` 像素增量直接等同为“过黑”。
    - 报告中应保留 `source_text_count`、`target_text_count`、`text_count_ratio`、`source_complexity`、`target_complexity`、`complexity_ratio` 和使用该比例后的动态阈值。
    - 复杂度归一化只能放宽质量阈值，不能放宽 ROI 外像素、protected text、旧字残留和边界硬约束。
17. 如果候选已经只差极小质量阈值，但视觉上仍指出过黑/过硬，本地 solver 应优先生成“降黑且不改变形态”的候选；不能通过继续加模糊、加噪声或改字体来绕过黑灰门禁。

## 照片质感规则

照片件的自然感来自多项小处理组合，而不是单一滤镜：

1. 背景修补必须保留原图的局部纹理和亮度，不应出现纯白、平滑涂抹或明显补丁。
2. 新文字应使用轻微模糊模拟拍摄/扫描，不应像清晰打印到图上。
3. 可使用 `photo_warp` 做小幅局部形变。
4. 可使用 `edge_breakup` 做轻微边缘断裂和锯齿感。
5. 可使用 `photo_noise` 将原图局部残差和随机噪声加回文字附近。
6. 可使用局部 JPEG 退化模拟压缩质感。
7. alpha 重采样只能作为高退化候选使用，不能默认套在所有候选上，否则容易变成灰蒙蒙。

## 姿态继承规则

对于照片中的小字，局部姿态比整行角度更重要。

1. 被替换字优先参考对应旧字槽位。例如 `旧文字 -> 新文字` 时，第 n 个目标字符优先参考第 n 个旧槽位的姿态。
2. 如果对应旧字槽位估计不稳定，可参考同一行相邻未替换字或标签字符作为约束。
3. 姿态方向必须由旧槽位和邻字估计得到，不能把某张图的“左倾/右倾”固化成通用规则。
4. 姿态继承只做小幅局部 shear，不做大幅旋转或重新排版。
5. 旧字的字形结构可能误导姿态估计，因此应用强度必须小于估计强度并设置上限。
6. 姿态继承结果必须写入报告：
   - `source_slot_shear`
   - `neighbor_shear`
   - `reference_shear`
   - `applied_shear`
7. 如果应用姿态后造成位置漂移、黑度变化或字体评分下降，必须让完整候选流程重新选择参数，不能沿用旧最终参数。

## 视觉模型规则

1. 视觉模型负责视觉排序和验收，不负责像素级硬校验。
2. 视觉模型可能把“黑、硬、淡、粗、细”的方向说反；本地指标必须能纠偏。
3. 视觉模型返回的 JSON patch 只能作为建议，必须经过本地范围限制和硬校验。
4. 当视觉模型要求 `opacity=0.86`、`blur=0.44` 等具体参数时，本地候选选择器应允许探索这些方向，但仍不能破坏硬指标。
5. 如果本地 `blocking_stage=null` 或本地阶段全部通过，但视觉模型仍返回 `revise`/`marginal` 并指出具体问题，流程必须记录 `vision_disagreement`，把问题映射成现有五阶段之一的 `vision_target`；这不是第六个公开 stage。
6. `vision_target` 只能表达方向和受限目标区间，不能把模型给出的单个参数值当作必须执行的最终参数。
7. 如果视觉模型 `pass` 但本地发现深黑核心过量、灰边过多、ROI 外变化或字体风格失败，必须改成 `revise`。
8. 如果视觉模型持续 `revise`，最终输出 rejected candidate，并保留每轮候选图、`final_acceptance_iterXX.json`、`vision_target` 和候选拒绝原因。
9. 如果视觉模型在上一轮说过黑，而本地补丁把候选变得更黑，候选选择器必须惩罚这种回退；不能因为字体、字号或细化分支触发旧规则而覆盖当前视觉方向。
10. 如果同一个视觉问题连续出现两轮以上，候选选择器必须把它升级为受限目标：朝该方向改善的候选应被优先比较，反方向候选必须记录惩罚或拒绝原因。
11. 对 `too_sharp + patch_visible`、`too_dark + too_sharp`、`patch_visible + white_specks` 等组合问题，只允许使用有预算、有 primary/secondary stage 声明的组合候选，不能恢复全量跨阶段搜索。
12. 视觉模型给出的 `suggested_patch` 或 `parameter_suggestions` 只要能转换为本地参数，就必须生成至少一个 forced seed candidate。
   - forced seed 可以被 stage filter、internal-strategy filter、constraint 或 prior-stage regression 拒绝，但拒绝必须写入 `revision_rounds[].forced_model_candidates` 或等价产物。
   - 如果模型建议被去重合并，应记录合并到哪个 candidate id；不能静默丢失。
13. 视觉模型不能承担连续参数控制器职责。模型负责指出可见问题和候选偏好；本地 solver 负责把建议变成候选、评估指标、选择下一轮或解释为什么停止。
14. 候选排序 prompt 和最终验收 prompt 不应要求模型同时完成审美排序、阶段仲裁和参数求解三件事；参数求解必须在本地报告中有确定性搜索和拒绝证据。

## 调参规则

1. 每轮只做小步参数变化，避免无法判断差异来源。
2. 如果候选已经通过当前阻塞阶段，但被后续阶段阻塞，必须把它作为可选推进候选进入下一轮；不能因为最终仍未验收就停在 `no_selectable_revision_candidate`。
3. 当问题是核心过黑：
   - 优先降低 `opacity`、`ink_gain`、`core_ink_gain`、`core_darken_strength`；
   - 可小幅增加 `blur` 或照片质感；
   - 不应增加 `stroke_opacity` 或大幅加墨。
   - 若同阶段候选的黑芯 severity 明显下降，即使尚未一次性通过 `ink_gray_balance`，也应允许进入下一轮继续小步迭代。
   - 若黑芯超限已经接近阈值，应使用 `near_threshold_overblack_micro_tuning`，并把常规网格、模型 forced seed 和微调候选分开计数、分开记录拒绝原因。
4. 当问题是核心不够黑：
   - 优先增加 `core_ink_gain` 或 `core_darken_strength`；
   - 不应直接使用粗体或黑体替代。
   - 如果只剩 `core_mean_gray_too_light` / `core_lighten_too_high` 且超限很小，应进入核心微调：只做 `0.003~0.006` 级别的 `core_ink_gain`、`core_darken_strength`、`alpha_contrast` 或极小 `opacity` 调整，不改变字体、字号、位置、模糊、mask 或背景参数。
5. 当问题是笔画不够粗但核心并不缺黑：
   - 先比较旧槽位和新字的 `<55`、`70-90`、`90-120`、`<165` 像素分布；
   - 如果 `<55` 核心已经接近或偏多，但 `70-90 + 90-120` 中间灰阶不足，应优先调整 `ink_gain`、`alpha_contrast` 和核心参数；若必须改变 `stroke_opacity`，应显式退回 `text_shape` 的次级 `stroke_body_shape`；
   - 如果目标字比旧字更复杂，不能机械使用同字数 `dark_pixel_ratio <= 1.12`，应按本地渲染复杂度小幅放宽；
   - 更复杂目标字的 `<55` 核心增量门槛也应小幅放宽，但必须有上限，并继续受邻字核心密度和视觉验收约束；
   - 不能只依靠 `120-165` 浅灰边增加来通过验收，`70-120` 中间笔画主体仍要接近旧槽位；
   - 如果同一行保留邻字的核心更实，应优先用邻字风格门槛触发轻微描边、核心密度恢复和减少灰雾，而不是只继续参考被替换旧字；
   - 一旦触发邻字核心密度问题，本轮补丁应禁止继续增加 `blur`、`photo_noise` 或 `edge_breakup`，优先小幅增加 `core_ink_gain`、`core_darken_strength` 或降低 blur；需要改变 `stroke_opacity` 时必须退回 `text_shape`；
   - 一旦触发外圈灰边过多问题，本轮补丁应允许调整 `alpha_contrast`、降低 `blur`、`photo_noise`、`edge_breakup`，并用 `core_ink_gain`/`core_darken_strength` 保住核心，不应继续扩大浅灰边；`stroke_opacity` 不在黑灰阶段改变；
   - 外圈灰边过多时，可小幅增加 `alpha_contrast`，把浅灰抗锯齿边缘收紧为更干净的中深色边缘或背景，而不是简单糊化；
   - 如果清灰边后邻字核心密度和外圈灰边都合格，应允许旧槽位 `70-120` 中间灰阶存在合理缺口，并交给最终视觉验收判断是否过硬；
   - 如果局部细笔画仍偏细但灰边已经很多，应优先轻微描边并小幅降低 `blur`/灰边噪声，不能继续用灰雾撑厚；
   - 不应继续单纯增加 `core_ink_gain`，否则会变成黑硬但笔画身体仍窄。
5. 当问题是边缘太清晰：
   - 优先增加 `blur`、`photo_noise`、`edge_breakup`、压缩质感；
   - 不应只降低透明度造成灰蒙。
6. 当问题是位置或基线：
   - 优先调整 per-character offsets；
   - 不能通过扩大 ROI 或整体居中掩盖槽位错误。
   - 垂直位置修正必须参考 `char_alignment_metrics.center_dy` 回到旧槽位中心线；不能把“看着偏上”机械处理成固定下移多像素，避免从偏上直接变成偏低。
7. 当问题是旧字残留：
   - 优先修正旧字掩码和槽位覆盖；
   - 候选灰边残留必须参考候选局部背景动态阈值，不能把拍照灰底或新字低 alpha 抗锯齿边缘当作旧残留；
   - 旧字核心残留仍然硬失败，不能因为动态灰边阈值而放过；
   - 不应通过加深新字遮盖残留。

## 产物和进度规则

每次 CLI/Web 处理都应可追踪：

1. `progress.jsonl` 记录阶段、候选数、每轮是否接受和分数。
2. `result.json` 记录最终参数、硬校验、视觉验收和每轮 revision。
3. 每个区域保留：
   - `vision_candidate_sheet.png`
   - `vision_final_compare.png`
   - `vision_final_compare_iterXX.png`
   - `final_acceptance_iterXX.json`
4. 未通过验收时，最终图可以展示 rejected candidate，但必须标记 `accepted=false` / `applied=false` 或等价状态。
5. Web 候选 drawer 至少显示最近或最有代表性的候选，便于用户观察处理过程。
6. 如果停止原因是 `no_selectable_revision_candidate`、`no_stage_specific_candidate_direction` 或某个 stage severity 无法继续下降，必须额外写出候选拒绝表：
   - 每个候选的 `candidate_id`、`origin`、`primary_stage`、`optimization_step`、`strict_pass`、`stage_pass`、`blocking_stage`、当前阶段 severity before/after、prior regression 和不可选原因。
   - 模型 forced seed 候选必须能从视觉建议追溯到渲染候选或拒绝记录。
   - UI 可以只展示摘要，但 `result.json` / `progress.jsonl` / region artifact 必须保留完整诊断。

## 代码位置映射

| 规则关注点 | 主要代码位置 | 主要报告字段 |
| --- | --- | --- |
| ROI 外不变、边缘不变 | `hard_check` | `outside_roi_changed_pixels`, `border_changed_pixels` |
| 字符槽位和行基线 | `dark_runs`, `build_region_plan`, `char_alignment_metrics`, `row_baseline_metrics` | `slot_boxes`, `char_alignment_metrics`, `row_baseline_metrics` |
| 字体风格 | `font_style_gate`, `build_font_style_reference` | `font_style_gate` |
| 黑度和灰边 | `strict_visual_metrics`, `char_gray_band_metrics`, `local_ink_balance_issues` | `strict_visual_metrics`, `char_gray_band_metrics` |
| 邻字风格 | `local_neighbor_style_issues`, `local_outer_gray_halo_issues`, `neighbor_core_density_recovery_patches`, `neighbor_outer_gray_cleanup_patches` | `local_neighbor_style_issues` |
| 照片质感 | `photo_texture_metrics`, `local_photo_texture_issues`, `photo_texture_recovery_patches`, `apply_photo_alpha_warp`, `apply_scan_edge_breakup`, `apply_photo_text_texture` | `photo_texture_metrics`, `local_photo_texture_issues` |
| 姿态继承 | `estimate_slot_edge_shear`, `reference_slot_shear`, `apply_char_slot_shear` | `char_pose_metrics` |
| 阶段门禁 | `stage_gate_for_report`, `stage_selection_penalty`, `report_stage_pass` | `stage_gate` |
| 运行产物结构 | `request_audit_payload`, `result_audit_payload`, `stage_progress_fields`, `model_stage_context`, `attach_stage_context_to_rank_report` in `run_artifacts.py` | `request.json`, `result.json`, `progress.jsonl`, `stage_context_by_candidate` |
| 形态重搜 | `text_shape_reset_candidates`, `shape_font_items`, `normalized_offset_candidates` | `revision_attempts[].origin=shape_reset`, `revision_rounds[].shape_reset_count` |
| 形态阶段墨色保护 | `ink_gray_candidate_grid(... allow_text_shape_guard=True)`, `text_shape_ink_guard_selectable` | `revision_rounds[].ink_guard_candidate_grid`, `revision_rounds[].ink_guard_count`, `revision_attempts[].origin=ink_guard_grid`, `revision_attempts[].ink_guard` |
| 迭代补丁 | `STAGE_PATCHER_SPECS`, `select_stage_patcher`, `dispatch_revision_patches`, `stage_patcher_registry_report`, `stage_patch_filter_report`, `revision_patches_for_round`, `black_core_reduction_patches`, `gray_stroke_recovery_patches` | `revision_rounds`, `revision_attempts`, `stage_optimization_policy`, `stage_patcher_dispatch`, `stage_filter_report`, `rejected_local_patches` |
| 视觉建议仲裁 | `filter_model_patch_records`, `model_suggestion_filter_report` in `model_suggestions.py` | `revision_rounds[].model_suggestion_filter`, `revision_rounds[].model_suggestion_attempts`, `revision_rounds[].model_conflicts` |
| 最终验收 | `evaluate_final`, `apply_local_acceptance_gate` | `final_acceptance` |

## 回归检查建议

修改核心流程后，至少跑一次 CLI：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py process \
  --image /path/to/input.jpg \
  --instruction '字段旧文字修改为新文字' \
  --output output/debug_regression_result.png
```

检查内容：

1. `accepted` 是否符合实际质量。
2. `strict_gate` 是否通过。
3. `local_ink_balance_issues` 是否为空。
4. `char_pose_metrics` 是否记录了被替换字的姿态继承。
5. 放大图中是否存在旧字残留、涂抹、过黑、过淡、过清晰、字距错误或基线错误。
