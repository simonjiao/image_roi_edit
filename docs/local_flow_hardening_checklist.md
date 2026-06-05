# Local Flow Hardening Plan and Checklist

本文档沉淀本地 ROI 文字替换流程的强化方案。目标不是继续增加经验参数，而是把流程升级为：

```text
原图参考画像
-> 阶段化本地求解
-> 本地指标主导候选选择
-> 视觉模型阶段验收
-> 可解释迭代和失败交付
```

本方案不接受折中交付。若候选没有通过硬校验、阶段门禁和最终视觉验收，只能作为 rejected candidate 展示，不能标记为成功应用。

文字形态门禁、放置策略、现有差距和分层联合优化设计见
[`docs/text_shape_joint_optimization_design.md`](text_shape_joint_optimization_design.md)。当本 checklist 与该设计文档描述同一能力时，checklist 记录实施状态，设计文档记录目标结构和实施顺序。
本文件同时作为“完善本地流程”的唯一实施 checklist，不再另建独立覆盖清单。

## Stage Refactor Execution Checklist

本节跟踪 `staged_roi_pipeline_design.md` 中“阶段化流程”从设计到代码的实施状态。完成标准不是某张图看起来更好，而是代码中存在清晰模块边界、每阶段有可验证输入输出、失败时有可追踪证据。

### Slice 1: 可执行 stage/profile 结构

- [x] 新增 `src/roi_image_edit/stages.py`，定义 `StageSpec`、`StageResult` 和 detector mapping。
- [x] 新增 `src/roi_image_edit/stage_profiles.py`，定义 `photo_scan`、`clean_digital`、`low_res_thumbnail`、`manual_roi_quick`。
- [x] `local_validation.stage_gate_for_report()` 委托 `stages.py`，报告包含 `profile`、`stage_status`、`allowed_patch_keys`、`blocked_patch_keys`。
- [x] CLI 和 Web payload 支持用户指定 profile，并写入 `result.json` / `progress.jsonl`。

验证：

```bash
.venv/bin/python -m compileall -q src scripts
.venv/bin/python - <<'PY'
from roi_image_edit.stage_profiles import stage_profile_choices
from roi_image_edit.stages import stage_specs
print(stage_profile_choices())
print([spec.id for spec in stage_specs("photo_scan")])
PY
```

### Slice 2: Stage patcher 调度

- [x] 新增 `src/roi_image_edit/stage_patchers.py`。
- [x] `revision_patches_for_round()` 缩成 stage dispatcher。
- [x] 每个 patcher 只生成当前 stage 允许的参数族。
- [x] 模型 JSON patch、rank patch、本地 patch 都经过同一个 stage filter。
- [x] patcher filter 输出 `stage_id`、`optimization_steps`、`allowed` / `rejection_reason`。

验证：

```bash
.venv/bin/python - <<'PY'
from roi_image_edit.stage_patchers import patch_allowed_for_stage
print(patch_allowed_for_stage({"photo_noise_delta": 0.02}, "text_shape")["allowed"])
print(patch_allowed_for_stage({"stroke_opacity_delta": 0.02}, "text_shape")["allowed"])
PY
```

### Slice 3: 前置 slot quality gate

- [x] 新增 `slot_quality_report`。
- [x] 候选生成前检查旧值槽位数、槽位完整性、底部/灰边覆盖、protected text 冲突。
- [x] 旧槽位不完整时直接返回 rejected，不进入字体/墨色/照片质感候选。
- [x] 字数减少时，多余旧槽位纳入前置清除报告；字数增加时，右边界受 protected text 限制。

验证：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py process \
  --image <image> \
  --instruction '<字段旧值修改为新值>' \
  --json
```

输出必须包含 `summary.plan.slot_quality_report`。报告中的
`length_change_report.extra_source_slots_for_cleanup` 记录字数减少时必须清理的旧槽位；
`length_change_report.right_boundary` 记录字数增加时右侧 protected text 对可用空间的限制。
失败时必须有 rejected candidate 和明确 `slot_quality_failed` 原因。

### Slice 4: 放置策略和单字形态变化检测

- [x] 新增 `placement_strategy` 字段，至少支持 `top_left_anchor`、`center_primary`、`left_anchor_span`、`baseline_numeric`、`manual_fallback`。
- [x] 新增单字形态变化检测，报告 `bbox_width_delta_ratio`、`bbox_height_delta_ratio`、`centroid_dx/dy`、`ink_area_ratio`。
- [x] 同字数 CJK 旧字和新字不同时自动切到 `center_primary`，并记录选择原因；`shape_change_report` 继续逐字验收实际偏差。
- [x] `text_shape` 未通过时，禁止 blur/noise/JPEG/background 成为主修复方向。

验证：`candidate_report` 中必须能看到 `placement_strategy` 和 `shape_change_report`；当 `shape_change_large=true` 时，候选生成必须包含中心优先候选。

### Slice 5: 分层候选产物和阶段证据

- [x] `progress.jsonl` 每轮写入 `pipeline_profile`、`stage_status`、`blocking_stage`、`allowed_patch_keys`、`blocked_patch_keys`。
- [x] `result.json` 写入最终候选的完整 stage evidence。
- [x] 保存 shape、ink-gray、photo texture、background cleanup 的 top candidate 或 rejected candidate 证据。
- [x] Web 候选抽屉展示 profile、stage、stage severity。
- [x] Web 候选抽屉展示候选来源、Optimization Step、模型建议和 patcher filter 信息。
- [x] 视觉 prompt 输入当前 stage context，输出建议不能越过本地 stage filter。

验证：一次失败任务也必须能从 `output/web/<run>/progress.jsonl`、`result.json`
和 `output/web/<run>/regions/<region>/stage_evidence/summary.json`
解释“卡在哪个阶段、为什么没有继续、下一轮应该调什么”。视觉排序输入必须在
`visual_eval_candidate_rank.json.local_stage_context` 中保留 `stage_context_by_candidate`。

## 设计目标转换情况

本节把 `staged_roi_pipeline_design.md` 和
`text_shape_joint_optimization_design.md` 中的目标拆成可实施、可验证的 checklist 项。
`[x]` 表示当前已有实施项或已经完成；`[ ]` 表示还需要代码、测试、报告或文档同步。
每个 `[ ]` 都必须能用测试、CLI/Web 输出、`progress.jsonl`、`result.json`、stage evidence
或 fixture 回归关闭，不能只靠文字说明关闭。

### A. 三层流程边界

- [x] 五个本地 stage 已写入本 checklist：`hard_boundary`、`text_shape`、`ink_gray_balance`、`photo_texture`、`background_cleanup`。
- [ ] 增加前置安全流程验收：`orientation_check`、`field_roi_selection`、`slot_quality_gate`、`protected_text_guard` 必须在候选生成前完成；验证方式是失败样例 `candidate_count=0` 或 rejected，且 `progress.jsonl` 记录失败步骤。
- [ ] 增加阶段门禁顺序验收：`src/roi_image_edit/stage_policy.py` 的 `STAGE_ORDER` 必须与本 checklist 的五阶段顺序一致；验证方式是单测读取常量并断言顺序。
- [ ] 增加阶段内 Optimization Step 验收：每个候选报告必须区分 `stage_id` 和 `optimization_step`，不能把 Optimization Step 当成新 stage；验证方式是 `result.json` 中同时存在两类字段。
- [ ] 增加视觉终检边界验收：视觉模型只能看本地 top candidates；验证方式是视觉请求记录中 `candidate_count <= vision_candidate_limit` 且包含本地 `stage_context`。

### B. 旧 7 类阶段术语映射到当前 5 个 stage

- [ ] `slot_alignment` 必须映射到 `hard_boundary` 的 ROI/slot 安全条件和 `text_shape.slot_alignment_search`；验证方式是 stage evidence 记录旧名、当前 stage、Optimization Step 和报告字段。
- [ ] `font_structure` 必须映射到 `text_shape.font_style_search`、`font_size_search`；验证方式是字体失败样例不会进入 `ink_gray_balance` 主调参。
- [ ] `pose_geometry` 必须映射到 `text_shape.pose_shear_search`，且不能固化某张图的左倾/右倾；验证方式是姿态报告来自旧槽位、邻字或局部投影指标。
- [ ] `stroke_body` 必须映射到 `text_shape.stroke_body_search`，且在真实笔画体量未过时不能被 `edge_quality` 或 `photo_texture` 抢先处理；验证方式是粗细失败样例的 `blocking_stage=text_shape`。
- [ ] `tone_gray` 必须映射到 `ink_gray_balance.core_black_search`、`mid_gray_body_search`、`opacity_search`；验证方式是黑芯过量和核心不足分别生成相反方向候选。
- [ ] `edge_quality` 必须拆到 `ink_gray_balance.outer_gray_control` 和 `photo_texture.edge_breakup_match`，并记录拆分依据；验证方式是灰边过量不会先破坏已通过的 stroke body。
- [ ] `photo_texture` 必须映射到 `photo_texture.blur_match`、`edge_breakup_match`、`noise_texture_match`、`jpeg_texture_match`、`residual_retexture`；验证方式是 `photo_texture` 只在形态和黑灰通过后成为 blocking stage。
- [ ] 更新所有 prompt、report、UI 文案中的旧 stage 名引用；验证方式是公开输出不再把旧 7 类阶段当成本地 gate。

### C. 全局硬约束

- [x] 输出尺寸与原图一致。
- [x] ROI 外像素不变。
- [x] 图片边缘像素不变。
- [x] protected text 不变。
- [ ] 增加“目标 ROI 覆盖完整旧字”独立验收：旧字核心、灰边、底部和倾斜外溢都必须在 source slot 或 cleanup mask 内；验证方式是 `slot_quality_report` 的逐项字段全部通过。
- [ ] 增加“新字不能覆盖后续未修改内容”独立验收：字数增加和 ROI 扩展时必须记录 `right_boundary`、protected box 距离和最小安全间距；验证方式是字数增加 fixture。

### D. 方向、字段和旧值 ROI

- [ ] 指令解析必须输出 `field`、`old_value`、`new_value`、解析置信和失败原因；验证方式是 CLI JSON 对姓名、日期、年龄、手动 ROI 四类输入都有字段。
- [ ] 自动方向选择不能只看整页方向；必须同时记录目标字段质量、旧值定位质量和最终方向理由；验证方式是旋转图片 fixture。
- [ ] 每个自动 ROI 任务必须同时记录 `search_roi` 和 `edit_roi`，并输出标注图；验证方式是 `stage_evidence` 中存在两个 ROI 的坐标和图片。
- [ ] `search_roi` 可以覆盖字段锚点和后续保护文本，`edit_roi` 必须收缩到旧值槽位和必要空白；验证方式是同一任务中 `search_roi` 面积大于等于 `edit_roi`，且 edit ROI 不含 label。
- [ ] 找不到字段或旧值时必须立即失败并保留 rejected 产物；验证方式是无目标字段 fixture 不能输出 `applied=true`。
- [ ] 自动 ROI 通用路径必须覆盖姓名、日期、年龄、数字编号和手动 ROI fallback；验证方式是每类至少一个 fixture 或 smoke command。

### E. 旧槽位完整性门禁

- [x] 输出 `slot_quality_report`。
- [ ] 逐字检查旧值字符数与槽位数匹配；验证方式是字数减少和字数增加 fixture 都记录 `source_count`、`target_count`。
- [ ] 逐槽检查核心笔画覆盖；验证方式是每个 slot 有 `core_coverage` 或等价字段。
- [ ] 逐槽检查灰边覆盖；验证方式是每个 slot 有 `gray_edge_coverage` 或等价字段。
- [ ] 逐槽检查底部覆盖；验证方式是底部裁切 fixture 不进入候选生成。
- [ ] 逐槽检查倾斜外溢覆盖；验证方式是倾斜旧字 fixture 的外溢像素纳入 source slot 或 cleanup mask。
- [ ] 检查槽位未混入字段标签、冒号或前置文本；验证方式是 label overlap 字段为 0 或低于动态阈值。
- [ ] 检查槽位未混入后续未修改文本；验证方式是 protected overlap 字段为 0 或低于动态阈值。
- [ ] 检查最后一个旧字不会被误判成 protected text；验证方式是最后字右下角外溢 fixture 能进入 cleanup mask。
- [ ] 字数减少时，多余旧槽位必须进入前置清除区域；验证方式是 `extra_source_slots_for_cleanup` 非空且有 mask 证据图。
- [ ] 字数增加时，右边界必须受后续 protected text 限制；验证方式是 `right_boundary`、`available_width`、`protected_gap` 写入报告。
- [ ] 旧槽位门禁失败必须阻塞候选生成或最终验收；验证方式是失败样例不能生成 accepted candidate。

### F. 放置策略选择

- [x] 报告包含 `placement_strategy` 和选择原因。
- [ ] 同字数 CJK 且字形变化小：必须验证 `top_left_anchor` 或等价策略，约束中心误差、字距、基线；验证方式是同字数小变化 fixture。
- [ ] 同字数 CJK 且字形变化大：必须验证 `center_primary`，约束左边界、基线、字距；验证方式是同字数大变化 fixture。
- [ ] 字数减少：目标字按旧值整体跨度排布，并清理多余旧槽位；验证方式是 3 字变 2 字 fixture。
- [ ] 字数增加：左边界锚定、向右扩展，且不覆盖 protected text；验证方式是 2 字变 3 字 fixture。
- [ ] 数字、日期、编号：左对齐和基线优先，保持数字节奏和字段宽度；验证方式是日期和年龄 fixture。
- [ ] 手动 ROI 且无旧值：使用保守居中或左对齐 fallback，并降低自动验收置信；验证方式是手动画框无旧值 fixture。
- [ ] 每个放置策略必须在 `result.json` 写入使用条件、关键约束、实际误差和是否通过；验证方式是 schema 或单测。

### G. 单字形态变化检测

- [x] 当前报告包含 `bbox_width_delta_ratio`、`bbox_height_delta_ratio`、`centroid_dx/dy`、`ink_area_ratio`。
- [ ] 每个 changed char 都必须生成旧槽位画像和新字候选画像；验证方式是 `shape_change_report.changed_chars[*]` 含 source/target image metrics。
- [ ] 增加 `row_projection_distance`；验证方式是报告字段存在并参与 `shape_change_large` 判定。
- [ ] 增加 `col_projection_distance`；验证方式是报告字段存在并参与 `shape_change_large` 判定。
- [ ] 增加 `margin_distribution_delta`；验证方式是报告字段存在并参与 `shape_change_large` 判定。
- [ ] 动态阈值必须来自旧槽位高度、邻字稳定性和字体候选分布；验证方式是报告写入每个阈值来源。
- [ ] 固定数字阈值只能作为第一版保守起点，并必须写入报告；验证方式是任何固定默认都有 `threshold_source=default`。
- [ ] 禁止用语义字表判断“单字变化大”；验证方式是形态检测代码没有目标字 hardcode，且测试覆盖不同字符。

### H. 字体形态联合搜索

- [ ] `text_shape` 阻塞时生成 shape candidate grid，且 grid 只包含字体、字号、放置、`text_dx/text_dy`、`char_offsets`、stroke body、shear；验证方式是 candidate 参数集合。
- [ ] 形态排序必须包含字高、字宽、字距、基线；验证方式是 shape score 明细字段。
- [ ] 形态排序必须包含单字中心与旧槽位中心误差；验证方式是 score 明细字段。
- [ ] 形态排序必须包含左边界和右边界误差；验证方式是 score 明细字段。
- [ ] 形态排序必须包含笔画面积和复杂度修正后的体量；验证方式是 score 明细字段。
- [ ] 形态排序必须包含姿态继承误差；验证方式是 shear/pose score 明细字段。
- [ ] 形态排序必须包含 protected text 距离；验证方式是 score 明细字段。
- [ ] 形态排序必须包含字体风格分数和可渲染字符检查；验证方式是 font report 字段。
- [ ] 形态没通过时，`blur`、`noise`、`jpeg_quality`、背景融合不能成为主修复方向；验证方式是 stage filter 单测。

### I. 黑灰比例搜索

- [ ] `ink_gray_balance` 只在形态 top candidates 上执行；验证方式是 ink candidate 的 parent shape candidate id 可追溯。
- [ ] 黑灰报告必须分开记录 `<55`、`<70`、`70-120`、`120-165`；验证方式是 hard report 字段。
- [ ] 核心太黑时，必须生成降低 `opacity`、`core_ink_gain` 或 `core_darken_strength` 的候选；验证方式是黑芯过量 fixture。
- [ ] 核心不足但灰边多时，不能继续加 blur 或扩大灰边，必须恢复核心密度并收紧外灰；验证方式是核心不足+灰边多 fixture。
- [ ] 旧字和邻字指标冲突时，必须记录仲裁，并优先同一行邻字作为风格上限；验证方式是 conflict report 字段。
- [ ] 黑灰阶段不能改变已经通过的字体、槽位和基线，除非重新回到 `text_shape`；验证方式是 candidate parent/rollback 记录。

### J. 照片质感搜索

- [ ] `photo_texture` 只能在 `text_shape` 和 `ink_gray_balance` 通过后执行；验证方式是 stage order 单测或失败 fixture。
- [ ] 可调参数必须限定为小幅 blur、edge breakup、局部噪声、压缩质感、轻微 alpha 退化、局部残差回填；验证方式是 stage patcher allowed keys。
- [ ] 目标必须是匹配原图拍照/扫描质感，不是把字弄糊；验证方式是报告同时记录 sharpness、breakup、noise、compression 指标。
- [ ] 照片质感不能破坏已通过的黑灰和形态指标；验证方式是 photo candidate 记录前后 stage severity。
- [ ] 文字过清晰、过干净、过糊、边缘无断裂必须进入 `photo_texture` 问题报告；验证方式是 issue type 枚举测试。

### K. 背景处理拆分

- [ ] 前置清除必须删除旧值槽位内旧字核心和灰边；验证方式是 source slot cleanup mask 和 residual metrics。
- [ ] 字数减少时，前置清除必须覆盖多余旧槽位；验证方式是 extra slot cleanup crop。
- [ ] 前置清除失败必须阻塞候选生成或最终验收；验证方式是旧残留 fixture 不能 deliver。
- [ ] 后置融合只围绕最终文字形态做局部融合；验证方式是 final candidate 的 background patch 范围不越过 target ROI 和保护文本。
- [ ] 后置融合必须分别报告补丁感、发白、发暗、平滑涂抹、纹理断裂和 ROI 边缘接缝；验证方式是 background metrics 字段。
- [ ] 后置融合不能掩盖旧槽位没清干净；验证方式是 cleanup failure 优先级高于 background naturalness。

### L. 分层联合优化和搜索预算

- [ ] 禁止全量笛卡尔积搜索；验证方式是候选生成报告记录分层阶段和剪枝数量，而不是单个全组合总数。
- [ ] Stage A shape search 本地候选预算为 300-1500，剪枝后保留 top 20-50；验证方式是候选统计字段。
- [ ] Stage B ink-gray search 本地候选预算为 100-800，剪枝后保留 top 8-20；验证方式是候选统计字段。
- [ ] Stage C photo texture search 本地候选预算为 30-200，剪枝后保留 top 3-8；验证方式是候选统计字段。
- [ ] Stage D vision final check 只看 top 3-8；验证方式是视觉请求候选数。
- [ ] 形态剪枝必须覆盖字高、中心、基线、字距、protected distance、字体风格、笔画体量、姿态继承；验证方式是 prune reason 枚举测试。
- [ ] 黑灰剪枝必须覆盖真黑核心、深色核心、外灰边、中灰笔画、复杂度修正；验证方式是 prune reason 枚举测试。
- [ ] 照片质感剪枝必须覆盖过锐、过糊、无断裂、背景平滑、白影/暗影/旧残留、ROI 梯度断裂；验证方式是 prune reason 枚举测试。

### M. Stage patcher 迁移边界

- [ ] 每个 stage patcher 必须声明 primary stage、allowed keys、blocked keys；验证方式是单测遍历 patcher registry。
- [ ] patcher 输出不能包含未声明参数；验证方式是单测。
- [ ] 跨 stage patch 必须被拒绝或声明主阶段、次级影响和不破坏前置阶段的依据；验证方式是 filter report 单测。
- [ ] 新增 patch 必须进入某个 stage patcher，不能散落在全局候选生成函数；验证方式是代码搜索和单测。
- [ ] 旧入口只能调用 stage dispatcher，stage dispatcher 不能反向调用旧全局混合补丁；验证方式是依赖方向检查或代码搜索。
- [ ] 临时双轨只允许用于同输入新旧结果差异验证，不能作为长期交付路径；验证方式是没有 runtime fallback 开关指向旧混合路径。
- [ ] 每个阶段迁移必须包含 detector、patcher、allowed/blocked 参数、失败用例、通过用例和 stage evidence；验证方式是测试目录和 fixture 记录。

### N. Profile 验收

- [ ] `photo_scan` 必须启用姿态和照片质感，且视觉模型不能在 stroke/shape 失败时 deliver；验证方式是 photo fixture。
- [ ] `clean_digital` 不启用 `photo_texture`，不鼓励 `photo_warp`，边缘应更干净；验证方式是 clean digital fixture。
- [ ] `low_res_thumbnail` 更重视字体结构和笔画体量，并要求视觉验收看放大图；验证方式是 low-res fixture 和 prompt payload。
- [ ] `manual_roi_quick` 只做最少阶段，未通过时保留 rejected candidate，不自动做复杂照片质感；验证方式是 manual ROI fixture。
- [ ] 同一张图可以用不同 profile 运行，并在 `result.json` 记录不同 stage order 或启用阶段差异；验证方式是 profile matrix smoke。

### O. 视觉模型 prompt 和本地仲裁

- [x] prompt 输入 stage context，输出建议不能越过本地 stage filter。
- [ ] prompt 必须先判断当前 `blocking_stage` 是否真实存在；验证方式是 prompt payload 和 JSON response schema。
- [ ] prompt 只能针对当前 `blocking_stage` 给建议；验证方式是建议 patch 经过 stage filter 并记录 rejected suggestion。
- [ ] prompt 不能建议当前阶段禁止参数；验证方式是 forbidden suggestion fixture 或 mock response。
- [ ] 如果模型认为前置阶段已通过，必须说明依据；验证方式是 response schema 包含 `basis` 或等价字段。
- [ ] 如果模型建议 deliver 但本地 `blocking_stage` 不为空，本地必须改为 revise；验证方式是 mock acceptance 单测。
- [ ] 视觉模型建议必须转成本地候选或记录不可转化原因；验证方式是 attempt record。

### P. 进度、UI 和失败产物

- [x] 失败也必须保留 rejected candidate、progress、result、stage evidence。
- [ ] CLI 每轮必须显示或输出 `round`、`profile`、`blocking_stage`、`reason`、`allowed_params`、`blocked_params`、`selected_optimization_step`；验证方式是 CLI JSON/文本 smoke。
- [ ] Web 每轮必须展示 `blocking_stage`、当前阶段失败原因、本轮候选图、因前置阶段失败被拒绝的候选、最终 `accepted` 状态；验证方式是浏览器截图或 DOM 测试。
- [ ] 失败任务必须保留自动方向选择报告、search/edit ROI 标注图、slot quality report、shape top candidates、ink-gray top candidates、photo texture top candidates、final visual candidates、rejected final candidate；验证方式是 run directory 文件存在性测试。

### Q. 回归 Case A-D

- [ ] Case A 字段旧文字修改为新文字：fixture、命令、预期 `blocking_stage`、核心密度阈值、外灰阈值和视觉不能覆盖本地失败条件必须写入本 checklist 或测试数据。
- [ ] Case A 必须验证 `stroke_body`/笔画体量在 edge cleanup 前通过；验证方式是粗细不足样例不能 deliver。
- [ ] Case B 字数减少：fixture、命令、多余旧槽位清理、后续文字不移动、旧残留阶段归因必须写入测试。
- [ ] Case C 字数增加：fixture、命令、不覆盖后续字段、ROI 扩展受 protected text 限制、先过字体字距再调颜色必须写入测试。
- [ ] Case D 干净数字图：fixture、命令、`clean_digital` profile、无照片噪声、干净边缘必须写入测试。
- [ ] 每个回归 case 必须保存 expected report fields，而不是只比较最终图片是否存在；验证方式是 JSON assertion。

### R. 反模式门禁

- [ ] 一个问题失败后不能把所有补丁族混合评分；验证方式是候选生成报告只显示当前 blocking stage 主导 patch。
- [ ] 视觉模型说 `ok` 不能覆盖本地阶段失败；验证方式是 mock response 单测。
- [ ] 字体没过时不能调 blur；验证方式是 stage filter 单测。
- [ ] 粗细没过时不能先清灰边；验证方式是粗细失败 fixture。
- [ ] 灰边过多时不能继续加 photo_noise 制造灰雾；验证方式是 edge/gray fixture。
- [ ] 不能把某张图的左倾/右倾写成通用规则；验证方式是代码搜索禁止具体图片/文字特例。
- [ ] 不能为单个字写特殊规则；验证方式是代码和 prompt 中无具体人名、具体目标字调参规则。
- [ ] 不能用更多迭代次数替代阶段判定；验证方式是 max rounds 增加前必须有 stage-specific new candidate direction。
- [ ] 不允许“先交付，再靠用户肉眼指出问题”作为成功标准；验证方式是 deliver 需要全部本地 stage 和最终视觉验收通过。

### S. 设计文档状态同步

- [ ] `text_shape_joint_optimization_design.md` 的“现有流程差距”表必须同步当前状态：已实现项标记为已覆盖，未实现项链接到本 checklist 对应条目。
- [ ] `staged_roi_pipeline_design.md` 的旧 7 阶段设计必须说明与当前 5 stage 结构的关系，避免两个阶段体系并存。
- [ ] README 中不能宣称本地流程完善完成，除非本节所有 `[ ]` 关闭。
- [ ] 提交或 PR 说明必须引用关闭的 checklist 项，不能只写“优化流程”。

### T. 代码落点和模块边界

- [x] 存在 `src/roi_image_edit/stages.py`，承载 `StageSpec`、`StageResult` 和 detector mapping。
- [x] 存在 `src/roi_image_edit/stage_profiles.py`，承载 profile 定义和加载。
- [x] 存在 `src/roi_image_edit/stage_patchers.py`，承载 stage patcher 调度入口。
- [ ] `local_validation.py` 中的 `local_*_issues` 必须继续收敛为 detector 底层函数，不能直接承担 stage policy；验证方式是 stage policy 单测只通过 `stages.py` 调用 detector。
- [ ] `revision_solver.py` 中的旧评分和候选选择逻辑必须继续收敛为 dispatcher/selector，不能重新生成跨阶段混合 patch；验证方式是代码搜索和 stage filter 单测。
- [ ] `processing_service.py` 不能继续承载 ROI 定位、候选生成、验收评分、修订求解器的长期核心逻辑；验证方式是这些能力迁移到 `roi_locator.py`、stage modules、solver modules 或等价 core 模块。
- [ ] `iterative_pipeline.py` 的视觉 prompt 上下文必须只通过 stage context 结构传递阶段信息，不能手写散落的 prompt 拼接逻辑；验证方式是 prompt payload 单测。
- [ ] `web_app.py` 必须只保留 HTTP/API/job 状态和本地 Web 服务启动，不承载阶段策略或图像处理逻辑；验证方式是代码搜索不出现 detector、patcher、candidate generation 逻辑。
- [ ] `result.json` 和 `progress.jsonl` 必须是所有阶段证据的稳定外部接口；验证方式是 schema 测试覆盖 stage、profile、candidate、patch、vision suggestion 和 rejection reason。

## Current Implementation Status

- 已接入 `reference_profile` 报告字段，包含旧文字、邻字、动态墨色阈值和动态核心变浅阈值。
- 已移除 `opacity >= 0.76` 作为硬下限，改为 `reference_profile.dynamic_ink.opacity_floor_for_excess_core`。
- 已修复“核心 + 黑”文本误判，明确区分 `too_dark` / `too_bold` / `too_light` / `too_thin`。
- 已接入 stage severity 优先选择，revision 记录包含 severity 前后值和 `selected_reason`。
- 已支持字段缺旧值的自动 ROI，当前覆盖 `name` 和 `receive_time`，非 CJK 日期时间按整段值槽处理。
- 已新增 `background_texture_metrics`，并把视觉 `patch_visible` / `ghost_visible` 转成本地背景修补候选。
- 已将旧字擦除区拆成 `background_white_ghost_residual` 和 `background_shadow_ghost_residual` 两类结构化残影；白影、暗影、低纹理分别走不同修复方向。
- 已把背景 retexture 和 ROI 扫描残差强度接入本地参数，`background_cleanup` 不再只能靠视觉模型描述补丁感。
- 对日期/数字这类旧值更长、新值更短的任务，不允许用整块矩形补丁清理尾部；旧尾部只能通过旧字符笔画 mask、灰边扩展和 ghost/shadow 指标处理。
- 已修复 severity 正向但低于显著阈值时 `near_best` 为空导致的迭代崩溃；小幅正向改善会继续迭代并写入 `selected_reason`。
- 已新增 stage optimization policy：每轮记录当前 blocking stage、允许/禁止 Optimization Step、被拒绝的本地 patch、模型建议和本地指标冲突。
- 已把视觉模型的 `parameter_suggestions` 转为本地 patch，并在 attempt record 里记录是否被本地约束截断及替代候选。
- 已在 Web 结果区显示 accepted/rejected、blocking stage、迭代轮次、停止原因和下一轮计划；候选抽屉显示最多 5 个代表性候选及背景摘要。
- 已给视觉 API 临时 502/503/504/连接错误增加短重试，避免完整流程被单次上游波动中断。
- 回归验证：
  - `接受时间修改2026-06-04`：`output/web/20260605_020104`，自动 ROI `[1296,94,1624,134]`，1 轮后 `accepted=true`，最终无本地 blocking stage。
  - `姓名陈芸修改为赵真真`：`output/web/20260605_020350`，3 轮后 `accepted=true`，最后一轮 `ink_gray_balance` severity `39.284 -> 0.0`。
  - `接受时间修改2026-06-04`：`output/web/20260605_024521`，视觉初验将背景问题映射为 `background_cleanup`，1 轮后 `accepted=true`，progress 写入 `basis_stage_source=vision_acceptance`。
  - `姓名陈芸修改为赵真真`：`output/web/20260605_024811`，6 轮后 `accepted=true`，`ink_gray_balance` severity `741.344 -> 170.464 -> 39.284 -> 0.0`，后续视觉墨色反馈继续回灌本地候选。

## 不折中目标

1. 不用固定经验值作为最终裁决，例如 `opacity >= 0.76` 只能作为旧实现问题记录，不能作为阻止调参的硬下限。
2. 不允许用照片质感、模糊、噪声或压缩掩盖字体、字距、基线、姿态、粗细问题。
3. 不允许视觉模型的主观 `pass` 覆盖本地硬校验、阶段门禁或原图参照指标。
4. 不允许同一 blocking stage 的关键指标反弹后仍被选中，除非报告中证明其它更高优先级指标必须这样处理。
5. 不允许在自动定位失败时静默输出原图或看似成功的结果。
6. 不允许交付未标注的失败产物。失败必须带有最终候选图、每轮候选图、指标报告和下一轮建议。

## Phase 1: Reference Profile

每个 ROI 在生成候选前必须建立 `reference_profile`。后续字体、墨色、灰边、姿态、背景质感都以它为主，不以固定经验值为主。

### 必须记录

- 旧文字逐字画像：
  - 字符槽位 `slot_box`
  - 字高、字宽、中心、基线、字距
  - `<55`, `<70`, `<90`, `<120`, `<165`
  - `55-70`, `70-90`, `90-120`, `120-165`
- 邻字画像：
  - 同一字段标签，例如 `姓名:`、`名:`、日期、年龄
  - 同一行未修改中文、数字或符号
  - 邻字核心密度、灰边比例、边缘模糊、局部倾斜
- 背景画像：
  - 修补区周边亮度均值和方差
  - 纹理残差强度
  - 局部亮度梯度
  - 噪声和压缩质感
- 字形复杂度修正：
  - 目标字和旧字的复杂度差异
  - 复杂目标字允许更多笔画面积，但不能突破邻字风格上限

### Checklist

- [x] 生成 `reference_profile.json` 或等价报告字段。
- [x] 报告中能区分旧字参照、邻字参照、背景参照。
- [x] 每个动态阈值都能说明来源：旧字、邻字、背景、复杂度修正或保守默认。
- [x] 如果没有可用邻字，报告必须写明邻字参照不可用，不能假装通过。
- [x] 如果旧字和邻字参照冲突，报告必须写出仲裁结果。

### Done Definition

一个任务不能进入候选排序，除非已经生成 reference profile，或者明确记录 reference profile 失败原因并中止处理。

## Phase 2: Replace Fixed Thresholds with Dynamic Gates

固定经验值只能作为初始搜索网格，不得作为最终硬限制。尤其是墨色阶段，必须由原图旧字和邻字指标决定是否继续调参。

### 必须替换的旧行为

- `opacity >= 0.76` 不能阻止 `roi_core_too_black` 后续继续降黑。
- “核心黑度偏重、核心过黑、核心黑像素过量”不能被解析为“需要更黑核心”。
- 如果视觉模型建议 `opacity 0.76 -> 0.72`，本地必须至少生成对应候选，不能被静默截断。
- 若本地指标显示 `<55` 真黑核心严重超标，下一轮不能选择让 `<55` 反弹的候选。

### 动态规则

- 如果新字 `<55` 真黑核心超过旧字和邻字允许范围，允许继续降低 `opacity`、`alpha_contrast`、`core_ink_gain`、`core_darken_strength`。
- 如果降低 `opacity` 后 `<55` 接近目标但 `120-165` 灰边过多，停止继续降 opacity，转为收紧灰边和照片边缘断裂。
- 如果新字偏淡但灰雾多，不直接加粗，应先判断缺的是核心、笔画身体还是照片边缘。
- 如果原字和邻字都很黑，可以提高 opacity 下限，但必须来自 reference profile。

### Checklist

- [x] 删除或降级 `ink_gray_balance` 分支里的固定 `0.76` 下限。
- [x] 增加 `too_black`、`too_bold`、`core_too_black`、`核心黑度偏重` 的明确解析。
- [x] 增加 `too_light`、`core_too_light`、`核心不够黑` 的明确解析。
- [x] 验证同一文本中出现“核心”和“黑”时，不会自动判定为想要更黑。
- [x] 每轮报告记录建议参数是否被本地约束截断。
- [x] 若建议被截断，必须写出截断原因和替代候选。

### Done Definition

当 blocking stage 是 `ink_gray_balance` 且黑芯过量时，流程必须能连续生成更低黑芯候选，直到本地动态门禁通过、转入其它阶段、或明确证明继续降低会破坏更高优先级指标。

## Phase 3: Stage-Specific Solvers

每个 blocking stage 使用自己的求解器。不能所有阶段共享一堆通用 patch 后靠总分碰运气。

### Stage Order

1. `hard_boundary`
2. `text_shape`
3. `ink_gray_balance`
4. `photo_texture`
5. `background_cleanup`

### Solver Rules

`hard_boundary`：

- 只处理尺寸、ROI 外、边缘、protected boxes。
- 失败时停止，不进入视觉模型验收。

`text_shape`：

- 只处理字体、字号、字槽、字距、基线、笔画身体、局部姿态。
- 不能通过降低黑度、加模糊、加噪声来掩盖形态问题。

`ink_gray_balance`：

- 只处理真黑核心、中灰笔画、外灰边。
- 黑芯过量时必须产生降低黑芯候选。
- 核心不足时必须产生恢复核心候选。
- 灰边过多时必须产生收边候选。

`photo_texture`：

- 只在 text shape 和 ink gray balance 通过后进入。
- 处理照片模糊、边缘断裂、局部噪声、压缩质感。

`background_cleanup`：

- 独立评估旧字残留、发白、过平滑、纹理断裂、边缘接缝。
- 不能只看 extra source slot，必须覆盖整个 target ROI 修补区。

### Checklist

- [x] 为每个 stage 定义允许参数和禁止参数。
- [x] `text_shape` 未通过时，禁止 `blur`、`photo_noise`、`jpeg_quality` 成为主调参方向。
- [x] `ink_gray_balance` 未通过时，禁止形态重搜或照片质感覆盖墨色问题，除非当前候选形态又失败。
- [x] `photo_texture` 未通过前，确认前两个阶段已通过。
- [x] `background_cleanup` 未通过时，优先修 mask、inpaint 和纹理恢复，不用新字遮盖旧残留。
- [x] 每轮只允许当前 blocking stage 的 Optimization Step 主导候选。

### Done Definition

每个 revision round 必须明确：

- 当前 blocking stage
- 该 stage 的 severity
- 允许 Optimization Step
- 被禁止的 patch
- 选中候选为什么没有违反阶段顺序

## Phase 4: Severity-First Candidate Selection

候选选择必须优先解决当前 blocking stage，不能只看总分。

### 规则

- 同一 blocking stage 下，优先选择 severity 下降最多的候选。
- 如果候选总分略差，但当前 stage severity 明显下降，应允许进入下一轮。
- 如果候选让当前 stage severity 反弹，应强惩罚。
- 如果连续两轮同一问题未改善，应扩大该方向搜索，而不是重复相同 patch。
- 如果视觉模型建议和本地指标冲突，应记录冲突并优先本地指标。

### Checklist

- [x] 在 `revision_attempts` 中记录每个候选的 stage severity。
- [x] 在 `revision_rounds` 中记录 selected candidate 的选择理由。
- [x] 记录当前 stage severity 是否比上一轮下降。
- [x] 若 `<55` 从接近合格反弹到明显超标，候选不得被选中，除非 text_shape 更高优先级重新失败并需要回退。
- [x] 若轮数结束但仍有明确建议，写出第 N+1 轮应该尝试的参数。

### Done Definition

一个 round 的 selected candidate 必须能解释为以下之一：

- 当前 blocking stage 通过；
- 当前 blocking stage severity 明显下降；
- 更高优先级 stage 重新失败，需要回退处理；
- 没有可选候选，但必须写出原因。

## Phase 5: Complete Background Naturalness Scoring

背景不是只要旧字清掉即可。照片件里，背景过白、过平滑、涂抹、接缝都会导致失败。

### 新增指标

- `patch_mean_delta`：修补区亮度均值相对周围的偏移。
- `patch_variance_ratio`：修补区纹理方差相对周围是否过低。
- `residual_energy_ratio`：修补区高频残差是否低于纸面背景。
- `gradient_continuity_error`：亮度梯度是否在 ROI 边界断裂。
- `edge_seam_pixels`：target ROI 边缘是否出现接缝。
- `white_glow_ratio`：修补区是否发白。
- `white_ghost_probe`：旧字掩码扩展区相对同 ROI 背景的高亮/偏暗结构残影。
- `shadow_ghost_ratio`：旧字掩码扩展区相对同 ROI 背景的暗灰残影。
- `smear_direction_score`：是否出现明显涂抹方向。

### Checklist

- [x] 增加 `background_texture_metrics`。
- [x] 把 background metrics 写入 hard report。
- [x] `background_cleanup` stage 使用这些指标，而不是只依赖视觉模型。
- [x] 区分白色 ghost、暗灰 ghost、低纹理补丁，避免把三者混成同一类调参。
- [x] 如果背景失败，下一轮优先修补背景，不继续调文字。
- [x] Web 候选抽屉显示背景对比 crop。

### Done Definition

最终验收通过前，背景修补区必须与周围背景在亮度、纹理、梯度和接缝上都可解释地接近。

## Phase 6: Narrow Vision Model Role

视觉模型负责整体视觉判断和阶段验收，不负责替代本地求解。

### Prompt 输出必须结构化

视觉模型必须返回：

```json
{
  "pass": false,
  "acceptance_level": "marginal",
  "final_decision": "revise",
  "blocking_stage": "ink_gray_balance",
  "direction": "reduce_true_black_core",
  "parameter_suggestions": [
    {"name": "opacity", "from": 0.76, "to": 0.72}
  ],
  "confidence": "medium",
  "visual_findings": {}
}
```

### Checklist

- [x] Prompt 明确要求区分 `too_dark`、`too_bold`、`too_light`、`too_thin`。
- [x] Prompt 明确禁止把“核心过黑”表达成“核心不够黑”。
- [x] Prompt 明确要求输出 blocking stage 和 direction。
- [x] 模型建议必须转成本地候选并写入 attempt record。
- [x] 本地指标冲突时，不能静默忽略模型建议，必须写入 conflict record。

### Done Definition

视觉模型每次 `revise` 都必须能转化为一组本地候选，或者明确记录为什么不能转化。

## Phase 7: Progress, Failure Delivery, and User Traceability

流程失败也必须有完整产物。

### Checklist

- [x] `progress.jsonl` 记录每轮 stage、severity、candidate count、selected candidate、accepted。
- [x] `result.json` 记录 rejected candidate 的最终图和未应用状态。
- [x] 即使 `accepted=false`，也写出最后候选图。
- [x] Web UI 显示当前 blocking stage 和迭代轮次。
- [x] Web UI 显示“为什么没有继续”的原因。
- [x] 若达到最大轮数，报告写出下一轮计划。
- [x] 候选 drawer 显示至多 5 个代表性候选：初始、最好、最近、最接近通过、最终 rejected。

### Done Definition

用户不需要读代码，就能从 Web 或报告中知道：

- 当前处理到哪一阶段；
- 为什么失败；
- 最后候选在哪里；
- 如果继续，下一步会调什么。

## Implementation Checklist

### A. Reference Profile

- [x] 新增 reference profile 构建函数。
- [x] 输出旧字逐字灰度带、位置、姿态。
- [x] 输出邻字灰度带、位置、姿态。
- [x] 输出背景纹理统计。
- [x] 报告动态阈值来源。

### B. Dynamic Ink Solver

- [x] 移除 `opacity >= 0.76` 作为硬下限。
- [x] 基于 reference profile 计算 opacity 搜索范围。
- [x] 修复“核心黑度偏重”误判为“想要更黑核心”。
- [x] 黑芯过量时强制生成降低黑芯候选。
- [x] 黑芯指标反弹时强惩罚。

### C. Stage Solvers

- [x] 给每个 stage 定义 Optimization Step。
- [x] 禁止跨 stage 参数抢先主导。
- [x] 每轮只让当前 blocking stage 的求解器主导。
- [x] 形态阶段失败时触发字体、字号、字槽、姿态重搜。
- [x] 墨色阶段通过前不进入照片质感主调参。

### D. Candidate Selection

- [x] 记录所有候选的 stage severity。
- [x] 当前 stage severity 下降优先。
- [x] 防止 `<55`、灰边、位置等关键指标来回反弹。
- [x] 记录候选为何被选中或被拒绝。

### E. Background Scoring

- [x] 实现全 target ROI 背景自然度指标。
- [x] 检查发白、过平滑、纹理残差、梯度断裂、接缝。
- [x] 将背景失败纳入 `background_cleanup` stage。

### F. Vision Prompt

- [x] 更新候选排序 prompt。
- [x] 更新最终验收 prompt。
- [x] 强制模型输出 `blocking_stage`、`direction`、`parameter_suggestions`。
- [x] 将模型建议转为本地候选并记录是否被截断。

### G. Web and CLI Traceability

- [x] Web 显示阶段进度。
- [x] Web 显示 accepted/rejected 状态。
- [x] Web 显示最终 rejected candidate 和失败原因。
- [x] CLI 输出每轮 progress。
- [x] 达到最大轮数时输出下一轮建议。

## Public Acceptance Checklist

完成上述改造后，每个回归任务必须检查：

- [x] 输出尺寸与原图一致。
- [x] ROI 外像素不变。
- [x] 边缘像素不变。
- [x] protected text 不变。
- [x] 自动定位失败立即报错。
- [x] `reference_profile` 完整或明确失败。
- [x] `text_shape` 通过后才进入 `ink_gray_balance`。
- [x] `ink_gray_balance` 由原字和邻字动态判断。
- [x] 没有固定经验值阻断已验证的调参方向。
- [x] `photo_texture` 不掩盖形态或墨色问题。
- [x] `background_cleanup` 覆盖整个修补区自然度。
- [x] 视觉模型建议被转成本地候选。
- [x] 模型和本地指标冲突被记录。
- [x] 失败时保留 rejected candidate、对比图、报告和下一轮计划。
- [ ] “设计目标转换情况”中的所有 `[ ]` 项全部关闭。
- [ ] 回归 Case A-D 已转成可执行用例并通过。
- [ ] 每个 stage patcher 的声明参数和拒绝跨阶段参数测试通过。

任何一项未满足，都不能宣称本地流程完善完成。
