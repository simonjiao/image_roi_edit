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

## Stage Refactor Execution Checklist

本节跟踪 `staged_roi_pipeline_design.md` 中“阶段化流程”从设计到代码的实施状态。完成标准不是某张图看起来更好，而是代码中存在清晰模块边界、每阶段有可验证输入输出、失败时有可追踪证据。

### Slice 1: 可执行 stage/profile 结构

- [x] 新增 `src/roi_image_edit/stages.py`，定义 `StageSpec`、`StageResult` 和 detector mapping。
- [x] 新增 `src/roi_image_edit/stage_profiles.py`，定义 `photo_scan`、`clean_digital`、`low_res_thumbnail`、`manual_roi_quick`。
- [x] `local_validation.stage_gate_for_report()` 委托 `stages.py`，报告包含 `profile`、`stage_status`、`allowed_patch_keys`、`blocked_patch_keys`。
- [ ] CLI 和 Web payload 支持用户指定 profile，并写入 `result.json` / `progress.jsonl`。

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

- [ ] 新增 `slot_quality_report`。
- [ ] 候选生成前检查旧值槽位数、槽位完整性、底部/灰边覆盖、protected text 冲突。
- [ ] 旧槽位不完整时直接返回 rejected，不进入字体/墨色/照片质感候选。
- [ ] 字数减少时，多余旧槽位纳入前置清除报告；字数增加时，右边界受 protected text 限制。

验证：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py process \
  --image <image> \
  --instruction '<字段旧值修改为新值>' \
  --json
```

输出必须包含 `summary.plan.slot_quality_report`。失败时必须有 rejected candidate 和明确 `slot_quality_failed` 原因。

### Slice 4: 放置策略和单字形态变化检测

- [ ] 新增 `placement_strategy` 字段，至少支持 `top_left_anchor`、`center_primary`、`left_anchor_span`、`baseline_numeric`、`manual_fallback`。
- [ ] 新增单字形态变化检测，报告 `bbox_width_delta_ratio`、`bbox_height_delta_ratio`、`centroid_dx/dy`、`ink_area_ratio`、投影距离和边距分布差异。
- [ ] 同字数 CJK 字形变化大时自动切到 `center_primary`，并记录选择原因。
- [ ] `text_shape` 未通过时，禁止 blur/noise/JPEG/background 成为主修复方向。

验证：`candidate_report` 中必须能看到 `placement_strategy` 和 `shape_change_report`；当 `shape_change_large=true` 时，候选生成必须包含中心优先候选。

### Slice 5: 分层候选产物和阶段证据

- [ ] `progress.jsonl` 每轮写入 `pipeline_profile`、`stage_status`、`blocking_stage`、`allowed_patch_keys`、`blocked_patch_keys`。
- [ ] `result.json` 写入最终候选的完整 stage evidence。
- [ ] 保存 shape、ink-gray、photo texture、background cleanup 的 top candidate 或 rejected candidate 证据。
- [ ] Web 候选抽屉展示 profile、stage、stage severity、patcher source。
- [ ] 视觉 prompt 输入当前 stage context，输出建议不能越过本地 stage filter。

验证：一次失败任务也必须能从 `output/web/<run>/progress.jsonl` 和 `result.json` 解释“卡在哪个阶段、为什么没有继续、下一轮应该调什么”。

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

任何一项未满足，都不能宣称本地流程完善完成。
