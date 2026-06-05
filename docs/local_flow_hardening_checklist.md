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

## Checklist 使用规则

本文件只有“设计目标转换情况”一节是权威实施 checklist。旧的阶段切片、阶段说明和实现记录不再用
`[x]` 表示完成，避免把“已有部分代码”误写成“目标能力已验收”。所有完成项必须满足：

1. 可实施：能落到具体代码、prompt、报告、fixture、测试或 UI/CLI 输出。
2. 可验证：条目中必须说明关闭它的证据来源。
3. 不宽泛：如果设计目标包含多个子要求，不能用一个大 `[x]` 覆盖。
4. 不补丁化：如果某项只能靠一次性参数或单图规则通过，先补 detector、stage 定义或测试，再标记完成。

## 设计目标转换情况

本节把 `staged_roi_pipeline_design.md` 和
`text_shape_joint_optimization_design.md` 中的目标拆成可实施、可验证的 checklist 项。
`[x]` 只表示该条目标已经完整实现，并且有同条或相邻说明中的测试、报告、fixture 或稳定输出证据；`[ ]` 表示还需要代码、测试、报告或文档同步。
每个 `[ ]` 都必须能用测试、CLI/Web 输出、`progress.jsonl`、`result.json`、stage evidence
或 fixture 回归关闭，不能只靠文字说明关闭。

### A. 三层流程边界

- [ ] 五个本地 stage 必须成为代码、prompt、报告和测试的唯一权威结构：`hard_boundary`、`text_shape`、`ink_gray_balance`、`photo_texture`、`background_cleanup`；验证方式是 stage order、prompt payload、result schema 和回归报告都只使用这五个 stage。
- [x] `StageSpec` 必须有稳定字段契约：`id`、`display_name`、`blocks_next`、`detect`、`optimization_steps`、`allowed_patch_keys`、`blocked_patch_keys`；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_stage_spec_field_contract_and_reports`。
- [x] `StageResult` 必须有稳定字段契约：`stage_id`、`display_name`、`passed`、`blocks_next`、`severity`、`issues`、`reason`、`allowed_patch_keys`、`blocked_patch_keys`；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_stage_result_field_contract_and_stage_context`。
- [x] 每个 stage 必须能回答四个问题：是否通过、失败是否阻塞后续、允许哪些参数、禁止哪些参数；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_stage_result_field_contract_and_stage_context`。
- [ ] 增加前置安全流程验收：`orientation_check`、`field_roi_selection`、`slot_quality_gate`、`protected_text_guard` 必须在候选生成前完成；验证方式是失败样例 `candidate_count=0` 或 rejected，且 `progress.jsonl` 记录失败步骤。
- [x] 增加阶段门禁顺序验收：`src/roi_image_edit/stage_policy.py` 的 `STAGE_ORDER` 必须与本 checklist 的五阶段顺序一致；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_stage_order_is_the_five_stage_contract`。
- [ ] 增加阶段内 Optimization Step 验收：每个候选报告必须区分 `stage_id` 和 `optimization_step`，不能把 Optimization Step 当成新 stage；验证方式是 `result.json` 中同时存在两类字段。
- [ ] 增加视觉终检边界验收：视觉模型只能看本地 top candidates；验证方式是视觉请求记录中 `candidate_count <= vision_candidate_limit` 且包含本地 `stage_context`。

### B. 旧 7 类诊断关注点映射到当前 5 个 stage

- [ ] `slot_alignment` 必须映射到 `hard_boundary` 的 ROI/slot 安全条件和 `text_shape.slot_alignment_search`；验证方式是 stage evidence 记录旧名、当前 stage、Optimization Step 和报告字段。
- [ ] `font_structure` 必须映射到 `text_shape.font_style_search`、`font_size_search`；验证方式是字体失败样例不会进入 `ink_gray_balance` 主调参。
- [ ] `pose_geometry` 必须映射到 `text_shape.pose_shear_search`，且不能固化某张图的左倾/右倾；验证方式是姿态报告来自旧槽位、邻字或局部投影指标。
- [ ] `stroke_body` 必须映射到 `text_shape.stroke_body_search`，且在真实笔画体量未过时不能被 `edge_quality` 或 `photo_texture` 抢先处理；验证方式是粗细失败样例的 `blocking_stage=text_shape`。
- [ ] `tone_gray` 必须映射到 `ink_gray_balance.core_black_search`、`mid_gray_body_search`、`opacity_search`；验证方式是黑芯过量和核心不足分别生成相反方向候选。
- [ ] `edge_quality` 必须拆到 `ink_gray_balance.outer_gray_control` 和 `photo_texture.edge_breakup_match`，并记录拆分依据；验证方式是灰边过量不会先破坏已通过的 stroke body。
- [ ] `photo_texture` 必须映射到 `photo_texture.blur_match`、`edge_breakup_match`、`noise_texture_match`、`jpeg_texture_match`、`residual_retexture`；验证方式是 `photo_texture` 只在形态和黑灰通过后成为 blocking stage。
- [ ] 更新所有 prompt、report、UI 文案中的旧 stage 名引用；验证方式是公开输出不再把旧 7 类诊断关注点当成本地 gate。

### C. 全局硬约束

- [ ] 输出尺寸与原图一致必须有 fixture 或 hard report 断言；验证方式是所有回归任务记录 `output.size == original.size`。
- [ ] ROI 外像素不变必须有 fixture 或 hard report 断言；验证方式是所有回归任务记录 ROI 外变化像素数。
- [ ] 图片边缘像素不变必须有 fixture 或 hard report 断言；验证方式是所有回归任务记录边缘变化像素数。
- [ ] protected text 不变必须有 fixture 或 hard report 断言；验证方式是所有自动 ROI 和手动 ROI 回归任务记录 protected box diff。
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

- [ ] `slot_quality_report` 必须有稳定 schema 和 fixture 断言；验证方式是逐字段检查 `source_count`、`target_count`、coverage、overlap、right boundary 和 cleanup mask。
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

- [ ] `placement_strategy` 和选择原因必须有 schema 和多场景 fixture 断言；验证方式是同字数、字数增减、日期/年龄、手动 ROI 都记录 strategy reason。
- [ ] 同字数 CJK 且字形变化小：必须验证 `top_left_anchor` 或等价策略，约束中心误差、字距、基线；验证方式是同字数小变化 fixture。
- [ ] 同字数 CJK 且字形变化大：必须验证 `center_primary`，约束左边界、基线、字距；验证方式是同字数大变化 fixture。
- [ ] 字数减少：目标字按旧值整体跨度排布，并清理多余旧槽位；验证方式是 3 字变 2 字 fixture。
- [ ] 字数增加：左边界锚定、向右扩展，且不覆盖 protected text；验证方式是 2 字变 3 字 fixture。
- [ ] 数字、日期、编号：左对齐和基线优先，保持数字节奏和字段宽度；验证方式是日期和年龄 fixture。
- [ ] 手动 ROI 且无旧值：使用保守居中或左对齐 fallback，并降低自动验收置信；验证方式是手动画框无旧值 fixture。
- [ ] 每个放置策略必须在 `result.json` 写入使用条件、关键约束、实际误差和是否通过；验证方式是 schema 或单测。

### G. 单字形态变化检测

- [ ] `bbox_width_delta_ratio`、`bbox_height_delta_ratio`、`centroid_dx/dy`、`ink_area_ratio` 必须有 schema 和 fixture 断言；验证方式是每个 changed char 都输出这些字段。
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
- [x] 形态没通过时，`blur`、`noise`、`jpeg_quality`、背景融合不能成为主修复方向；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_text_shape_stage_rejects_photo_or_background_primary_patches`。

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

- [x] stage patch filter 已能按当前 blocking stage 接受/拒绝补丁族；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_text_shape_stage_rejects_photo_or_background_primary_patches`。
- [ ] 每个 stage patcher 必须声明 primary stage、allowed keys、blocked keys；验证方式是单测遍历 patcher registry。
- [ ] patcher 输出不能包含未声明参数；验证方式是单测。
- [ ] 跨 stage patch 必须被拒绝或声明主阶段、次级影响和不破坏前置阶段的依据；验证方式是 filter report 单测。
- [ ] 新增 patch 必须进入某个 stage patcher，不能散落在全局候选生成函数；验证方式是代码搜索和单测。
- [ ] 旧入口只能调用 stage dispatcher，stage dispatcher 不能反向调用旧全局混合补丁；验证方式是依赖方向检查或代码搜索。
- [ ] 临时双轨只允许用于同输入新旧结果差异验证，不能作为长期交付路径；验证方式是没有 runtime fallback 开关指向旧混合路径。
- [ ] 每个阶段迁移必须包含 detector、patcher、allowed/blocked 参数、失败用例、通过用例和 stage evidence；验证方式是测试目录和 fixture 记录。

### N. Profile 验收

- [x] profile registry 已公开 `photo_scan`、`clean_digital`、`low_res_thumbnail`、`manual_roi_quick`，且 `clean_digital` 当前禁用 `photo_texture`；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_profile_contracts_cover_current_profiles`。
- [ ] `photo_scan` 必须启用姿态和照片质感，且视觉模型不能在 stroke/shape 失败时 deliver；验证方式是 photo fixture。
- [ ] `clean_digital` 不启用 `photo_texture`，不鼓励 `photo_warp`，边缘应更干净；验证方式是 clean digital fixture。
- [ ] `low_res_thumbnail` 更重视字体结构和笔画体量，并要求视觉验收看放大图；验证方式是 low-res fixture 和 prompt payload。
- [ ] `manual_roi_quick` 只做最少阶段，未通过时保留 rejected candidate，不自动做复杂照片质感；验证方式是 manual ROI fixture。
- [ ] 同一张图可以用不同 profile 运行，并在 `result.json` 记录不同 stage order 或启用阶段差异；验证方式是 profile matrix smoke。
- [ ] 用户指定 profile 必须优先于自动 profile 建议；验证方式是同一输入同时存在自动建议和显式 `--profile` 时，`result.json` 使用用户指定 profile 并记录自动建议仅供参考。

### O. 视觉模型 prompt 和本地仲裁

- [ ] prompt 输入 stage context、输出建议不能越过本地 stage filter 必须有 mock 或 fixture 验证；验证方式是 forbidden suggestion 被本地拒绝并写入 attempt record。
- [ ] prompt 必须先判断当前 `blocking_stage` 是否真实存在；验证方式是 prompt payload 和 JSON response schema。
- [ ] prompt 只能针对当前 `blocking_stage` 给建议；验证方式是建议 patch 经过 stage filter 并记录 rejected suggestion。
- [ ] prompt 不能建议当前阶段禁止参数；验证方式是 forbidden suggestion fixture 或 mock response。
- [ ] 如果模型认为前置阶段已通过，必须说明依据；验证方式是 response schema 包含 `basis` 或等价字段。
- [ ] 如果模型建议 deliver 但本地 `blocking_stage` 不为空，本地必须改为 revise；验证方式是 mock acceptance 单测。
- [ ] 视觉模型建议必须转成本地候选或记录不可转化原因；验证方式是 attempt record。

### P. 进度、UI 和失败产物

- [ ] 失败也必须保留 rejected candidate、progress、result、stage evidence；验证方式是失败任务 run directory 文件存在性测试。
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
- [x] 字体没过时不能调 blur；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_stage_contracts.py::StageContractsTest.test_text_shape_stage_rejects_photo_or_background_primary_patches`。
- [ ] 粗细没过时不能先清灰边；验证方式是粗细失败 fixture。
- [ ] 灰边过多时不能继续加 photo_noise 制造灰雾；验证方式是 edge/gray fixture。
- [ ] 不能把某张图的左倾/右倾写成通用规则；验证方式是代码搜索禁止具体图片/文字特例。
- [ ] 不能为单个字写特殊规则；验证方式是代码和 prompt 中无具体人名、具体目标字调参规则。
- [ ] 不能用更多迭代次数替代阶段判定；验证方式是 max rounds 增加前必须有 stage-specific new candidate direction。
- [ ] 不允许“先交付，再靠用户肉眼指出问题”作为成功标准；验证方式是 deliver 需要全部本地 stage 和最终视觉验收通过。

### S. 设计文档状态同步

- [ ] `text_shape_joint_optimization_design.md` 的“现有流程差距”表必须同步当前状态：已实现项标记为已覆盖，未实现项链接到本 checklist 对应条目。
- [x] `staged_roi_pipeline_design.md` 的旧 7 类诊断关注点必须说明与当前 5 stage 结构的关系，避免两个阶段体系并存；证据：`.venv/bin/python -m unittest discover -s tests`，`tests/test_design_alignment.py::DesignAlignmentTest.test_staged_design_declares_five_stage_gate_and_old_concern_mapping`。
- [ ] README 中不能宣称本地流程完善完成，除非本节所有 `[ ]` 关闭。
- [ ] 提交或 PR 说明必须引用关闭的 checklist 项，不能只写“优化流程”。

### T. 代码落点和模块边界

- [ ] `src/roi_image_edit/stages.py` 必须成为 stage contract 的唯一入口，承载 `StageSpec`、`StageResult`、detector mapping 和 prompt context；验证方式是 schema/unit test 与依赖方向检查。
- [ ] `src/roi_image_edit/stage_profiles.py` 必须成为 profile 定义和加载的唯一入口；验证方式是 profile matrix smoke 和用户指定 profile 覆盖自动建议测试。
- [ ] `src/roi_image_edit/stage_patchers.py` 必须成为 stage patcher filter/dispatch 的唯一入口；验证方式是 patcher registry 单测、allowed/blocked keys 单测和旧混合路径依赖检查。
- [ ] `local_validation.py` 中的 `local_*_issues` 必须继续收敛为 detector 底层函数，不能直接承担 stage policy；验证方式是 stage policy 单测只通过 `stages.py` 调用 detector。
- [ ] `revision_solver.py` 中的旧评分和候选选择逻辑必须继续收敛为 dispatcher/selector，不能重新生成跨阶段混合 patch；验证方式是代码搜索和 stage filter 单测。
- [ ] `processing_service.py` 不能继续承载 ROI 定位、候选生成、验收评分、修订求解器的长期核心逻辑；验证方式是这些能力迁移到 `roi_locator.py`、stage modules、solver modules 或等价 core 模块。
- [ ] `iterative_pipeline.py` 的视觉 prompt 上下文必须只通过 stage context 结构传递阶段信息，不能手写散落的 prompt 拼接逻辑；验证方式是 prompt payload 单测。
- [ ] `web_app.py` 必须只保留 HTTP/API/job 状态和本地 Web 服务启动，不承载阶段策略或图像处理逻辑；验证方式是代码搜索不出现 detector、patcher、candidate generation 逻辑。
- [ ] `result.json` 和 `progress.jsonl` 必须是所有阶段证据的稳定外部接口；验证方式是 schema 测试覆盖 stage、profile、candidate、patch、vision suggestion 和 rejection reason。

## 已知实现证据

本节不是完成清单，只记录当前仓库里已经存在、后续可以复用的实现证据。它不能替代 A-T 的验收项。

- `reference_profile` 已有报告字段，包含旧文字、邻字、动态墨色阈值和动态核心变浅阈值。
- 固定 `opacity >= 0.76` 硬下限已移除或降级，改为动态墨色范围。
- 视觉反馈中的核心黑度、太黑、太淡、太硬、太细已经有结构化解析路径。
- 已有 stage severity、selected reason、模型建议截断记录和替代候选记录。
- 自动 ROI 已覆盖部分字段场景，但还不是所有字段通用路径。
- 背景指标已有白影、暗影、低纹理、纹理残差、ROI 扫描残差等结构化方向。
- Web/CLI 已有 accepted/rejected、blocking stage、迭代轮次、停止原因和候选抽屉展示。
- 既有烟雾产物包括 `output/web/20260605_020104`、`output/web/20260605_020350`、`output/web/20260605_024521`、`output/web/20260605_024811`；这些是历史证据，不等同于 Case A-D 可执行回归。

## 公开验收门槛

本地流程不能宣称完善完成，除非以下全部满足：

- [ ] “设计目标转换情况”中的所有 `[ ]` 项全部关闭。
- [ ] A-T 中保留的每个 `[x]` 都有同条或相邻说明中的当前证据，证据不足时必须降级为 `[ ]`。
- [ ] 回归 Case A-D 已转成可执行用例并通过。
- [ ] 每个 stage patcher 的声明参数和拒绝跨阶段参数测试通过。
- [ ] README、设计文档和 prompt 不再声称未验证能力已经完成。
- [ ] 所有交付任务都能从 `result.json`、`progress.jsonl`、stage evidence 和候选图解释通过或失败。

任何一项未满足，都不能宣称本地流程完善完成。
