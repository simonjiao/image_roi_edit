# Workflow Architecture Boundaries

本文档只记录 ROI 替换 workflow 的架构边界。它不是主能力 checklist；
主能力验收仍以 [`workflow_checklist.md`](workflow_checklist.md) 为准。

## 模块边界

- `src/roi_image_edit/run_artifacts.py` 承载 request、result、progress 和 prompt stage context 的运行产物 helper。
- `src/roi_image_edit/stages.py` 是 stage contract 的入口，承载 `StageSpec`、`StageResult`、detector mapping 和 prompt context。
- `src/roi_image_edit/stage_profiles.py` 保留内部策略注册表；Web 用户可见 profile resolver 要被分类驱动 workflow 取代。
- `src/roi_image_edit/stage_patchers.py` 是 stage patcher filter/dispatch 的入口。
- `src/roi_image_edit/local_validation.py` 中的 `local_*_issues` 是 detector 底层函数，不直接承担 stage policy。
- `src/roi_image_edit/revision_selector.py` 承载 delivery checks、revision selection scoring 和 candidate parameter constraints。
- `src/roi_image_edit/revision_solver.py` 承载 stage-specific candidate grids，不重新生成跨阶段混合 patch。
- `src/roi_image_edit/region_processing.py` 承载 region-level candidate rendering、vision checks、stage candidate evidence 和 revision loop。
- `src/roi_image_edit/processing_service.py` 保留 payload/job 编排，避免承载 ROI 定位、候选生成、验收评分或修订求解核心逻辑。
- `src/roi_image_edit/iterative_pipeline.py` 保留底层渲染、指标、字体、硬校验和 vision-client primitives；视觉 prompt 阶段信息通过 stage context 结构传递。
- `src/roi_image_edit/web_app.py` 只保留 HTTP/API/job 状态和本地 Web 服务启动，不承载阶段策略或图像处理逻辑。
