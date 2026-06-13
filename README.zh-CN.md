# ROI 图片文字替换

English README: [`README.md`](README.md).

许可证：[`GPL-3.0-or-later`](LICENSE)。

## 工作流

1. 使用项目内打包的 prompt 资产和当前流水线代码。
2. 用本地 PIL/OpenCV 生成 ROI 候选图。
3. 输出硬校验报告，检查图片尺寸、ROI 外像素、边缘像素和受保护文字框。
4. 使用配置好的视觉模型和对应 prompt 对候选图排序。
5. 只接受小步参数补丁。
6. 最多迭代到 `--max-iterations`。
7. 最后执行硬校验和最终验收 prompt。

实现规则和持续沉淀的验收标准记录在
[`docs/roi_text_replacement_rules.md`](docs/roi_text_replacement_rules.md)。
不折中的 ROI 替换 workflow checklist 记录在
[`docs/workflow_checklist.md`](docs/workflow_checklist.md)。
修改 ROI 流水线时，如果工作流、硬门槛、迭代策略或视觉验收标准发生变化，
需要同步更新这份规则文档。

## 安装

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Prompt 文本文件打包在
[`src/roi_image_edit/prompts/`](src/roi_image_edit/prompts/) 下。CLI、Web
和环境检查都只从这个 package resource 目录读取 prompt。

## 模块边界

- `src/roi_image_edit/web_app.py`：只保留 Web 入口。负责静态文件、
  `/api/process*`、内存 job 状态和本地 HTTP 服务启动。
- `src/roi_image_edit/processing_service.py`：Web 和 CLI 共用的图片处理服务。
  负责解析处理 payload、编排图片/区域处理、执行视觉检查和写入运行产物。
- `src/roi_image_edit/roi_locator.py`：负责指令解析、文档方向评分、字段 ROI 选择、
  旧文字槽位检测和 `RenderPlan` 构建。
- `src/roi_image_edit/local_validation.py`：负责原图参照画像、本地/硬校验报告、
  背景/照片指标和本地候选评分。
- `src/roi_image_edit/revision_solver.py`：负责修订候选评分、形态 reset 候选和
  最终字体修订候选。
- `src/roi_image_edit/stage_policy.py`：阶段顺序和 Optimization Step 策略。
  Web 代码只能通过处理服务使用这些策略，不能在 `web_app.py` 里重新定义。
- `src/roi_image_edit/stages.py`：负责可执行阶段定义、`StageSpec`、`StageResult`、
  detector mapping 和阶段报告生成。
- `src/roi_image_edit/stage_profiles.py`：负责照片/扫描、干净数字图、低分辨率缩略图、
  手动 ROI 快速处理等 profile。
- `src/roi_image_edit/stage_patchers.py`：负责按阶段生成 patch、解析模型 patch、
  stage filter 和 revision patch dispatch。
- `src/roi_image_edit/iterative_pipeline.py`：更底层的渲染、指标、字体、硬校验和
  OpenAI 兼容视觉客户端基础能力。

修改流水线逻辑时，HTTP/UI 相关内容留在 `web_app.py`，编排留在
`processing_service.py`，领域逻辑放入对应 core 模块。

## CLI

检查依赖、打包 prompt、API 配置和字体可用性：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py check-env
```

安装能自动安装的推荐字体：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py install-fonts
```

运行流水线：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py run \
  --metadata /path/to/metadata.json \
  --vision auto \
  --acceptance-mode strict \
  --max-iterations 8 \
  --max-candidates 12
```

在项目、本机用户和系统字体目录下扫描所有可加载字体，再进行排序：

```bash
.venv/bin/python scripts/roi_image_edit_cli.py run \
  --metadata /path/to/metadata.json \
  --vision auto \
  --font-source scan \
  --font-candidate-pool-size 12 \
  --max-candidates 48 \
  --sheet-scale 6 \
  --sheet-cols 4
```

启动本地 Web UI：

```bash
.venv/bin/python scripts/roi_image_edit_web.py --host 127.0.0.1 --port 8787
```

Web 页面支持批量上传图片。左侧原图上可以画一个或多个矩形，输入类似
`旧文字替换为新文字` 的修改说明，然后点击 `处理全部`。右侧显示修改后的图片。
使用 `>>>` 可以打开候选图抽屉，最多显示 5 张本地候选预览。
候选抽屉会显示 profile、blocking stage、候选来源、Optimization Step、模型建议数量
和被应用的 patch 摘要。
结果区域也会显示当前 blocking stage、迭代轮次、停止原因和下一轮计划；如果最终
候选未被接受，会明确标注为 rejected candidate，不会当作已应用交付图。
顶部 profile 控件会写入每次任务，`photo_scan` 是默认照片/扫描件流程；
`clean_digital` 不启用照片质感阶段。

每次 Web 运行都会保存到 `output/web/<run_id>/`，包括 `request.json`、
`result.json`、原图和最终图。如果用户画的矩形大于文字本身，Web 流水线会先把
编辑目标收缩到矩形内检测到的旧文字组件。Web 处理也会运行视觉候选排序和最终验收
prompt；每个区域的视觉产物写入
`output/web/<run_id>/regions/<region_id>/`。
每个区域还会写入 `stage_evidence/summary.json`，并为 `text_shape`、
`ink_gray_balance`、`photo_texture`、`background_cleanup` 保存该阶段 top
候选或明确说明没有该阶段候选。

运行进度会打印到 stderr，同时写入 `output/<run_id>/progress.jsonl`。另一个终端
可以这样跟踪：

```bash
tail -f output/<run_id>/progress.jsonl
```

每个 revision round 都会记录 blocking stage、stage 来源、stage policy、候选数、
被选中的候选、本地/模型冲突，以及模型建议是否被本地约束截断。

严格模式会同时检查灰度覆盖和字体风格。已知旧文字时，字体风格门槛会用每个本地
候选字体渲染旧文字 ROI，并且只把受保护标签 `名` 作为辅助风格参考。它还会输出
字体类别，并在存在宋体/明朝或 CJK 衬线候选时惩罚微软雅黑这类现代无衬线字体。

严格门槛也会用 `--max-core-mean-gray-delta`、
`--max-edge-mean-gray-delta` 和 `--min-dark-pixel-ratio` 检查核心笔画深度和
灰边相似度，所以高度正确但明显偏浅或边缘过硬的候选，会在视觉模型验收前被拒绝。
默认最小深色像素覆盖是 `0.88`，用于避免候选虽然平均灰度接近，但笔画覆盖不足。
`--max-core-lighten-delta` 和 `--max-edge-lighten-delta` 会限制方向性的变浅问题，
防止一个视觉上看似可接受但仍偏淡的候选替换掉本地指标更接近的候选。

严格门槛还会通过 `--max-char-center-dx` 和
`--max-char-center-distance-delta` 检查逐字中心位置和中心间距，避免某个字横向漂移
到字距不再匹配原文。对于这些小图，默认逐字水平中心限制是 2px。

视觉模型只能在 `--max-model-local-score-delta` 允许的范围内覆盖本地 fallback，
并且不能选择旧文字字体风格比例比本地 fallback 差超过
`--max-model-font-style-ratio-delta` 的候选。

脚本仍然会渲染并硬校验所有生成候选，但默认只把本地排序靠前、数量不超过
`--vision-candidate-limit` 的候选发送给视觉候选排序 prompt。这样可以把阶段 prompt
控制在 OpenAI 兼容视觉网关可处理的大小内，同时保留完整的本地硬校验报告。

本地评分还会在 `--blur-score-free-margin` 之后使用
`--blur-score-weight` 给模糊程度一个小偏好，让其它指标接近但更清晰的候选能优先于
过软的渲染。`--ink-gain` 会在模糊前增强字形 alpha，`--core-ink-gain` 只在模糊后
加深高 alpha 的笔画核心，`--core-darken-strength` 会加深已经渲染出来的高置信核心
像素，但不扩张低 alpha 的抗锯齿边缘；`--alpha-contrast` 会收紧模糊后的 alpha
过渡，`--stroke-opacity` 会增加一个小数强度的外描边。

本地评分会分别比较真实黑色核心（`<55` / `<70`）、深灰核心带（`55-70`）、
中灰笔画内部（`70-90`）、更宽的深色主体（`<90` / `<120`）以及 `120-165`
外层灰边。硬报告还包含逐字灰度带，因此替换结果不能用全 ROI 总量合格来掩盖某个
目标字偏弱的问题。`55-70` 或 `70-90` 像素过多会让笔画显得发灰，即使 `<70`
总数可接受；`120-165` 像素过多则会呈现灰蒙蒙的外轮廓。

当前评分更偏向把灰色笔画内部转成真正的黑色核心像素，而不是靠增加灰边获得深度；
硬门槛仍然会限制总深色像素面积。模型调参补丁如果视觉上通过，但让本地真黑核心和
灰度带指标变差，也会被拒绝。这只是评分过程；最终图片仍然只会修改目标 ROI 内部。

## 字体

脚本按源文档推荐顺序检查字体，并用于候选生成。项目本地 Windows 字体可以放到
`fonts/` 下：

- `fonts/simsun.ttc`
- `fonts/simfang.ttf`
- `fonts/msyh.ttc`

`fonts/` 下的字体二进制是本地运行资产，已被 Git 忽略。本机可以继续保留这些授权
字体；除非确认字体允许再分发，否则不要把专有字体文件提交到仓库。

在当前 macOS 环境里，系统注册了 `PingFangUI.ttc`，但 Pillow 报告它是
`unknown file format`，所以在提供 Pillow 可加载的 PingFang 文件之前，流水线会把
它排除在可用渲染顺序之外。

默认情况下，`run` 使用 `--font-ranking style`。当已知旧文字时，它会根据当前图片
的旧文字 ROI 风格分数重新排序可用字体。使用 `--font-ranking document` 可以保持
文档顺序。metadata 没有提供旧文字时，使用 `--source-text` 明确传入旧文字。

使用 `--font-source scan` 会扫描 `fonts/`、`~/Library/Fonts`、
`/Library/Fonts`、`/System/Library/Fonts` 和
`/System/Library/Fonts/Supplemental`。扫描到的字体必须能被 Pillow 加载，并且能
渲染目标文字、旧文字和风格参考文字里的每个字符。对任何必需字符回退成缺字框的
字体，会写入 `summary.json` 的 `rejected_font_candidates`。使用扫描模式时可以
提高 `--font-candidate-pool-size`，让更多排序后的字体进入候选生成。

## 运行 fixture 流水线

```bash
.venv/bin/python scripts/run_iterative_roi.py \
  --metadata /path/to/metadata.json \
  --vision auto \
  --max-iterations 8 \
  --max-candidates 12
```

脚本从这里读取 API 配置：

```text
.env
```

使用 `--vision off` 可以只运行本地候选生成和硬校验。

## 输出

每次运行都会在 `output/` 下写入一个带时间戳的目录，包括：

- `original_crop.png`
- `iteration_*/contact_sheet.png`
- `iteration_*/hard_check_report.json`
- `iteration_*/visual_eval_candidate_rank.json`
- `iteration_*/visual_eval_tuning.json`
- `final_crop.png`
- `final_full.png`
- `final_acceptance.json`
- `final_acceptance_strict.json`
- `progress.jsonl`
- `summary.json`
