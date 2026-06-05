const fileInput = document.getElementById("fileInput");
const processButton = document.getElementById("processButton");
const imageList = document.getElementById("imageList");
const statusText = document.getElementById("statusText");
const template = document.getElementById("imageTemplate");
const profileSelect = document.getElementById("profileSelect");

const state = {
  items: [],
  dragging: null,
};

function uid(prefix = "id") {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function setStatus(text) {
  statusText.textContent = text;
}

function updateProcessButton() {
  processButton.disabled = !state.items.some(
    (item) => item.regions.length > 0 || item.elements?.instruction?.value.trim(),
  );
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = dataUrl;
  });
}

function getItem(itemId) {
  return state.items.find((item) => item.id === itemId);
}

function canvasPoint(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.max(0, Math.min(canvas.width, (event.clientX - rect.left) * scaleX)),
    y: Math.max(0, Math.min(canvas.height, (event.clientY - rect.top) * scaleY)),
  };
}

function normalizedRect(start, end) {
  const x = Math.min(start.x, end.x);
  const y = Math.min(start.y, end.y);
  const w = Math.abs(end.x - start.x);
  const h = Math.abs(end.y - start.y);
  return {
    x: Math.round(x),
    y: Math.round(y),
    w: Math.round(w),
    h: Math.round(h),
  };
}

function drawCanvas(item) {
  const canvas = item.elements.canvas;
  const ctx = canvas.getContext("2d");
  canvas.width = item.image.naturalWidth;
  canvas.height = item.image.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(item.image, 0, 0);

  ctx.lineWidth = Math.max(2, Math.round(Math.min(canvas.width, canvas.height) / 300));
  ctx.font = `${Math.max(12, Math.round(canvas.width / 80))}px ui-sans-serif`;
  item.regions.forEach((region, index) => {
    ctx.strokeStyle = region.auto ? "#9a5b00" : "#147d64";
    ctx.fillStyle = region.auto ? "rgba(154, 91, 0, 0.13)" : "rgba(20, 125, 100, 0.12)";
    ctx.fillRect(region.rect.x, region.rect.y, region.rect.w, region.rect.h);
    ctx.strokeRect(region.rect.x, region.rect.y, region.rect.w, region.rect.h);
    ctx.fillStyle = region.auto ? "#7b4700" : "#0d604c";
    ctx.fillText(String(index + 1), region.rect.x + 4, Math.max(14, region.rect.y - 4));
  });

  if (state.dragging && state.dragging.itemId === item.id) {
    const rect = normalizedRect(state.dragging.start, state.dragging.current);
    ctx.setLineDash([8, 5]);
    ctx.strokeStyle = "#b85f00";
    ctx.fillStyle = "rgba(184, 95, 0, 0.13)";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    ctx.setLineDash([]);
  }
}

function updateRegionCount(item) {
  const autoCount = item.regions.filter((region) => region.auto).length;
  item.elements.regionCount.textContent = autoCount
    ? `${item.regions.length} 个矩形，${autoCount} 个自动`
    : `${item.regions.length} 个矩形`;
  updateProcessButton();
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return String(value);
  }
  return Math.abs(number) >= 10 ? number.toFixed(1) : number.toFixed(3);
}

function candidateMetaLines(candidate) {
  const metrics = candidate.metrics || {};
  const background = candidate.background || {};
  const issues = Array.isArray(background.issues)
    ? background.issues.filter(Boolean).slice(0, 2).join(", ")
    : "";
  const lines = [
    `${candidate.index}. ${candidate.kind || "candidate"} | ${candidate.label}`,
    `profile ${candidate.pipeline_profile || "photo_scan"} | stage ${candidate.blocking_stage || "pass"} | severity ${formatNumber(candidate.stage_severity)} | score ${formatNumber(candidate.score)}`,
    `lt55 ${formatNumber(metrics.lt55_delta)} | 55-70 ${formatNumber(metrics.band_55_70_delta)} | 70-90 ${formatNumber(metrics.band_70_90_delta)}`,
    `background ${issues || "ok"} | var ${formatNumber(background.patch_variance_ratio)} | residual ${formatNumber(background.residual_energy_ratio)}`,
  ];
  if (candidate.selection_reason) {
    lines.push(`selected: ${candidate.selection_reason}`);
  }
  return lines;
}

function renderCandidates(item, candidates) {
  const list = item.elements.candidateList;
  list.innerHTML = "";
  if (!candidates || candidates.length === 0) {
    list.innerHTML = '<div class="candidate-meta">没有候选图</div>';
    return;
  }
  candidates.slice(0, 5).forEach((candidate) => {
    const box = document.createElement("div");
    box.className = "candidate-item";
    const meta = document.createElement("div");
    meta.className = "candidate-meta";
    candidateMetaLines(candidate).forEach((line) => {
      const row = document.createElement("div");
      row.className = "candidate-meta-line";
      row.textContent = line;
      meta.appendChild(row);
    });
    const img = document.createElement("img");
    img.alt = "";
    img.src = candidate.dataUrl;
    box.append(meta, img);
    list.appendChild(box);
  });
}

function renderTrace(item, entry) {
  const panel = item.elements.tracePanel;
  panel.innerHTML = "";
  const regions = Array.isArray(entry.regions) ? entry.regions : [];
  if (!regions.length) {
    panel.style.display = "none";
    return;
  }
  regions.forEach((region) => {
    const trace = region.summary?.trace || {};
    const vision = region.summary?.vision || {};
    const nextPlan = trace.next_round_plan || vision.next_round_plan || {};
    const lines = [
      `${region.id || "region"} | ${region.accepted ? "accepted" : "rejected"} | profile ${
        region.summary?.plan?.pipeline_profile || "photo_scan"
      } | stage ${trace.final_blocking_stage || "pass"} | rounds ${trace.revision_round_count ?? 0}`,
    ];
    if (trace.last_round_stop_reason) {
      lines.push(`stop: ${trace.last_round_stop_reason}`);
    }
    if (trace.last_round_selected_reason) {
      lines.push(`selected: ${trace.last_round_selected_reason}`);
    }
    if (nextPlan.blocking_stage || nextPlan.actions?.length) {
      lines.push(
        `next: ${nextPlan.blocking_stage || "vision"} | ${(nextPlan.actions || []).slice(0, 1).join(" ")}`,
      );
    }
    lines.forEach((line) => {
      const row = document.createElement("div");
      row.className = "trace-line";
      row.textContent = line;
      panel.appendChild(row);
    });
  });
  panel.style.display = "block";
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function progressEventText(record) {
  const event = record?.event || "queued";
  switch (event) {
    case "queued":
      return "等待开始处理";
    case "run_started":
      return `任务已创建：profile ${record.pipeline_profile || "photo_scan"}`;
    case "image_started":
      return `开始处理：${record.source_text || "-"} -> ${record.target_text || "-"}，profile ${
        record.pipeline_profile || "photo_scan"
      }`;
    case "auto_roi_finished":
      return `自动定位完成：${record.region_count ?? 0} 个区域，方向 ${
        record.orientation || "none"
      }，方向分 ${formatNumber(record.direction_score)}`;
    case "region_started":
      return `开始处理区域：${record.source_text || "-"} -> ${record.target_text || "-"}`;
    case "region_candidates_started":
      return `生成候选图：${record.candidate_count ?? "-"} 个`;
    case "region_candidates_finished":
      return `候选图生成完成：${record.rendered ?? "-"} 个，最佳分 ${formatNumber(record.best_score)}`;
    case "region_initial_acceptance":
      return `初始验收：${record.accepted ? "通过" : "未通过"}，硬边界 ${
        record.hard_boundary_pass ? "通过" : "未通过"
      }，严格门槛 ${record.strict_pass ? "通过" : "未通过"}`;
    case "revision_round_started":
      return `第 ${record.round ?? "-"} 轮调参开始：${record.basis_blocking_stage || "vision"}，profile ${
        record.pipeline_profile || "photo_scan"
      }`;
    case "revision_round_candidates":
      return `第 ${record.round ?? "-"} 轮候选：${record.patch_count ?? 0} 个修正，${
        record.shape_reset_count ?? 0
      } 个形态重算`;
    case "revision_round_finished":
      return `第 ${record.round ?? "-"} 轮完成：${record.accepted ? "通过" : "未通过"}，${
        record.blocking_stage || record.current_blocking_stage || "pass"
      }`;
    case "finalist_revision_started":
      return `终检候选开始：${record.candidate_count ?? 0} 个`;
    case "finalist_revision_candidate_started":
      return `终检候选 ${record.index ?? "-"}/${record.total ?? "-"}：${
        record.font_name || "font"
      } ${record.font_size || ""}px`;
    case "finalist_revision_candidate_finished":
      return `终检候选 ${record.index ?? "-"}/${record.total ?? "-"} 完成：${
        record.accepted ? "通过" : "未通过"
      }，${record.blocking_stage || record.final_decision || "revise"}`;
    case "finalist_revision_finished":
      return `终检候选结束：${record.accepted ? "通过" : "未通过"}`;
    case "region_finished":
      return `区域完成：${record.accepted ? "通过" : "未通过"}，迭代 ${
        record.revision_rounds ?? 0
      } 轮`;
    case "image_finished":
      return `图片完成：${record.accepted ? "验收通过" : "未通过，显示最后候选"}`;
    case "image_failed":
    case "job_failed":
      return `处理失败：${record.error || "未知错误"}`;
    case "run_finished":
      return "任务结束";
    default:
      return event;
  }
}

function renderProgress(item, events) {
  const visibleEvents = events.length ? events : [{ event: "queued" }];
  const latest = visibleEvents[visibleEvents.length - 1] || { event: "queued" };
  const panel = document.createElement("div");
  panel.className = "progress-panel";

  const title = document.createElement("div");
  title.className = "progress-title";
  title.textContent = "处理中";

  const current = document.createElement("div");
  current.className = "progress-current";
  current.textContent = progressEventText(latest);

  const list = document.createElement("div");
  list.className = "progress-list";
  visibleEvents.slice(-6).forEach((record, index, sliced) => {
    const row = document.createElement("div");
    row.className = index === sliced.length - 1 ? "progress-line active" : "progress-line";
    row.textContent = progressEventText(record);
    list.appendChild(row);
  });

  panel.append(title, current, list);
  item.elements.emptyResult.replaceChildren(panel);
  item.elements.emptyResult.style.display = "flex";
  item.elements.resultImage.style.display = "none";
  item.elements.visionStatus.style.display = "none";
  item.elements.tracePanel.style.display = "none";
}

function renderProgressSnapshot(events, processable) {
  const commonEvents = events.filter((record) => !record.image_id);
  const byImage = new Map();
  events.forEach((record) => {
    if (!record.image_id) {
      return;
    }
    const imageEvents = byImage.get(record.image_id) || [];
    imageEvents.push(record);
    byImage.set(record.image_id, imageEvents);
  });
  processable.forEach((item) => {
    renderProgress(item, [{ event: "queued" }, ...commonEvents, ...(byImage.get(item.id) || [])]);
  });
}

async function pollProcessJob(jobId, processable) {
  for (;;) {
    const response = await fetch(`/api/process/status?job_id=${encodeURIComponent(jobId)}`);
    const status = await response.json();
    if (!response.ok || !status.ok) {
      throw new Error(status.error || `HTTP ${response.status}`);
    }
    renderProgressSnapshot(status.events || [], processable);
    if (status.done) {
      if (status.error) {
        throw new Error(status.error);
      }
      return status.result;
    }
    await sleep(800);
  }
}

async function renderProcessResult(result) {
  for (const entry of result.images) {
    const item = getItem(entry.id);
    if (!item) {
      continue;
    }
    item.node.classList.remove("processing");
    item.elements.instruction.disabled = false;
    if (!entry.ok) {
      item.node.classList.add("error");
      renderCandidates(item, []);
      item.elements.emptyResult.style.display = "block";
      item.elements.emptyResult.textContent = entry.error || "处理失败";
      item.elements.resultImage.style.display = "none";
      item.elements.visionStatus.style.display = "none";
      item.elements.tracePanel.style.display = "none";
      continue;
    }
    if (entry.sourceDataUrl && entry.sourceDataUrl !== item.dataUrl) {
      item.dataUrl = entry.sourceDataUrl;
      item.image = await loadImage(entry.sourceDataUrl);
    }
    if (Array.isArray(entry.regions) && entry.regions.length > 0) {
      item.regions = entry.regions.map((region) => {
        const roi = region.roi || [0, 0, 0, 0];
        return {
          id: region.id,
          auto: Boolean(region.auto),
          rect: {
            x: roi[0],
            y: roi[1],
            w: Math.max(0, roi[2] - roi[0]),
            h: Math.max(0, roi[3] - roi[1]),
          },
        };
      });
      drawCanvas(item);
      updateRegionCount(item);
    }
    item.elements.resultImage.src = entry.resultDataUrl;
    item.elements.resultImage.style.display = "block";
    item.elements.emptyResult.style.display = "none";
    const revisionRounds = (entry.regions || []).reduce((total, region) => {
      const rounds = region?.summary?.vision?.revision_rounds;
      return total + (Array.isArray(rounds) ? rounds.length : 0);
    }, 0);
    const roundsText = revisionRounds ? `，已迭代 ${revisionRounds} 轮` : "";
    const rejectedArtifactText = entry.artifacts?.final_is_rejected_candidate
      ? "，右侧显示最后候选图"
      : "";
    item.elements.visionStatus.textContent = entry.accepted
      ? `视觉验收通过${roundsText}`
      : `视觉验收未通过${roundsText}${rejectedArtifactText}，未应用为交付图`;
    item.elements.visionStatus.className = `vision-status ${entry.accepted ? "pass" : "fail"}`;
    item.elements.visionStatus.style.display = "block";
    renderTrace(item, entry);
    renderCandidates(item, entry.candidates);
  }
}

function renderItem(item) {
  const node = template.content.firstElementChild.cloneNode(true);
  item.node = node;
  item.elements = {
    fileName: node.querySelector(".file-name"),
    removeButton: node.querySelector(".remove-button"),
    canvas: node.querySelector(".source-canvas"),
    instruction: node.querySelector(".instruction-input"),
    clearRects: node.querySelector(".clear-rects"),
    regionCount: node.querySelector(".region-count"),
    resultShell: node.querySelector(".result-shell"),
    resultImage: node.querySelector(".result-image"),
    emptyResult: node.querySelector(".empty-result"),
    visionStatus: node.querySelector(".vision-status"),
    tracePanel: node.querySelector(".trace-panel"),
    drawerToggle: node.querySelector(".drawer-toggle"),
    candidateList: node.querySelector(".candidate-list"),
  };
  item.elements.fileName.textContent = item.filename;
  item.elements.instruction.value = item.instruction;

  item.elements.removeButton.addEventListener("click", () => {
    state.items = state.items.filter((entry) => entry.id !== item.id);
    node.remove();
    updateProcessButton();
    setStatus(state.items.length ? `${state.items.length} 张图片` : "等待图片");
  });

  item.elements.clearRects.addEventListener("click", () => {
    item.regions = [];
    drawCanvas(item);
    updateRegionCount(item);
  });

  item.elements.instruction.addEventListener("input", updateProcessButton);

  item.elements.drawerToggle.addEventListener("click", () => {
    item.elements.resultShell.classList.toggle("drawer-open");
    item.elements.drawerToggle.textContent = item.elements.resultShell.classList.contains("drawer-open")
      ? "<<<"
      : ">>>";
  });

  item.elements.canvas.addEventListener("pointerdown", (event) => {
    item.elements.canvas.setPointerCapture(event.pointerId);
    const point = canvasPoint(item.elements.canvas, event);
    state.dragging = { itemId: item.id, start: point, current: point };
    drawCanvas(item);
  });

  item.elements.canvas.addEventListener("pointermove", (event) => {
    if (!state.dragging || state.dragging.itemId !== item.id) {
      return;
    }
    state.dragging.current = canvasPoint(item.elements.canvas, event);
    drawCanvas(item);
  });

  item.elements.canvas.addEventListener("pointerup", () => {
    if (!state.dragging || state.dragging.itemId !== item.id) {
      return;
    }
    const rect = normalizedRect(state.dragging.start, state.dragging.current);
    state.dragging = null;
    if (rect.w >= 4 && rect.h >= 4) {
      item.regions.push({ id: uid("region"), rect });
    }
    drawCanvas(item);
    updateRegionCount(item);
  });

  imageList.appendChild(node);
  drawCanvas(item);
  updateRegionCount(item);
}

async function addFiles(files) {
  if (!files.length) {
    return;
  }
  setStatus("读取图片");
  for (const file of files) {
    const dataUrl = await fileToDataUrl(file);
    const image = await loadImage(dataUrl);
    const item = {
      id: uid("image"),
      filename: file.name,
      dataUrl,
      image,
      regions: [],
      instruction: "",
      elements: {},
      node: null,
    };
    state.items.push(item);
    renderItem(item);
  }
  setStatus(`${state.items.length} 张图片`);
}

async function processAll() {
  const processable = state.items.filter(
    (item) => item.regions.length > 0 || item.elements.instruction.value.trim(),
  );
  if (!processable.length) {
    return;
  }
  processButton.disabled = true;
  setStatus("处理中");
  processable.forEach((item) => {
    item.node.classList.add("processing");
    item.node.classList.remove("error");
    item.elements.instruction.disabled = true;
    renderProgress(item, [{ event: "queued" }]);
  });

  const payload = {
    profile: profileSelect.value,
    maxCandidates: 120,
    maxRevisionRounds: 12,
    images: processable.map((item) => ({
      id: item.id,
      filename: item.filename,
      dataUrl: item.dataUrl,
      instruction: item.elements.instruction.value,
      regions: item.regions,
    })),
  };

  try {
    const startResponse = await fetch("/api/process/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const started = await startResponse.json();
    if (!startResponse.ok || !started.ok) {
      throw new Error(started.error || `HTTP ${startResponse.status}`);
    }
    const result = await pollProcessJob(started.jobId, processable);
    await renderProcessResult(result);
    setStatus("处理完成");
  } catch (error) {
    setStatus("处理失败");
    processable.forEach((item) => {
      item.node.classList.remove("processing");
      item.node.classList.add("error");
      item.elements.instruction.disabled = false;
      item.elements.emptyResult.textContent = error.message;
      item.elements.emptyResult.style.display = "block";
      item.elements.resultImage.style.display = "none";
      item.elements.visionStatus.style.display = "none";
      item.elements.tracePanel.style.display = "none";
    });
  } finally {
    updateProcessButton();
  }
}

fileInput.addEventListener("change", async () => {
  await addFiles(Array.from(fileInput.files || []));
  fileInput.value = "";
});

processButton.addEventListener("click", processAll);
