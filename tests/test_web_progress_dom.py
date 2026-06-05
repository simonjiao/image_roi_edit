from __future__ import annotations

import shutil
import subprocess
import textwrap
import unittest


class WebProgressDomTest(unittest.TestCase):
    def run_node_dom_check(self, script: str) -> None:
        self.assertIsNotNone(shutil.which("node"), "node is required for the Web DOM smoke test")
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=".",
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            self.fail(completed.stderr or completed.stdout)

    def test_progress_dom_exposes_stage_reason_candidates_and_accepted_state(self) -> None:
        self.run_node_dom_check(
            textwrap.dedent(
                r"""
                const fs = require("fs");
                const vm = require("vm");

                class Element {
                  constructor(tag = "div") {
                    this.tagName = tag;
                    this.children = [];
                    this.textContent = "";
                    this.style = {};
                    this.className = "";
                    this.dataset = {};
                    this._src = "";
                    this.classList = {
                      add: (...names) => {
                        const set = new Set(String(this.className || "").split(/\s+/).filter(Boolean));
                        names.forEach((name) => set.add(name));
                        this.className = Array.from(set).join(" ");
                      },
                      remove: (...names) => {
                        const set = new Set(String(this.className || "").split(/\s+/).filter(Boolean));
                        names.forEach((name) => set.delete(name));
                        this.className = Array.from(set).join(" ");
                      },
                    };
                  }
                  set innerHTML(value) {
                    this.children = [];
                    this.textContent = value || "";
                  }
                  get innerHTML() {
                    return this.textContent;
                  }
                  set src(value) {
                    this._src = value;
                  }
                  get src() {
                    return this._src;
                  }
                  append(...nodes) {
                    this.children.push(...nodes);
                  }
                  appendChild(node) {
                    this.children.push(node);
                  }
                  replaceChildren(...nodes) {
                    this.children = nodes;
                    this.textContent = "";
                  }
                  addEventListener() {}
                }

                function collectText(node) {
                  return [node.textContent || "", ...node.children.map(collectText)].join("\n");
                }

                function findByClass(node, className, out = []) {
                  if (String(node.className || "").split(/\s+/).includes(className)) {
                    out.push(node);
                  }
                  node.children.forEach((child) => findByClass(child, className, out));
                  return out;
                }

                const ids = {
                  fileInput: new Element("input"),
                  processButton: new Element("button"),
                  imageList: new Element("main"),
                  statusText: new Element("div"),
                  imageTemplate: { content: { firstElementChild: new Element("section") } },
                  profileSelect: new Element("select"),
                };
                const context = {
                  console,
                  document: {
                    getElementById: (id) => ids[id],
                    createElement: (tag) => new Element(tag),
                  },
                  Image: function Image() {},
                  FileReader: function FileReader() {},
                };
                vm.createContext(context);
                vm.runInContext(fs.readFileSync("web/app.js", "utf8"), context);

                const item = {
                  elements: {
                    emptyResult: new Element("div"),
                    resultImage: new Element("img"),
                    visionStatus: new Element("div"),
                    tracePanel: new Element("div"),
                  },
                };
                context.renderProgress(item, [
                  {
                    event: "revision_round_finished",
                    round: 2,
                    blocking_stage: "text_shape",
                    blocking_stage_reason: "stroke_body_too_light",
                    patch_count: 4,
                    accepted: false,
                    final_decision: "revise",
                  },
                ]);

                const text = collectText(item.elements.emptyResult);
                if (!text.includes("blocking_stage=text_shape")) throw new Error(text);
                if (!text.includes("reason=stroke_body_too_light")) throw new Error(text);
                if (!text.includes("candidates=4")) throw new Error(text);
                if (!text.includes("accepted=false")) throw new Error(text);
                const active = findByClass(item.elements.emptyResult, "active")[0];
                if (!active) throw new Error("missing active progress row");
                if (active.dataset.blockingStage !== "text_shape") throw new Error(JSON.stringify(active.dataset));
                if (active.dataset.accepted !== "false") throw new Error(JSON.stringify(active.dataset));
                """
            )
        )

    def test_failed_result_dom_shows_rejected_input_and_final_accepted_state(self) -> None:
        self.run_node_dom_check(
            textwrap.dedent(
                r"""
                const fs = require("fs");
                const vm = require("vm");

                class Element {
                  constructor(tag = "div") {
                    this.tagName = tag;
                    this.children = [];
                    this.textContent = "";
                    this.style = {};
                    this.className = "";
                    this.dataset = {};
                    this._src = "";
                    this.classList = {
                      add: (...names) => {
                        const set = new Set(String(this.className || "").split(/\s+/).filter(Boolean));
                        names.forEach((name) => set.add(name));
                        this.className = Array.from(set).join(" ");
                      },
                      remove: (...names) => {
                        const set = new Set(String(this.className || "").split(/\s+/).filter(Boolean));
                        names.forEach((name) => set.delete(name));
                        this.className = Array.from(set).join(" ");
                      },
                    };
                  }
                  set innerHTML(value) {
                    this.children = [];
                    this.textContent = value || "";
                  }
                  get innerHTML() {
                    return this.textContent;
                  }
                  set src(value) {
                    this._src = value;
                  }
                  get src() {
                    return this._src;
                  }
                  append(...nodes) {
                    this.children.push(...nodes);
                  }
                  appendChild(node) {
                    this.children.push(node);
                  }
                  replaceChildren(...nodes) {
                    this.children = nodes;
                    this.textContent = "";
                  }
                  addEventListener() {}
                }

                function collectText(node) {
                  return [node.textContent || "", ...node.children.map(collectText)].join("\n");
                }

                const ids = {
                  fileInput: new Element("input"),
                  processButton: new Element("button"),
                  imageList: new Element("main"),
                  statusText: new Element("div"),
                  imageTemplate: { content: { firstElementChild: new Element("section") } },
                  profileSelect: new Element("select"),
                };
                const context = {
                  console,
                  document: {
                    getElementById: (id) => ids[id],
                    createElement: (tag) => new Element(tag),
                  },
                  Image: function Image() {},
                  FileReader: function FileReader() {},
                };
                vm.createContext(context);
                vm.runInContext(fs.readFileSync("web/app.js", "utf8"), context);

                const item = {
                  id: "img1",
                  node: new Element("section"),
                  elements: {
                    instruction: new Element("input"),
                    resultImage: new Element("img"),
                    emptyResult: new Element("div"),
                    visionStatus: new Element("div"),
                    tracePanel: new Element("div"),
                    candidateList: new Element("div"),
                  },
                };
                context.__item = item;
                vm.runInContext("state.items.push(__item)", context);

                (async () => {
                  await context.renderProcessResult({
                    images: [
                      {
                        id: "img1",
                        ok: false,
                        accepted: false,
                        applied: false,
                        error: "无法自动定位",
                        resultDataUrl: "data:image/png;base64,cmVqZWN0ZWQ=",
                        candidates: [],
                        artifacts: { final_is_rejected_candidate: true },
                        stage_evidence: {
                          failure: {
                            failure_stage: "pre_candidate_generation",
                            pre_candidate_gate_report: {
                              failed_gate: "field_roi_selection",
                              candidate_count: 0,
                            },
                          },
                        },
                      },
                    ],
                  });
                  const trace = collectText(item.elements.tracePanel);
                  if (item.elements.resultImage.style.display !== "block") throw new Error("result image hidden");
                  if (item.elements.resultImage.src !== "data:image/png;base64,cmVqZWN0ZWQ=") {
                    throw new Error(item.elements.resultImage.src);
                  }
                  if (!item.elements.visionStatus.textContent.includes("accepted=false")) {
                    throw new Error(item.elements.visionStatus.textContent);
                  }
                  if (!trace.includes("field_roi_selection")) throw new Error(trace);
                })().catch((error) => {
                  console.error(error);
                  process.exit(1);
                });
                """
            )
        )


if __name__ == "__main__":
    unittest.main()
