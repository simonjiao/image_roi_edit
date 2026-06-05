from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebCanvasZoomTest(unittest.TestCase):
    def test_source_template_exposes_canvas_zoom_controls(self) -> None:
        html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
        css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

        for class_name in (
            "zoom-toolbar",
            "zoom-out",
            "zoom-slider",
            "zoom-value",
            "zoom-in",
            "zoom-reset",
            "rotate-left",
            "rotate-right",
            "open-editor",
            "image-editor-modal",
            "image-editor-canvas",
        ):
            self.assertIn(class_name, html)

        self.assertIn(".source-pane .canvas-shell", css)
        self.assertIn(".image-editor-modal", css)
        self.assertIn(".image-editor-canvas-shell", css)
        self.assertIn("overflow: auto", css)
        self.assertIn("max-width: none", css)
        self.assertIn("max-height: none", css)

    def test_zoom_keeps_canvas_pointer_coordinates_in_image_space(self) -> None:
        self.assertIsNotNone(shutil.which("node"), "node is required for the Web zoom smoke test")
        script = textwrap.dedent(
            r"""
            const fs = require("fs");
            const vm = require("vm");

            class Element {
              constructor(tag = "div") {
                this.tagName = tag;
                this.style = {};
                this.value = "";
                this.textContent = "";
                this.clientWidth = 0;
                this.clientHeight = 0;
              }
              addEventListener() {}
              getBoundingClientRect() {
                return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight };
              }
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

            const canvas = new Element("canvas");
            canvas.width = 1000;
            canvas.height = 500;
            canvas.getBoundingClientRect = () => ({ left: 10, top: 20, width: 500, height: 250 });
            const point = context.canvasPoint(canvas, { clientX: 260, clientY: 145 });
            if (point.x !== 500 || point.y !== 250) {
              throw new Error(JSON.stringify(point));
            }

            const shell = new Element("div");
            shell.clientWidth = 524;
            shell.clientHeight = 300;
            const slider = new Element("input");
            const zoomValue = new Element("div");
            const item = {
              image: { naturalWidth: 1000, naturalHeight: 500 },
              elements: {
                canvas,
                canvasShell: shell,
                zoomSlider: slider,
                zoomValue,
              },
            };

            context.resetItemZoomToFit(item);
            if (item.zoom !== 0.5) throw new Error(`fit zoom ${item.zoom}`);
            if (canvas.style.width !== "500px" || canvas.style.height !== "250px") {
              throw new Error(JSON.stringify(canvas.style));
            }
            if (slider.value !== "50" || zoomValue.textContent !== "50%") {
              throw new Error(`${slider.value} ${zoomValue.textContent}`);
            }

            context.setItemZoom(item, 2);
            if (item.zoom !== 2 || item.zoomMode !== "manual") throw new Error(`${item.zoom} ${item.zoomMode}`);
            if (canvas.style.width !== "2000px" || canvas.style.height !== "1000px") {
              throw new Error(JSON.stringify(canvas.style));
            }
            if (slider.value !== "200" || zoomValue.textContent !== "200%") {
              throw new Error(`${slider.value} ${zoomValue.textContent}`);
            }

            const clockwise = context.rotateRectForImageTurn({ x: 10, y: 20, w: 30, h: 40 }, 100, 80, 1);
            if (JSON.stringify(clockwise) !== JSON.stringify({ x: 20, y: 10, w: 40, h: 30 })) {
              throw new Error(JSON.stringify(clockwise));
            }
            const counterClockwise = context.rotateRectForImageTurn({ x: 10, y: 20, w: 30, h: 40 }, 100, 80, -1);
            if (JSON.stringify(counterClockwise) !== JSON.stringify({ x: 20, y: 60, w: 40, h: 30 })) {
              throw new Error(JSON.stringify(counterClockwise));
            }
            """
        )
        completed = subprocess.run(
            ["node", "-e", script],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            self.fail(completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
