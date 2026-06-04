from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from roi_image_edit.prompt_assets import PROMPT_NAMES, prompt_exists, prompt_resource


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_font_path(filename: str) -> str:
    return str(PROJECT_ROOT / "fonts" / filename)


@dataclass(frozen=True)
class FontSpec:
    name: str
    purpose: str
    priority: int
    paths: tuple[str, ...]
    install_hint: str
    auto_install: str | None = None
    category: str = "fallback_sans"


RECOMMENDED_FONTS: tuple[FontSpec, ...] = (
    FontSpec(
        name="GBSN",
        purpose="Preferred Song/Ming-style low-resolution candidate from the source document.",
        priority=10,
        paths=(
            "~/Library/Fonts/gbsn00lp.ttf",
            "/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf",
        ),
        install_hint="Install Arphic GBSN, for example from Debian package fonts-arphic-gbsn00lp.",
        auto_install="debian:fonts-arphic-gbsn00lp",
        category="song_ming",
    ),
    FontSpec(
        name="NotoSerif",
        purpose="Preferred CJK serif fallback from the source document.",
        priority=20,
        paths=(
            "~/Library/Fonts/NotoSerifCJK.ttc",
            "/Library/Fonts/NotoSerifCJK.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK.ttc",
        ),
        install_hint="brew install --cask font-noto-serif-cjk",
        auto_install="brew:font-noto-serif-cjk",
        category="cjk_serif",
    ),
    FontSpec(
        name="Songti",
        purpose="macOS serif fallback.",
        priority=30,
        paths=(
            "/System/Library/Fonts/Supplemental/Songti.ttc",
            "/Library/Fonts/Songti.ttc",
            "~/Library/Fonts/Songti.ttc",
        ),
        install_hint="Bundled on many macOS systems.",
        category="song_ming",
    ),
    FontSpec(
        name="SimSun",
        purpose="Windows Song-style fallback.",
        priority=40,
        paths=(
            project_font_path("simsun.ttc"),
            project_font_path("simsun.ttf"),
            "C:/Windows/Fonts/simsun.ttc",
            "C:/Windows/Fonts/simsun.ttf",
        ),
        install_hint="Place simsun.ttc under project fonts/ or use a Windows system font path.",
        category="song_ming",
    ),
    FontSpec(
        name="FangSong",
        purpose="Windows FangSong-style fallback.",
        priority=45,
        paths=(
            project_font_path("simfang.ttf"),
            project_font_path("simfang.ttc"),
            project_font_path("fangsong.ttf"),
            project_font_path("fangsong.ttc"),
            "C:/Windows/Fonts/simfang.ttf",
            "C:/Windows/Fonts/simfang.ttc",
        ),
        install_hint="Place simfang.ttf under project fonts/ or use a Windows system font path.",
        category="cjk_serif",
    ),
    FontSpec(
        name="NotoSans",
        purpose="CJK sans fallback from the source document.",
        priority=50,
        paths=(
            "~/Library/Fonts/NotoSansCJK.ttc",
            "/Library/Fonts/NotoSansCJK.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK.ttc",
        ),
        install_hint="brew install --cask font-noto-sans-cjk",
        auto_install="brew:font-noto-sans-cjk",
        category="modern_sans",
    ),
    FontSpec(
        name="UMing",
        purpose="Arphic Ming fallback from the source document.",
        priority=60,
        paths=(
            "~/Library/Fonts/uming.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ),
        install_hint="Install Arphic UMing, for example from Debian package fonts-arphic-uming.",
        auto_install="debian:fonts-arphic-uming",
        category="cjk_serif",
    ),
    FontSpec(
        name="PingFang",
        purpose="macOS sans fallback.",
        priority=70,
        paths=(
            "/System/Library/PrivateFrameworks/FontServices.framework/Versions/A/Resources/Reserved/PingFangUI.ttc",
            "/System/Library/PrivateFrameworks/FontServices.framework/Resources/Reserved/PingFangUI.ttc",
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/LanguageSupport/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/PingFang.ttc",
        ),
        install_hint="Bundled on macOS, often as reserved system font PingFangUI.ttc.",
        category="modern_sans",
    ),
    FontSpec(
        name="MicrosoftYaHei",
        purpose="Windows sans fallback.",
        priority=80,
        paths=(
            project_font_path("msyh.ttc"),
            project_font_path("msyh.ttf"),
            project_font_path("msyhbd.ttc"),
            project_font_path("msyhl.ttc"),
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyh.ttf",
        ),
        install_hint="Place msyh.ttc under project fonts/ or use a Windows system font path.",
        category="modern_sans",
    ),
    FontSpec(
        name="STHeiti",
        purpose="Last-resort macOS sans fallback.",
        priority=90,
        paths=(
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
        ),
        install_hint="Bundled on many macOS systems.",
        category="modern_sans",
    ),
    FontSpec(
        name="ArialUnicode",
        purpose="Last-resort broad Unicode fallback.",
        priority=100,
        paths=(
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
        ),
        install_hint="Bundled on some macOS systems.",
        category="fallback_sans",
    ),
)


PREFERRED_STYLE_CATEGORIES = frozenset({"song_ming", "cjk_serif"})
FONT_FILE_SUFFIXES = frozenset({".ttf", ".ttc", ".otf", ".otc"})
DEFAULT_FONT_SCAN_DIRS: tuple[str, ...] = (
    str(PROJECT_ROOT / "fonts"),
    "~/Library/Fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
)


def font_category_penalty(category: str | None) -> float:
    penalties = {
        "song_ming": 0.0,
        "cjk_serif": 0.08,
        "modern_sans": 0.45,
        "fallback_sans": 0.60,
        "manual": 0.10,
    }
    return penalties.get(category or "manual", 0.25)


DEBIAN_FONT_PACKAGES = {
    "fonts-arphic-gbsn00lp": {
        "url": "https://deb.debian.org/debian/pool/main/f/fonts-arphic-gbsn00lp/fonts-arphic-gbsn00lp_2.11-16_all.deb",
        "font_file": "gbsn00lp.ttf",
    },
    "fonts-arphic-uming": {
        "url": "https://deb.debian.org/debian/pool/main/f/fonts-arphic-uming/fonts-arphic-uming_0.2.20080216.2-11_all.deb",
        "font_file": "uming.ttc",
    },
}


def expand_path(path: str) -> Path:
    return Path(path).expanduser()


def first_existing_path(spec: FontSpec) -> Path | None:
    for raw_path in spec.paths:
        path = expand_path(raw_path)
        if path.exists():
            return path
    return None


def pil_font_load_result(path: Path | None) -> tuple[bool | None, str | None]:
    if path is None:
        return None, None
    try:
        from PIL import ImageFont

        ImageFont.truetype(str(path), 20)
        return True, None
    except Exception as exc:
        return False, str(exc)


def glyph_signature(path: Path, char: str, *, size: int = 32) -> tuple[tuple[int, int], bytes] | None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(path), size)
    image = Image.new("L", (size * 3, size * 3), 0)
    draw = ImageDraw.Draw(image)
    draw.text((size // 2, size // 2), char, font=font, fill=255)
    bbox = image.getbbox()
    if bbox is None:
        return None
    cropped = image.crop(bbox)
    return cropped.size, cropped.tobytes()


def missing_glyph_signature(path: Path, *, size: int = 32) -> tuple[tuple[int, int], bytes] | None:
    for sentinel in ("\U0010ffff", "\ue000", "\uffff"):
        signature = glyph_signature(path, sentinel, size=size)
        if signature is not None:
            return signature
    return None


def font_missing_text_chars(path: Path, text: str, *, size: int = 32) -> list[str]:
    missing_signature = missing_glyph_signature(path, size=size)
    missing: list[str] = []
    for char in dict.fromkeys(ch for ch in text if not ch.isspace()):
        try:
            signature = glyph_signature(path, char, size=size)
        except Exception:
            missing.append(char)
            continue
        if signature is None or (missing_signature is not None and signature == missing_signature):
            missing.append(char)
    return missing


def resolve_recommended_fonts(manual_font: str | None = None) -> list[tuple[str, str]]:
    if manual_font:
        path = expand_path(manual_font)
        if not path.exists():
            raise FileNotFoundError(manual_font)
        return [(path.stem, str(path))]

    resolved: list[tuple[str, str]] = []
    for spec in sorted(RECOMMENDED_FONTS, key=lambda item: item.priority):
        path = first_existing_path(spec)
        loadable, _ = pil_font_load_result(path)
        if path and loadable:
            resolved.append((spec.name, str(path)))
    if not resolved:
        raise FileNotFoundError("No recommended CJK font found. Run check-env or pass --font-path.")
    return resolved


def font_category_for_name(name: str, path: str | None = None) -> str:
    text = f"{name} {path or ''}".lower()
    if any(token in text for token in ("song", "simsun", "simsong", "gbsn", "ming", "mincho", "uming", "stfangsong")):
        return "song_ming"
    if any(token in text for token in ("serif", "kaiti", "kai", "fangsong", "simfang", "仿宋", "fangzheng", "fz")):
        return "cjk_serif"
    if any(token in text for token in ("hei", "sans", "yahei", "pingfang", "heiti", "gothic", "arial")):
        return "modern_sans"
    return "manual"


def iter_font_files(scan_dirs: tuple[str, ...] | list[str] | None = None) -> list[Path]:
    dirs = scan_dirs or DEFAULT_FONT_SCAN_DIRS
    seen: set[Path] = set()
    files: list[Path] = []
    for raw_dir in dirs:
        root = expand_path(raw_dir)
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in FONT_FILE_SUFFIXES:
            paths = [root]
        else:
            paths = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in FONT_FILE_SUFFIXES]
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return sorted(files, key=lambda item: str(item).lower())


def resolve_scanned_fonts(
    manual_font: str | None = None,
    *,
    scan_dirs: tuple[str, ...] | list[str] | None = None,
    include_recommended: bool = True,
    max_fonts: int = 500,
) -> list[tuple[str, str]]:
    if manual_font:
        return resolve_recommended_fonts(manual_font)

    resolved: list[tuple[str, str]] = []
    seen_paths: set[Path] = set()

    if include_recommended:
        for name, raw_path in resolve_recommended_fonts(None):
            path = expand_path(raw_path).resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            resolved.append((name, str(path)))

    for path in iter_font_files(scan_dirs):
        if len(resolved) >= max_fonts:
            break
        expanded = path.expanduser()
        path_key = expanded.resolve()
        if path_key in seen_paths:
            continue
        loadable, _ = pil_font_load_result(expanded)
        if not loadable:
            continue
        seen_paths.add(path_key)
        resolved.append((expanded.stem, str(expanded)))

    if not resolved:
        raise FileNotFoundError("No loadable CJK font found from recommended fonts or scan dirs.")
    return resolved


def font_category_for_path(font_path: str, font_name: str | None = None) -> str:
    expanded = expand_path(font_path)
    try:
        expanded_resolved = expanded.resolve()
    except OSError:
        expanded_resolved = expanded
    for spec in RECOMMENDED_FONTS:
        if font_name and spec.name == font_name:
            return spec.category
        for raw_path in spec.paths:
            candidate = expand_path(raw_path)
            try:
                candidate_resolved = candidate.resolve()
            except OSError:
                candidate_resolved = candidate
            if candidate == expanded or candidate_resolved == expanded_resolved:
                return spec.category
    return font_category_for_name(font_name or expanded.stem, str(expanded))


def font_environment_report() -> dict[str, Any]:
    fonts: list[dict[str, Any]] = []
    for spec in sorted(RECOMMENDED_FONTS, key=lambda item: item.priority):
        path = first_existing_path(spec)
        loadable, load_error = pil_font_load_result(path)
        fonts.append(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "category": spec.category,
                "priority": spec.priority,
                "available": path is not None,
                "pil_loadable": loadable,
                "pil_load_error": load_error,
                "resolved_path": str(path) if path else None,
                "checked_paths": [str(expand_path(p)) for p in spec.paths],
                "install_hint": spec.install_hint,
                "auto_install": spec.auto_install,
            }
        )
    return {
        "recommended_fonts": fonts,
        "usable_font_order": [
            {
                "name": name,
                "path": path,
                "category": font_category_for_path(path, name),
            }
            for name, path in resolve_recommended_fonts()
        ]
        if any(item["available"] for item in fonts)
        else [],
    }


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def dependency_report() -> list[dict[str, Any]]:
    deps = ["PIL", "numpy", "cv2"]
    return [
        {"module": dep, "available": importlib.util.find_spec(dep) is not None}
        for dep in deps
    ]


def prompt_report() -> list[dict[str, Any]]:
    return [
        {"name": name, "path": str(prompt_resource(name)), "available": prompt_exists(name)}
        for name in PROMPT_NAMES
    ]


def api_report(env_path: Path) -> dict[str, Any]:
    env = {**load_dotenv(env_path), **os.environ}
    return {
        "env_path": str(env_path),
        "env_file_exists": env_path.exists(),
        "has_openai_api_key": bool(env.get("OPENAI_API_KEY")),
        "openai_base_url": env.get("OPENAI_BASE_URL"),
        "openai_judge_model": env.get("OPENAI_JUDGE_MODEL") or env.get("OPENAI_MODEL"),
    }


def environment_report(env_path: Path, metadata_path: Path | None = None) -> dict[str, Any]:
    return {
        "dependencies": dependency_report(),
        "fonts": font_environment_report(),
        "prompts": prompt_report(),
        "api": api_report(env_path),
        "metadata": {
            "path": str(metadata_path) if metadata_path else None,
            "available": metadata_path.exists() if metadata_path else None,
        },
    }


def install_recommended_fonts() -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    brew = shutil.which("brew")
    if brew:
        for cask in ("font-noto-sans-cjk", "font-noto-serif-cjk"):
            proc = subprocess.run(
                [brew, "install", "--cask", cask],
                text=True,
                capture_output=True,
                check=False,
            )
            actions.append(
                {
                    "installer": "brew",
                    "target": cask,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[-2000:],
                    "stderr": proc.stderr[-2000:],
                }
            )
    else:
        actions.append({"installer": "brew", "target": "noto-cjk", "error": "brew not found"})

    user_fonts = Path("~/Library/Fonts").expanduser()
    user_fonts.mkdir(parents=True, exist_ok=True)
    for package, spec in DEBIAN_FONT_PACKAGES.items():
        proc = _install_debian_font_package(package, spec["url"], spec["font_file"], user_fonts)
        actions.append(proc)
    return {"actions": actions, "fonts": font_environment_report()}


def _install_debian_font_package(package: str, url: str, font_file: str, user_fonts: Path) -> dict[str, Any]:
    if not shutil.which("curl") or not shutil.which("ar") or not shutil.which("tar"):
        return {
            "installer": "debian-package",
            "target": package,
            "error": "curl, ar, and tar are required",
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        deb_path = tmp_path / f"{package}.deb"
        curl = subprocess.run(
            ["curl", "-fsSLo", str(deb_path), url],
            text=True,
            capture_output=True,
            check=False,
        )
        if curl.returncode != 0:
            return {
                "installer": "debian-package",
                "target": package,
                "returncode": curl.returncode,
                "stderr": curl.stderr[-2000:],
            }

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        ar_proc = subprocess.run(
            ["ar", "x", str(deb_path)],
            cwd=extract_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if ar_proc.returncode != 0:
            return {
                "installer": "debian-package",
                "target": package,
                "returncode": ar_proc.returncode,
                "stderr": ar_proc.stderr[-2000:],
            }

        data_archives = list(extract_dir.glob("data.tar.*"))
        if not data_archives:
            return {"installer": "debian-package", "target": package, "error": "data.tar.* not found"}
        tar_proc = subprocess.run(
            ["tar", "-xf", str(data_archives[0])],
            cwd=extract_dir,
            text=True,
            capture_output=True,
            check=False,
        )
        if tar_proc.returncode != 0:
            return {
                "installer": "debian-package",
                "target": package,
                "returncode": tar_proc.returncode,
                "stderr": tar_proc.stderr[-2000:],
            }

        matches = list(extract_dir.rglob(font_file))
        if not matches:
            return {"installer": "debian-package", "target": package, "error": f"{font_file} not found"}
        target = user_fonts / font_file
        shutil.copy2(matches[0], target)
        return {
            "installer": "debian-package",
            "target": package,
            "installed": str(target),
            "source_url": url,
        }


def print_report(report: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if report.get("dependencies"):
        print("Dependencies:")
        for dep in report["dependencies"]:
            print(f"  [{'ok' if dep['available'] else 'missing'}] {dep['module']}")

    print("Fonts:")
    for item in report["fonts"]["recommended_fonts"]:
        if not item["available"]:
            status = "missing"
        elif item["pil_loadable"]:
            status = "ok"
        else:
            status = "invalid"
        suffix = ""
        if item.get("pil_load_error"):
            suffix = f" ({item['pil_load_error']})"
        print(f"  [{status}] {item['name']}: {item['resolved_path'] or item['install_hint']}{suffix}")

    if report.get("prompts"):
        print("Prompts:")
        for item in report["prompts"]:
            print(f"  [{'ok' if item['available'] else 'missing'}] {item['path']}")

    if report.get("api"):
        api = report["api"]
        print("API:")
        print(f"  env: {api['env_path']} ({'exists' if api['env_file_exists'] else 'missing'})")
        print(f"  base_url: {api['openai_base_url']}")
        print(f"  model: {api['openai_judge_model']}")
        print(f"  api_key: {'present' if api['has_openai_api_key'] else 'missing'}")
