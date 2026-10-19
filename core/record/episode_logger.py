from __future__ import annotations

import json
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .images import StreamingVideoWriter, save_png, to_uint8_hwc


JSONL_REASON_LIMIT = 1500
_SENSITIVE_METADATA_KEYS = {
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
    "password",
    "secret",
    "client_secret",
}


def _jsonable(value: Any) -> Any:
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__name__)
        return f"<callable {module}.{name}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _redact_sensitive_metadata(value: Any) -> Any:
    """Return metadata with credential values removed recursively.

    Environment-variable *names* such as ``api_key_env`` remain useful for
    reproduction and are not credentials themselves.
    """
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SENSITIVE_METADATA_KEYS:
                redacted[name] = "[REDACTED]"
            else:
                redacted[name] = _redact_sensitive_metadata(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive_metadata(item) for item in value]
    return value


def _truncate_record_reasoning(record: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the step record with each VLM reasoning string capped for the
    compact steps.jsonl. The full text is preserved separately in steps.json."""
    vlm = record.get("vlm")
    if not isinstance(vlm, dict):
        return record
    truncated_vlm: dict[str, Any] = {}
    for role, entry in vlm.items():
        if isinstance(entry, dict):
            entry = {
                key: (
                    value[:JSONL_REASON_LIMIT] + "..."
                    if key in ("why", "reasoning", "step_size_reasoning", "raw")
                    and isinstance(value, str)
                    and len(value) > JSONL_REASON_LIMIT
                    else value
                )
                for key, value in entry.items()
            }
        truncated_vlm[role] = entry
    out = dict(record)
    out["vlm"] = truncated_vlm
    return out


class EpisodeLogger:
    def __init__(
        self,
        root_dir: str | Path,
        task_id: int,
        video_fps: float = 2.0,
    ) -> None:
        now = datetime.now(timezone(timedelta(hours=8)))
        date_stamp = now.strftime("%m%d")
        time_stamp = now.strftime("%H-%M-%S")
        base_dir = Path(root_dir)
        self.run_dir = base_dir / date_stamp / f"task_{task_id}" / time_stamp
        self.agentview_dir = self.run_dir / "images" / "agentview"
        self.wrist_dir = self.run_dir / "images" / "wrist"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._steps_path = self.run_dir / "steps.jsonl"
        self._steps_file = self._steps_path.open("w", encoding="utf-8")
        self._steps_json_path = self.run_dir / "steps.json"
        # Full per-step records with the complete (un-truncated) reasoning, dumped to
        # steps.json at close; steps.jsonl keeps the compact, reasoning-truncated form.
        self._full_records: list[dict[str, Any]] = []
        # Analysis video: frames STREAM to disk as they are logged (crash-safe
        # fragmented mp4 -- a force-killed run leaves rollout_live.mp4 playable);
        # close() renames it to rollout_success/failure.mp4.
        self.video = StreamingVideoWriter(
            self.run_dir / "rollout_live.mp4",
            fps=min(30.0, max(0.5, float(video_fps))),
        )

    def write_metadata(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(_redact_sensitive_metadata(data)), f, indent=2, sort_keys=True)

    def write_summary(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def write_plan(self, data: dict[str, Any]) -> None:
        with (self.run_dir / "subgoals.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def write_planner_diagnostics(self, data: dict[str, Any]) -> None:
        """Persist planner routes/errors without mixing them into the plan contract."""
        with (self.run_dir / "planner_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(data), f, indent=2, sort_keys=True)

    def save_planner_prompt(self, attempt: int, prompt: str) -> None:
        prompts_dir = self.run_dir / "planner_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"attempt_{int(attempt):02d}.txt").write_text(str(prompt), encoding="utf-8")

    def log_step(
        self,
        step_idx: int,
        agentview: np.ndarray,
        wrist: Optional[np.ndarray],
        record: dict[str, Any],
    ) -> None:
        """Save the scene, optional wrist, and one semantic decision."""
        agentview_path = self.agentview_dir / f"{step_idx:04d}.png"
        save_png(agentview_path, agentview)
        record = dict(record)
        if wrist is not None:
            wrist_path = self.wrist_dir / f"{step_idx:04d}.png"
            save_png(wrist_path, wrist)
        # steps.json keeps the full reasoning; steps.jsonl gets a reasoning-truncated copy.
        self._full_records.append(_jsonable(record))
        self._steps_file.write(
            json.dumps(_jsonable(_truncate_record_reasoning(record)), separators=(",", ":")) + "\n"
        )
        self._steps_file.flush()
        self.video.append(_make_analysis_frame(agentview=agentview, wrist=wrist, record=record))

    def save_controller_prompt(self, step_idx: int, prompt: str) -> None:
        """Save the rendered prompt; DebugTrace records the exact image inputs."""
        prompts_dir = self.run_dir / "controller_prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        (prompts_dir / f"{step_idx:04d}.txt").write_text(str(prompt), encoding="utf-8")

    def close(self, success: bool, fps: float) -> Path:
        self._steps_file.close()
        with self._steps_json_path.open("w", encoding="utf-8") as f:
            json.dump(self._full_records, f, indent=2, ensure_ascii=False)
        video_path = self.run_dir / ("rollout_success.mp4" if success else "rollout_failure.mp4")
        # Frames were already streamed at the writer's fps (fixed at open time);
        # a differing late fps request cannot re-time them, only note it.
        if self.video.frame_count and abs(float(fps) - self.video.fps) > 1e-6:
            print(
                f"[logger] note: video was streamed at {self.video.fps:g} fps "
                f"(close requested {float(fps):g})."
            )
        self.video.close(final_path=video_path)
        return video_path


# Analysis video: header, reasoning, scene and wrist panels, and stage status.
# The canvas is divisible by 16 so the video encoder does not resize it.
_PANEL = 512
_HEADER_H = 64
_REASON_H = 96
_LABEL_H = 32
_BAND_H = 48

# Clean high-contrast light theme: near-white ground, crisp near-black text, and
# saturated (but not neon) accents so states pop at a glance.
BG_HEADER = (251, 251, 249)  # near-white
BG_REASON = (245, 245, 242)
BG_LABEL = (236, 237, 233)
BG_PANEL = (223, 225, 221)  # letterbox / gutters around the camera frames
RULE_C = (205, 208, 203)  # thin separators
FG = (30, 34, 39)  # crisp near-black (primary text)
MUT = (94, 101, 110)
FAINT = (150, 156, 163)
ACC_NEUTRAL = (100, 116, 139)  # slate   -- shared front view
_OK_C = (22, 163, 74)  # green
_WARN_C = (217, 119, 6)  # amber


def _make_analysis_frame(
    agentview: np.ndarray, wrist: Optional[np.ndarray], record: dict[str, Any]
) -> np.ndarray:
    """Compose one polished, information-dense frame of the saved rollout video."""
    views = [agentview, wrist if wrist is not None else np.zeros_like(to_uint8_hwc(agentview))]
    labels = ["AGENT VIEW", "WRIST"]
    accents = [ACC_NEUTRAL, ACC_NEUTRAL]

    width = _PANEL * len(views)
    panels_y = _HEADER_H + _REASON_H + _LABEL_H
    band_y = panels_y + _PANEL
    canvas = Image.new("RGB", (width, band_y + _BAND_H), color=BG_PANEL)
    draw = ImageDraw.Draw(canvas)
    big = _font(26, bold=True)
    std = _font(18)
    std_b = _font(18, bold=True)
    small = _font(15)

    # Camera panels: letterboxed (aspect preserved), never squeezed.
    for idx, view in enumerate(views):
        canvas.paste(_letterbox(view, _PANEL), (_PANEL * idx, panels_y))

    # Header: step number + right-aligned telemetry.
    draw.rectangle((0, 0, width, _HEADER_H), fill=BG_HEADER)
    draw.text((16, 16), f"STEP {int(record.get('i', 0)):03d}", fill=FG, font=big)
    draw.text((220, 24), str(record.get("action_source", "controller")), fill=MUT, font=small)
    draw.line((0, _HEADER_H - 1, width, _HEADER_H - 1), fill=RULE_C, width=1)

    # Reasoning strip: the model's words, up to four wrapped lines (~full for the
    # concise CoT this stack runs; steps.json always keeps the untruncated text).
    draw.rectangle((0, _HEADER_H, width, _HEADER_H + _REASON_H), fill=BG_REASON)
    reason = _vlm_reason(record)
    if reason != "-":
        draw.text((16, _HEADER_H + 6), "WHY", fill=FAINT, font=small)
        per_line = max(60, (width - 80) // 9)
        text = " ".join(str(reason).split())
        wrapped = textwrap.wrap(text, width=per_line)[:4]
        if len(wrapped) == 4 and len(text) > per_line * 4:
            wrapped[3] = wrapped[3][: per_line - 3] + "..."
        for i, line in enumerate(wrapped):
            draw.text((62, _HEADER_H + 6 + i * 21), line, fill=MUT, font=std)

    # Label bar: panel names with a per-panel accent underline.
    label_y = _HEADER_H + _REASON_H
    draw.rectangle((0, label_y, width, label_y + _LABEL_H), fill=BG_LABEL)
    for idx, (label, accent) in enumerate(zip(labels, accents)):
        x0 = _PANEL * idx
        draw.text((x0 + 16, label_y + 7), label, fill=MUT, font=small)
        draw.rectangle(
            (x0 + 12, label_y + _LABEL_H - 4, x0 + _PANEL - 12, label_y + _LABEL_H - 2),
            fill=accent,
        )

    # Stage and action under the wrist panel.
    draw.rectangle((0, band_y, width, band_y + _BAND_H), fill=BG_HEADER)
    draw.line((0, band_y, width, band_y), fill=RULE_C, width=1)
    x0 = _PANEL + 16
    _draw_segments(draw, x0, band_y + 4, [(_stage_label(record), FG, False)], std, std_b)
    width_m = record.get("after", {}).get("gripper_width_m")
    grip = f"grip {width_m:.3f} m" if width_m is not None else ""
    line2 = [(str(record.get("action", "-")) + "   ", FG, True), (grip, MUT, False)]
    for text, color in status_flags(record):
        line2.append((f"   {text}", color, False))
    _draw_segments(draw, x0, band_y + 25, line2, std, std_b)

    # Thin separators between panels, spanning label bar through status band.
    for idx in range(1, len(views)):
        x0 = _PANEL * idx
        draw.line((x0, label_y, x0, band_y + _BAND_H), fill=RULE_C, width=1)
    return np.asarray(canvas)


def _letterbox(view: np.ndarray, size: int) -> Image.Image:
    """Fit a camera frame into a size x size tile, preserving its aspect ratio."""
    img = Image.fromarray(to_uint8_hwc(view))
    scale = min(size / img.width, size / img.height)
    new = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new, Image.Resampling.BILINEAR)
    tile = Image.new("RGB", (size, size), BG_PANEL)
    tile.paste(img, ((size - new[0]) // 2, (size - new[1]) // 2))
    return tile


def _stage_label(record: dict[str, Any]) -> str:
    stage = record.get("stage", {})
    return f"STAGE {int(record.get('stage_index', 0)) + 1} · {stage.get('motion', '-')}"


def status_flags(record: dict[str, Any]) -> list[tuple[str, tuple[int, int, int]]]:
    flags: list[tuple[str, tuple[int, int, int]]] = []
    if record.get("success"):
        flags.append(("success", _OK_C))
    if record.get("recovery"):
        flags.append(("recovery", _WARN_C))
    if record.get("auto_release"):
        flags.append(("auto-release", _WARN_C))
    return flags


def _draw_segments(draw: Any, x: int, y: int, segments: Any, font: Any, bold: Any) -> None:
    for text, color, is_bold in segments:
        active = bold if is_bold else font
        draw.text((x, y), text, fill=color, font=active)
        x += draw.textlength(text, font=active)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    )
    for base in (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation2",
        "/usr/share/fonts/truetype/liberation",
    ):
        for name in names:
            try:
                return ImageFont.truetype(f"{base}/{name}", size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _vlm_reason(record: dict[str, Any]) -> str:
    entry = record.get("vlm", {}).get("c", {})
    return str(entry.get("why") or "-")
