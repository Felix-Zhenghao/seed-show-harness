from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def prepare_view(image: np.ndarray, rotation_degrees: int = 0) -> np.ndarray:
    """Rotate a simulator camera image counterclockwise in quarter turns."""
    view = to_uint8_hwc(image)
    return np.ascontiguousarray(np.rot90(view, k=(int(rotation_degrees) % 360) // 90))


def image_to_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(to_uint8_hwc(image)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_png(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8_hwc(image)).save(path)


class StreamingVideoWriter:
    """Append-as-you-go MP4 writer whose file survives a killed process.

    Frames stream straight to disk as a FRAGMENTED MP4 (``-movflags
    frag_keyframe+empty_moov``, intra-only zero-latency x264, one flushed fragment
    per frame): every fragment is self-contained, so a run that is force-killed
    (SIGKILL, double Ctrl+C mid-encode, power loss) still leaves a playable file
    with every logged frame -- no lost moov atom. ``close(final_path)`` finalizes the
    stream and renames it to the final name; on a crash the fragmented
    ``live_path`` simply remains, already watchable. (Deliberately NO faststart
    remux at close: ffmpeg 4.2's stream-copy from fragmented mp4 drops the final
    fragment's frame.)
    """

    def __init__(self, live_path: str | Path, fps: float) -> None:
        self.live_path = Path(live_path)
        self.fps = float(fps)
        self._writer = None
        self._count = 0

    @property
    def frame_count(self) -> int:
        return self._count

    def append(self, frame: np.ndarray) -> None:
        frame = to_uint8_hwc(frame)
        if self._writer is None:
            self.live_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio.get_writer(
                self.live_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                # Intra-only + zerolatency: no encoder lookahead/GOP buffering, so
                # EVERY appended frame is flushed to disk as its own fragment --
                # the crash-safety guarantee. (Analysis videos are a few hundred
                # frames at ~2 fps; the intra-only size cost is irrelevant next to
                # the per-step PNGs already saved.)
                output_params=[
                    "-g",
                    "1",
                    "-tune",
                    "zerolatency",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                ],
            )
        self._writer.append_data(frame)
        self._count += 1

    def close(self, final_path: Optional[str | Path] = None) -> Optional[Path]:
        """Finalize the stream and (with ``final_path``) rename the video to its
        final name. Returns the written file's path, or None when no frame was
        ever appended."""
        if self._writer is None:
            return None
        self._writer.close()
        self._writer = None
        if final_path is None:
            return self.live_path
        final = Path(final_path)
        final.parent.mkdir(parents=True, exist_ok=True)
        self.live_path.replace(final)
        return final
