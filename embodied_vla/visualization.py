from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from embodied_vla.experts import ExpertPhase


def make_vla_attention_panel(
    rgb: NDArray[np.uint8],
    heatmaps: NDArray[np.floating],
    *,
    instruction: str,
    predicted_phase: int,
    step: int,
    predicted_coordinates: NDArray[np.floating],
    ground_truth_coordinates: NDArray[np.floating],
    pixel_valid: NDArray[np.bool_],
    scale: int = 4,
) -> NDArray[np.uint8]:
    """Compose scene, target attention, and goal attention into one audit frame."""

    if heatmaps.shape[0] != 2:
        raise ValueError(f"expected two grounding heatmaps, got {heatmaps.shape}")
    source = np.asarray(rgb, dtype=np.uint8)
    height, width = source.shape[:2]
    overlays = [
        _attention_overlay(source, heatmaps[0], color=(230, 55, 55)),
        _attention_overlay(source, heatmaps[1], color=(35, 190, 210)),
    ]
    panels = [source, *overlays]
    labels = ("Scene", "Target attention", "Goal attention")
    display_width = width * scale
    display_height = height * scale
    header_height = 48
    canvas = Image.new(
        "RGB",
        (display_width * 3, display_height + header_height),
        color=(22, 24, 28),
    )
    draw = ImageDraw.Draw(canvas)
    phase_name = ExpertPhase(predicted_phase).name.lower()
    draw.text(
        (5, 3),
        f"step={step:03d}  predicted phase={phase_name}",
        fill=(245, 245, 245),
    )
    draw.text((5, 17), f"task: {instruction}", fill=(220, 220, 220))
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        panel_image = Image.fromarray(panel, mode="RGB").resize(
            (display_width, display_height),
            resample=Image.Resampling.NEAREST,
        )
        x_offset = index * display_width
        canvas.paste(panel_image, (x_offset, header_height))
        draw.text((x_offset + 5, 33), label, fill=(235, 235, 235))
        if index > 0:
            coordinate_index = index - 1
            _draw_coordinate_marker(
                draw,
                predicted_coordinates[coordinate_index],
                x_offset=x_offset,
                y_offset=header_height,
                width=display_width,
                height=display_height,
                color=(255, 255, 255),
                cross=False,
            )
            if pixel_valid[coordinate_index]:
                _draw_coordinate_marker(
                    draw,
                    ground_truth_coordinates[coordinate_index],
                    x_offset=x_offset,
                    y_offset=header_height,
                    width=display_width,
                    height=display_height,
                    color=(80, 240, 100),
                    cross=True,
                )
    return np.asarray(canvas, dtype=np.uint8)


def _attention_overlay(
    rgb: NDArray[np.uint8],
    heatmap: NDArray[np.floating],
    *,
    color: tuple[int, int, int],
) -> NDArray[np.uint8]:
    height, width = rgb.shape[:2]
    probabilities = np.asarray(heatmap, dtype=np.float32)
    intensity = probabilities / max(float(probabilities.max()), 1e-8)
    intensity_image = Image.fromarray(np.uint8(np.clip(intensity, 0.0, 1.0) * 255))
    intensity = np.asarray(
        intensity_image.resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    alpha = (intensity / 255.0 * 0.62)[..., None]
    color_array = np.asarray(color, dtype=np.float32)
    blended = rgb.astype(np.float32) * (1.0 - alpha) + color_array * alpha
    return np.uint8(np.clip(blended, 0.0, 255.0))


def _draw_coordinate_marker(
    draw: ImageDraw.ImageDraw,
    coordinate: NDArray[np.floating],
    *,
    x_offset: int,
    y_offset: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
    cross: bool,
) -> None:
    x = x_offset + int(np.clip(float(coordinate[0]), 0.0, 1.0) * (width - 1))
    y = y_offset + int(np.clip(float(coordinate[1]), 0.0, 1.0) * (height - 1))
    radius = 6
    if cross:
        draw.line((x - radius, y, x + radius, y), fill=color, width=1)
        draw.line((x, y - radius, x, y + radius), fill=color, width=1)
    else:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=1)
