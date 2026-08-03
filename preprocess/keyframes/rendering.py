"""Image-only rendering helpers, deliberately independent of video decoding."""
from __future__ import annotations

from PIL import Image

from preprocess.keyframes.contracts import RenderProfile


def render_image(source: Image.Image, destination, profile: RenderProfile) -> tuple[int, int]:
    """Preserve or proportionally resize one decoded image per ``profile``."""
    image = source.convert("RGB")
    width, height = image.size
    short_edge = min(width, height)

    if profile.target_short_edge_px is not None and (
        short_edge > profile.target_short_edge_px or profile.allow_upscale
    ):
        scale = profile.target_short_edge_px / short_edge
        # LANCZOS is high quality for both downscaling and the explicitly
        # requested upscaling case.  Dimensions are clamped for malformed input.
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    if profile.image_format == "png":
        image.save(destination, format="PNG", compress_level=profile.png_compress_level)
    else:
        image.save(
            destination,
            format="JPEG",
            quality=profile.jpeg_quality,
            optimize=True,
            progressive=True,
        )
    return image.size
