"""Megapixel + aspect -> aligned width/height.

Mirrors what ResolutionSelector does in the reference workflow, so the pack can
drive MiniMaxH3ReferenceToVideo's width/height directly and the two passes
(0.3 MP draft, 1.2 MP final) differ by one number.
"""

import math
import re

ASPECTS = ["9:16", "16:9", "1:1", "4:5", "5:4", "4:3", "3:4", "21:9", "2:1", "1:2"]


def parse_aspect(text, fallback=(9, 16)):
    match = re.search(r"(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)", str(text))
    if not match:
        return fallback
    w, h = float(match.group(1)), float(match.group(2))
    if w <= 0 or h <= 0:
        return fallback
    return w, h


def align_to(value, align):
    align = max(1, int(align))
    return max(align, int(round(float(value) / align)) * align)


def wh_for(megapixels, aspect, align=32):
    """Width/height closest to ``megapixels`` at ``aspect``, both multiples of align."""
    aw, ah = parse_aspect(aspect)
    total = max(0.01, float(megapixels)) * 1e6
    ratio = aw / ah
    width = align_to(ratio * math.sqrt(total / ratio), align)
    # Derive the height FROM the aligned width rather than aligning both
    # independently. Rounding each axis on its own drifts the aspect, so a
    # 0.3 MP draft came out 416x736 while its 1.2 MP final came out 832x1504 —
    # different shapes, which meant the stitcher had to letterbox between
    # passes and clips visibly changed proportion on playback.
    height = align_to(width / ratio, align)
    return width, height


def pass_sizes(draft_megapixels, final_megapixels, aspect, align=32):
    """Draft and final dimensions, with the final an exact multiple of the draft.

    Sizing each pass independently cannot hold the aspect: both axes get
    rounded to `align` separately and the two passes end up different shapes
    (4:5 at 0.3 MP lands on 480x608, at 1.2 MP on 992x1248). Different shapes
    mean the stitcher letterboxes between passes and clips change proportion
    on playback.

    Scaling the draft by a whole number makes the aspect identical by
    construction — and lands the final on a clean 2x for the latent upscaler,
    which is what the two-pass H3 graph wants anyway.
    """
    draft = wh_for(draft_megapixels, aspect, align)
    ratio = max(1.0, float(final_megapixels) / max(1e-6, float(draft_megapixels)))
    scale = max(1, int(round(math.sqrt(ratio))))
    final = (draft[0] * scale, draft[1] * scale)
    return draft, final, scale


def describe(megapixels, aspect, align=32):
    w, h = wh_for(megapixels, aspect, align)
    return "%dx%d (%.2f MP, %s, align %d)" % (w, h, w * h / 1e6, aspect, align)
