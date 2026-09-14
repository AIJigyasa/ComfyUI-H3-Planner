"""H3 frame-length ladder.

MiniMax H3 does not accept an arbitrary frame count. The reference workflow's
Math Expression

    max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17

reduces to: length must satisfy ``length % 17 == 5``, minimum 5, at 24 fps.
That is a ladder of rungs 17 frames (0.7083 s) apart.

Everything here is pure arithmetic with no ComfyUI imports, so it can be tested
standalone. The modulus/remainder are parameters rather than constants because
they may shift with a different VAE or turbo LoRA.
"""

MODULUS = 17
REMAINDER = 5
MINIMUM = 5


def snap_frames(seconds, fps, modulus=MODULUS, remainder=REMAINDER,
                minimum=MINIMUM, direction="up"):
    """Return a valid H3 frame count for ``seconds`` at ``fps``.

    ``direction`` is "up" (never render less than asked — the reference
    workflow's behaviour, and what trim-on-stitch relies on) or "nearest".
    """
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    modulus = max(1, int(modulus))
    remainder = int(remainder) % modulus
    minimum = max(1, int(minimum))

    raw = int(round(float(seconds) * fps))
    up = raw + ((remainder - raw) % modulus)
    while up < minimum:
        up += modulus

    if direction != "nearest":
        return up

    down = up - modulus
    if down < minimum:
        return up
    return down if (raw - down) < (up - raw) else up


def frames_to_seconds(frames, fps):
    return float(frames) / float(fps)


def rungs(fps, max_seconds=15.0, modulus=MODULUS, remainder=REMAINDER,
          minimum=MINIMUM):
    """Every valid (frames, seconds) pair up to ``max_seconds``."""
    modulus = max(1, int(modulus))
    remainder = int(remainder) % modulus
    out = []
    frames = remainder if remainder >= minimum else remainder + modulus
    while frames < minimum:
        frames += modulus
    while frames / float(fps) <= max_seconds:
        out.append((frames, frames / float(fps)))
        frames += modulus
    return out


def ceiling(fps, max_seconds=15.0, **kw):
    """Highest rung that still fits inside ``max_seconds``.

    At 24 fps with the default ladder this is 345 frames = 14.375 s — 15.0 s
    itself is not reachable.
    """
    grid = rungs(fps, max_seconds, **kw)
    return grid[-1] if grid else (0, 0.0)


def describe(fps, max_seconds=15.0, **kw):
    grid = rungs(fps, max_seconds, **kw)
    if not grid:
        return "no valid lengths"
    body = ", ".join("%d(%.3fs)" % (f, s) for f, s in grid)
    return "%d rungs @ %g fps: %s" % (len(grid), fps, body)
