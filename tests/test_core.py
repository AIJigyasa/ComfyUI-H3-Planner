"""Standalone checks for the parts that must never be wrong.

No ComfyUI, no torch. Run from the pack root:

    python tests/test_core.py
"""

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h3_planner import ladder, resolution, store  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def close(label, got, want, tol=1e-6):
    if abs(got - want) <= tol:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s: got %r want %r" % (label, got, want))


# --------------------------------------------------------------------------
print("\nladder — must reproduce the reference workflow's Math Expression")

def reference_expression(a, fps=24):
    """max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17"""
    base = max(5, round(a * fps))
    return base + (5 - (base % 17)) % 17


for seconds in (0.1, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 12.5, 14.0):
    check("snap(%.1fs) matches workflow" % seconds,
          ladder.snap_frames(seconds, 24), reference_expression(seconds))

check("6.0s -> 158 frames", ladder.snap_frames(6.0, 24), 158)
close("158 frames == 6.583s", ladder.frames_to_seconds(158, 24), 6.5833333, 1e-5)
check("every rung is legal",
      all(f % 17 == 5 for f, _ in ladder.rungs(24, 15.0)), True)
check("ceiling under 15s is 345 frames", ladder.ceiling(24, 15.0)[0], 345)
close("ceiling is 14.375s", ladder.ceiling(24, 15.0)[1], 14.375, 1e-6)
check("21 rungs under 15s", len(ladder.rungs(24, 15.0)), 21)
check("snap never returns less than asked",
      all(ladder.snap_frames(s / 10.0, 24) >= max(5, round(s / 10.0 * 24))
          for s in range(1, 150)), True)
check("nearest rounds down when down is closer",
      ladder.snap_frames(5.95, 24, direction="nearest"), 141)
check("nearest rounds up when up is closer",
      ladder.snap_frames(6.4, 24, direction="nearest"), 158)
check("up never rounds down", ladder.snap_frames(5.95, 24, direction="up"), 158)
check("modulus=1 disables the ladder",
      ladder.snap_frames(6.0, 24, modulus=1, remainder=0), 144)

print("\nresolution")
w, h = resolution.wh_for(0.3, "9:16", 32)
check("0.3 MP 9:16 aligned to 32", (w % 32, h % 32), (0, 0))
check("portrait stays portrait", w < h, True)
close("0.3 MP is about right", w * h / 1e6, 0.3, 0.05)
w2, h2 = resolution.wh_for(1.2, "9:16", 32)
check("1.2 MP is bigger", (w2 > w and h2 > h), True)

# The two passes must be the same shape. Sizing each independently drifted the
# aspect, which letterboxed the stitch and changed proportion on playback.
drifted = [a for a in resolution.ASPECTS
           if abs(resolution.pass_sizes(0.3, 1.2, a, 32)[0][0]
                  / resolution.pass_sizes(0.3, 1.2, a, 32)[0][1]
                  - resolution.pass_sizes(0.3, 1.2, a, 32)[1][0]
                  / resolution.pass_sizes(0.3, 1.2, a, 32)[1][1]) > 1e-12]
check("no aspect drifts between draft and final", drifted, [])
d, f, scale = resolution.pass_sizes(0.3, 1.2, "9:16", 32)
check("0.3 -> 1.2 MP is a clean 2x", scale, 2)
check("and the final is exactly double", f, (d[0] * 2, d[1] * 2))
check("both axes stay aligned",
      (f[0] % 32, f[1] % 32, d[0] % 32, d[1] % 32), (0, 0, 0, 0))

# --------------------------------------------------------------------------
print("\nstate machine")

tmp = tempfile.mkdtemp(prefix="h3planner_test_")
try:
    path = os.path.join(tmp, "timeline.json")
    project = {"name": "t", "fps": 24.0, "base_seed": 99, "default_duration": 6.0}
    authored = {"segments": [
        {"id": "a", "duration": 3.0, "prompt": "one"},
        {"id": "b", "duration": 6.0, "prompt": "two"},
        {"id": "c", "duration": 4.0, "prompt": "three"},
    ]}
    tl = store.normalize_timeline(authored, project)
    store.save(path, tl)

    check("three segments", len(tl["segments"]), 3)
    check("seeds are deterministic",
          tl["segments"][0]["seed"],
          store.normalize_timeline(authored, project)["segments"][0]["seed"])
    check("seeds differ per segment",
          len({s["seed"] for s in tl["segments"]}), 3)
    check("base seed changes them",
          tl["segments"][0]["seed"] ==
          store.normalize_timeline(authored, dict(project, base_seed=1))
          ["segments"][0]["seed"], False)

    seg, why = store.select(tl, "draft")
    check("claims the first segment", seg["id"], "a")
    tl, seg = store.claim(path, seg["id"], "draft")
    check("claimed segment is running", seg["state"], store.RUNNING)

    seg2, _ = store.select(tl, "draft")
    check("a running segment is skipped", seg2["id"], "b")

    store.complete(path, "a", "draft", {"file": "a.mp4"}, frames=73, duration=3.04)
    tl = store.load(path)
    check("completed state", store.find(tl, "a")["state"], "draft")
    seg3, _ = store.select(tl, "draft")
    check("next claim moves on", seg3["id"], "b")

    for sid in ("b", "c"):
        store.claim(path, sid, "draft")
        store.complete(path, sid, "draft", {"file": sid + ".mp4"})
    tl = store.load(path)
    seg4, why4 = store.select(tl, "draft")
    check("extra queued runs find nothing", seg4, None)
    check("progress is complete", store.progress(tl, "draft")[:2], (3, 3))

    seg5, _ = store.select(tl, "final")
    check("the final pass is outstanding again", seg5["id"], "a")

    # re-author: edit one prompt, everything else keeps its clips
    authored["segments"][1]["prompt"] = "two, rewritten"
    fresh = store.normalize_timeline(authored, project)
    merged = store.merge(tl, fresh)
    check("untouched segment keeps its clip",
          bool(store.find(merged, "a")["clips"].get("draft")), True)
    check("edited segment loses its clip",
          bool(store.find(merged, "b")["clips"].get("draft")), False)
    check("edited segment is pending again",
          store.find(merged, "b")["state"], store.PENDING)
    check("third segment untouched",
          bool(store.find(merged, "c")["clips"].get("draft")), True)

    # stale sweep
    store.save(path, merged)
    merged, s = store.claim(path, "b", "draft")
    s["claimed_at"] = 0  # a crashed run from 1970
    revived = store.sweep_stale(merged, 1800)
    check("stale run is reclaimed", revived, ["b"])
    check("reclaimed segment is pending", s["state"], store.PENDING)

    # forgiving authoring
    bare = store.normalize_timeline(["just a prompt string"], project)
    check("a bare list of strings works", bare["segments"][0]["id"], "seg_01")
    close("default duration applied", bare["segments"][0]["target_duration"], 6.0)

    try:
        store.normalize_timeline({"segments": []}, project)
        check("empty segments rejected", False, True)
    except ValueError:
        check("empty segments rejected", True, True)

    try:
        store.normalize_timeline({"segments": [{"id": "x"}, {"id": "x"}]}, project)
        check("duplicate ids rejected", False, True)
    except ValueError:
        check("duplicate ids rejected", True, True)

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
