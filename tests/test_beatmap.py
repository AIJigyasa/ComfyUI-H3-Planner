"""The beat map: bar lines out of audio, and a timeline cut on them.

    python tests/test_beatmap.py

The grid maths is checked on a synthetic click track with a known tempo, so it
is exact rather than approximate. The node itself is then run for real. If
librosa is missing the analysis half skips and the rest still runs.
"""

import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = tempfile.mkdtemp(prefix="h3beat_")
os.environ["USERPROFILE"] = os.environ["HOME"] = WORK

import numpy as np  # noqa: E402

from h3_planner import beatmap, ladder, nodes_plan, splitter, store  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def ok(label, condition, detail=""):
    check(label + (" (%s)" % detail if detail else ""), bool(condition), True)


def project(name):
    return nodes_plan.H3PlannerProject().build(
        name, 24.0, "9:16", "draft", 0.3, 1.2, 32, 4242, 17, 5, 5, "up",
        10.5, 8.0, 10)[0]


def click_track(bpm=120.0, bars=16, sr=22050, beats_per_bar=4):
    """A metronome with an accent on beat one, so the phase is knowable."""
    beat = 60.0 / bpm
    total = beat * beats_per_bar * bars
    n = int(total * sr)
    y = np.zeros(n, dtype=np.float32)
    for i in range(beats_per_bar * bars):
        at = int(i * beat * sr)
        length = min(int(0.04 * sr), n - at)
        if length <= 0:
            continue
        env = np.exp(-np.linspace(0, 12, length)).astype(np.float32)
        freq = 1800.0 if i % beats_per_bar == 0 else 900.0
        loud = 1.0 if i % beats_per_bar == 0 else 0.4
        tone = np.sin(2 * np.pi * freq * np.arange(length) / sr)
        y[at:at + length] += (tone * env * loud).astype(np.float32)
    return {"waveform": y[None, None, :], "sample_rate": sr}


try:
    # ----------------------------------------------------------------------
    print("\nchoosing how many bars make a clip")
    bar = 2.601                       # 92.3 BPM, from a real track
    check("5-10s takes 3 bars", beatmap.group_size(bar, 5.0, 10.0), 3)
    check("2-4s takes 1 bar", beatmap.group_size(bar, 2.0, 4.0), 1)
    check("a forced grouping is honoured",
          beatmap.group_size(bar, 5.0, 10.0, prefer=2), 2)
    check("a forced grouping outside the window is refused",
          beatmap.group_size(bar, 5.0, 10.0, prefer=8), 3)
    # 2 bars = 5.202s against a 5.2s ceiling: a millisecond is not a reason
    ok("a near miss is allowed through",
       beatmap.group_size(bar, 4.5, 5.2) == 2,
       str(beatmap.group_size(bar, 4.5, 5.2)))
    check("a window with no whole bar in it says so",
          beatmap.group_size(1.6, 5.0, 6.0), 0)

    # ----------------------------------------------------------------------
    print("\nlaying cuts on the bar grid")
    fake = {"tempo": 92.3, "bar_seconds": bar, "beats_per_bar": 4,
            "bars": [round(i * bar, 3) for i in range(41)],
            "duration": round(40 * bar, 3), "drops": []}
    cuts, missed, leftover = beatmap.plan_cuts(fake, 5.0, 10.0)
    ok("it produced cuts", len(cuts) > 3, str(len(cuts)))
    ok("every boundary is a real bar line",
       all(any(abs(c["start"] - b) < 1e-6 for b in fake["bars"]) for c in cuts))
    ok("every full clip is inside the window",
       all(5.0 - 0.05 <= c["duration"] <= 10.0 + 0.05
           for c in cuts if not c.get("partial")),
       str(sorted({round(c["duration"], 3) for c in cuts})))
    ok("no clip is ever longer than the maximum",
       max(c["duration"] for c in cuts) <= 10.05,
       "%.3f" % max(c["duration"] for c in cuts))
    ok("the clips are contiguous",
       all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(cuts, cuts[1:])))
    check("nothing was missed with no drops", missed, [])

    # The tail used to be thrown away when merging it would break the ceiling,
    # so the last seconds of a track simply never rendered.
    ok("the cuts reach the end of the track",
       abs(cuts[-1]["end"] - fake["duration"]) < 0.05,
       "%.3f of %.3f" % (cuts[-1]["end"], fake["duration"]))
    ok("a short tail is flagged as partial, not passed off as a full clip",
       cuts[-1].get("partial") == "tail" or cuts[-1]["duration"] >= 5.0,
       str(cuts[-1]))
    check("and nothing is reported left over", leftover, [])

    # ----------------------------------------------------------------------
    print("\nthe lead-in before the first downbeat")
    # The real case: a 92 BPM rap whose first downbeat librosa put at 2.554s.
    # The first 2.554s of the track was never rendered, and the opening shot
    # of the treatment vanished from the video with nothing said about it.
    late = {"tempo": 92.0, "bar_seconds": 2.608, "beats_per_bar": 4,
            "bars": [round(2.554 + i * 2.608, 3) for i in range(19)],
            "duration": 49.366, "drops": []}
    lead, _, _ = beatmap.plan_cuts(late, 5.0, 10.0)
    check("the first clip starts at the top of the track",
          lead[0]["start"], 0.0)
    ok("it took fewer bars so the lead-in fits under the ceiling",
       lead[0]["duration"] <= 10.05 and lead[0]["bars"] == 2,
       "%.3fs, %d bar(s)" % (lead[0]["duration"], lead[0]["bars"]))
    ok("and it is labelled a pickup", lead[0].get("partial") == "pickup")
    ok("the clip after it is back on a bar line",
       any(abs(lead[1]["start"] - b) < 1e-6 for b in late["bars"]),
       "%.3f" % lead[1]["start"])
    ok("the cuts cover the whole track",
       lead[0]["start"] == 0.0
       and abs(lead[-1]["end"] - late["duration"]) < 0.05,
       "%.3f to %.3f" % (lead[0]["start"], lead[-1]["end"]))
    ok("the beat label says what it is",
       beatmap.label(lead[0], 0) == "pickup + 2 bar(s)",
       beatmap.label(lead[0], 0))

    off, _, _ = beatmap.plan_cuts(late, 5.0, 10.0, cover=False)
    ok("with cover off it still starts on the first downbeat",
       abs(off[0]["start"] - 2.554) < 1e-6, "%.3f" % off[0]["start"])

    # A lead-in that no grouping can absorb gets a card of its own rather
    # than being dropped: a 4.0s pickup in front of a 3-bar clip is 11.8s.
    huge = dict(late, bars=[round(4.0 + i * 2.608, 3) for i in range(19)])
    pick, _, _ = beatmap.plan_cuts(huge, 7.0, 8.0)
    check("an unabsorbable lead-in becomes its own clip", pick[0]["start"], 0.0)
    ok("flagged as a pickup, short as it is",
       pick[0].get("partial") == "pickup" and pick[0]["bars"] == 0,
       "%.3fs" % pick[0]["duration"])

    # a drop gets its own cut when the window allows a shorter clip
    dropped = dict(fake, drops=[{"at": round(5 * bar, 3), "lift": 1.6}])
    cuts2, missed2, _ = beatmap.plan_cuts(dropped, 5.0, 10.0)
    ok("a cut starts exactly on the drop",
       any(abs(c["start"] - 5 * bar) < 1e-3 for c in cuts2),
       str([c["start"] for c in cuts2[:4]]))
    ok("and it is flagged as one",
       any(c["drop"] for c in cuts2))

    # a drop one bar in cannot be honoured without breaking the minimum
    tight = dict(fake, drops=[{"at": round(1 * bar, 3), "lift": 1.6}])
    _, missed3, _ = beatmap.plan_cuts(tight, 5.0, 10.0)
    ok("an unreachable drop is reported, not silently dropped",
       len(missed3) == 1, str(missed3))

    try:
        beatmap.plan_cuts({"tempo": 150.0, "bar_seconds": 1.6, "bars": [0, 1.6],
                           "duration": 30.0, "drops": []}, 5.0, 6.0)
        ok("an impossible window raises", False, "it returned")
    except RuntimeError as ex:
        ok("an impossible window raises", "no whole number of bars" in str(ex),
           str(ex)[:70])

    # ----------------------------------------------------------------------
    print("\nmusical time survives the frame ladder")
    segs = beatmap.as_segments(cuts)
    check("one segment per cut", len(segs), len(cuts))
    ok("target_duration is the exact musical span",
       all(abs(s["target_duration"] - c["duration"]) < 1e-9
           for s, c in zip(segs, cuts)))
    ok("audio_start is the exact musical position",
       all(abs(s["audio_start"] - c["start"]) < 1e-9
           for s, c in zip(segs, cuts)))
    clock = 0.0
    for seg in segs:
        frames = ladder.snap_frames(seg["target_duration"], 24.0)
        rendered = frames / 24.0
        ok("%s renders its musical length, give or take a frame" % seg["id"],
           rendered >= seg["target_duration"] - 1.0 / 24.0,
           "%.3f vs %.3f" % (rendered, seg["target_duration"]))
        clock += seg["target_duration"]
    ok("the trimmed timeline still matches the music",
       abs(clock - (cuts[-1]["end"] - cuts[0]["start"])) < 1e-6,
       "%.4f" % clock)

    # ----------------------------------------------------------------------
    print("\na treatment cut on supplied boundaries")
    treatment = "\n".join([
        "detailed_description:",
        "[Shot 1] cinematic 35mm, he steps onto the roof.",
        "[Shot 2] 00:06.000 he crosses to the ledge.",
        "[Shot 3] 00:12.000 a wide of the skyline.",
        "[Shot 4] 00:30.000 he turns back.",
    ])
    supplied = [{"start": 0.0, "end": 6.0, "bars": 2},
                {"start": 6.0, "end": 12.0, "bars": 2},
                {"start": 12.0, "end": 18.0, "bars": 2}]
    segs2, ctx, warns = splitter.split_on_cuts(treatment, supplied, 36.0)
    check("one segment per supplied cut", len(segs2), 3)
    check("durations come from the cuts",
          [s["target_duration"] for s in segs2], [6.0, 6.0, 6.0])
    check("audio_start comes from the cuts",
          [s["audio_start"] for s in segs2], [0.0, 6.0, 12.0])
    check("each shot lands in its own segment",
          [s["source"]["shot_numbers"] for s in segs2], [[1], [2], [3]])
    ok("the shared sections came through", "style_prefix" in ctx)

    # a shot outside every cut is written and then never rendered
    early = "\n".join([
        "detailed_description:",
        "[Shot 1] cinematic 35mm, a close-up of the chain.",
        "[Shot 2] 00:06.000 he steps onto the roof.",
    ])
    _, _, warns_lost = splitter.split_on_cuts(
        early, [{"start": 2.554, "end": 8.554, "bars": 2}], 36.0)
    ok("a shot before the first cut is reported, not silently dropped",
       any("will NOT appear" in w and "Shot 1" in w for w in warns_lost),
       str(warns_lost))

    # a cut covering no shot must not be silently empty
    sparse = [{"start": 0.0, "end": 6.0}, {"start": 18.0, "end": 24.0}]
    segs3, _, warns3 = splitter.split_on_cuts(treatment, sparse, 36.0)
    check("a segment is still produced", len(segs3), 2)
    ok("and the reuse is reported",
       any("covers no shot" in w for w in warns3), str(warns3))

    # ----------------------------------------------------------------------
    print("\nthe node, on a click track of known tempo")
    try:
        import librosa  # noqa: F401
        have = True
    except Exception as ex:
        have = False
        print("  skip  librosa not importable (%s)" % ex)

    if have:
        from h3_planner import nodes_beat
        audio = click_track(bpm=120.0, bars=16)
        node = nodes_beat.H3PlannerBeatMap()
        timeline, cuts_json, report = node.map_beats(
            project("beat_e2e"), audio, 5.0, 10.0, 4, 0, True, 0.0, 0.0)

        import json
        data = json.loads(cuts_json)
        found = data["analysis"]["tempo"]
        ok("it found the tempo", abs(found - 120.0) < 3.0, "%.1f BPM" % found)
        ok("and the bar length", abs(data["analysis"]["bar_seconds"] - 2.0) < 0.1,
           "%.3fs" % data["analysis"]["bar_seconds"])
        ok("it produced a timeline", len(timeline["segments"]) >= 2,
           str(len(timeline["segments"])))
        ok("every clip is inside the window",
           all(5.0 - 0.06 <= s["target_duration"] <= 10.0 + 0.06
               for s in timeline["segments"]),
           str([s["target_duration"] for s in timeline["segments"]]))
        ok("no segment plans longer than it renders",
           all(s["render_duration"] >= s["target_duration"] - 1e-9
               for s in timeline["segments"]),
           str([(s["target_duration"], s["render_duration"])
                for s in timeline["segments"]]))
        ok("the report accounts for frame rounding", "rounding" in report)
        ok("audio_start increases down the track",
           [s["audio_start"] for s in timeline["segments"]]
           == sorted(s["audio_start"] for s in timeline["segments"]))
        ok("no prompts were invented",
           all(not s.get("prompt") for s in timeline["segments"]))
        ok("the report names the tempo", "BPM" in report, report[:60])
        ok("the timeline was saved",
           store.load(project("beat_e2e")["timeline_path"]) is not None)

        # and the slicer accepts those cuts
        from h3_planner import nodes_slice
        sliced, srep = nodes_slice.H3PlannerSegmentSlicer().slice_treatment(
            project("beat_slice"), treatment, 36.0, 2, 2.0, "background only",
            True, cuts=cuts_json)
        check("the slicer used the beat map's cuts",
              len(sliced["segments"]), len(timeline["segments"]))
        ok("and says so in its report", "supplied by the beat map" in srep,
           srep[:120])
        ok("every segment got a prompt",
           all(isinstance(s["prompt"], dict) for s in sliced["segments"]))

        # a bad cuts string must refuse clearly
        try:
            nodes_slice.H3PlannerSegmentSlicer().slice_treatment(
                project("beat_bad"), treatment, 36.0, 2, 2.0,
                "background only", True, cuts="not json at all")
            ok("malformed cuts are refused", False, "it proceeded")
        except ValueError as ex:
            ok("malformed cuts are refused", "H3 Beat Map" in str(ex),
               str(ex)[:70])

finally:
    shutil.rmtree(WORK, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
