"""Checks for the treatment splitter and the cast tagging rules.

No ComfyUI, no torch, no model calls — image loading is stubbed so the tag
arithmetic can be tested on its own.

    python tests/test_prompt.py
"""

import json
import re
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h3_planner import nodes_prompt, splitter  # noqa: E402
from h3_planner.nodes_prompt import _even_segments  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


TREATMENT = """subject_definitions:
<Subject 1> is the young rapper, black hooded jacket, thin gold chain.

summary:
[reference generation] A 60 second street performance.

retention_analysis:
<Subject 1>: attribute_transfer, [Shot 1], [Shot 3]

detailed_description:
[Shot 1] 2D illustration, cinematic, a medium-wide shot frames <Subject 1>.
[Shot 2] At 00:05.000 the camera cuts to a low angle as he starts the verse.
[Shot 3] At 00:11.500 the shot transitions to a tracking shot down the lane.
[Shot 4] At 00:24.000 the camera cuts to a close-up of his hands.
[Shot 5] At 00:31.000 the shot changes to a rooftop wide.
[Shot 6] At 00:44.000 the camera cuts to the crowd.
[Shot 7] At 00:52.000 the shot cuts to the final wide.

overall_soundscape:
Street ambience, autorickshaws, footsteps.

non_diegetic_music:
N/A
"""

print("\nsections")
sections = splitter.parse_sections(TREATMENT)
check("all six sections found", len(sections), 6)
check("subject_definitions kept", sections["subject_definitions"].startswith("<Subject 1>"), True)
check("music section kept", sections["non_diegetic_music"], "N/A")
check("headerless text still splits",
      splitter.parse_sections("[Shot 1] a plain prompt")["detailed_description"],
      "[Shot 1] a plain prompt")

print("\nshots")
shots = splitter.parse_shots(sections["detailed_description"], 60.0)
check("seven shots", len(shots), 7)
check("shot 1 starts at zero", shots[0]["start"], 0.0)
check("shot 2 start parsed", shots[1]["start"], 5.0)
check("shot 3 start parsed", shots[2]["start"], 11.5)
check("last shot ends at the duration", shots[-1]["end"], 60.0)
check("spans are contiguous",
      all(abs(shots[i]["end"] - shots[i + 1]["start"]) < 1e-9
          for i in range(len(shots) - 1)), True)

print("\nstyle prefix")
check("prefix lifted off shot 1",
      splitter.style_prefix(sections["detailed_description"]),
      "2D illustration, cinematic")
check("no prefix when shot 1 has none",
      splitter.style_prefix("[Shot 1] a woman walks into the room."), "")

print("\ngrouping")
segs, ctx, warns = splitter.split_treatment(TREATMENT, 60.0, 14.375, 2.0, 4)
check("no warnings on a clean treatment", warns, [])
check("every segment fits the ceiling",
      all(s["target_duration"] <= 14.375 + 1e-9 for s in segs), True)
check("total duration is preserved",
      round(sum(s["target_duration"] for s in segs), 3), 60.0)
check("no shot appears twice",
      sorted(n for s in segs for n in s["source"]["shot_numbers"]),
      [1, 2, 3, 4, 5, 6, 7])
check("no shot is lost",
      len([n for s in segs for n in s["source"]["shot_numbers"]]), 7)
check("every segment's first shot is rebased to zero",
      all(s["source"]["shots"][0]["at"] == 0.0 for s in segs), True)
check("later shots keep their offset",
      segs[0]["source"]["shots"][1]["at"], 5.0)
check("segments are contiguous",
      all(abs(segs[i]["source"]["end"] - segs[i + 1]["source"]["start"]) < 1e-9
          for i in range(len(segs) - 1)), True)
check("context carries the prefix", ctx["style_prefix"], "2D illustration, cinematic")

# a shot longer than the ceiling cannot be split — it must survive whole
LONG = ("detailed_description:\n"
        "[Shot 1] cinematic, a slow push across the room.\n"
        "[Shot 2] At 00:20.000 the camera cuts to the window.\n")
segs2, _, warns2 = splitter.split_treatment(LONG, 40.0, 14.375, 2.0, 4)
check("an over-long shot becomes its own segment", len(segs2), 2)
check("it is not split", segs2[0]["source"]["shot_numbers"], [1])
check("and it is flagged", any("past the" in w for w in warns2), True)

check("max_shots_per_segment is respected",
      all(len(s["source"]["shot_numbers"]) <= 2
          for s in splitter.split_treatment(TREATMENT, 60.0, 14.375, 2.0, 2)[0]),
      True)

print("\ncast tagging")

tmp = tempfile.mkdtemp(prefix="h3planner_cast_")
try:
    from h3_planner import nodes_cast, paths

    for name in ("hero.png", "bottle.png", "frame.png", "street.png",
                 "track.mp3", "plate.mp4"):
        open(os.path.join(tmp, name), "wb").close()

    paths.cast_dir = lambda create=True: tmp          # stub the upload dir
    nodes_cast.load_image = lambda path: ("IMAGE", os.path.basename(path))
    # Adversarial in the one way that matters here: the stubs record the trim
    # they were given, so a card whose in/out points never reach ffmpeg fails
    # instead of quietly handing the sampler the whole file.
    CUTS = {}
    nodes_cast.media.load_audio_file = lambda path, **kw: (
        CUTS.__setitem__(os.path.basename(path),
                         (kw.get("start"), kw.get("end")))
        or ("AUDIO", path))
    nodes_cast.media.decode_video = lambda path, **kw: (
        CUTS.__setitem__(os.path.basename(path),
                         (kw.get("start"), kw.get("end")))
        or ("FRAMES", path))

    NI, NA, NV = (nodes_cast.MAX_IMAGE_SLOTS, nodes_cast.MAX_AUDIO_SLOTS,
                  nodes_cast.MAX_VIDEO_SLOTS)

    def build(entries, **wired):
        out = nodes_cast.H3PlannerCastBoard().build(
            json.dumps({"entries": entries}), **wired)
        check("one value per declared output slot", len(out),
              len(nodes_cast.OUTPUT_NAMES))
        return (out[0], out[1:1 + NI], out[1 + NI:1 + NI + NA],
                out[1 + NI + NA:1 + NI + NA + NV], out[-1])

    cast, slots, audios, videos, report = build([
        {"key": "hero", "role": "character", "file": "hero.png"},
        {"key": "bottle", "role": "product", "file": "bottle.png"},
        {"key": "open", "role": "first_frame", "file": "frame.png"},
        {"key": "song", "role": "audio", "file": "track.mp3"},
        {"key": "plate", "role": "video", "file": "plate.mp4"},
    ])
    audio = audios[0]
    got = [m["tag"] for m in cast["members"]]
    check("Subject and Picture are numbered independently",
          got, ["<Subject 1>", "<Subject 2>", "<Picture 1>",
                "<Audio 1>", "<Video 1>"])
    check("only images take slots", cast["size"], 3)
    check("image slots are filled from 1",
          [s is not None for s in slots],
          [True, True, True] + [False] * (NI - 3))
    check("audio is decoded out", audio is not None, True)
    check("video path is reported", "plate.mp4" in report, True)
    check("and its frames reach the video slot", videos[0], ("FRAMES",
          os.path.join(tmp, "plate.mp4")))
    check("a product image is a Subject, not a Picture",
          cast["members"][1]["tag"], "<Subject 2>")
    check("counts reported", cast["counts"],
          {"Subject": 2, "Picture": 1, "Video": 1, "Audio": 1})

    # reordering renumbers within each kind
    cast2 = build([
        {"key": "bottle", "role": "product", "file": "bottle.png"},
        {"key": "hero", "role": "character", "file": "hero.png"},
    ])[0]
    check("reordering renumbers the tags",
          [(m["key"], m["tag"]) for m in cast2["members"]],
          [("bottle", "<Subject 1>"), ("hero", "<Subject 2>")])

    # a disabled entry is skipped without shifting the ones before it
    cast3 = build([
        {"key": "hero", "role": "character", "file": "hero.png"},
        {"key": "bottle", "role": "product", "file": "bottle.png",
         "disabled": True},
        {"key": "street", "role": "environment", "file": "street.png"},
    ])[0]
    check("disabled entries are skipped",
          [(m["key"], m["tag"]) for m in cast3["members"]],
          [("hero", "<Subject 1>"), ("street", "<Subject 2>")])

    # a missing file is reported, not crashed on
    cast4 = build([
        {"key": "ghost", "role": "character", "file": "nope.png"},
        {"key": "hero", "role": "character", "file": "hero.png"},
    ])
    report4 = cast4[-1]
    cast4 = cast4[0]
    check("a missing file is reported", "missing" in report4, True)
    check("and the rest still number from 1",
          [m["tag"] for m in cast4["members"]], ["<Subject 1>"])

    # role/asset mismatch
    cast5 = build([
        {"key": "song", "role": "character", "file": "track.mp3"},
    ])
    report5 = cast5[-1]
    cast5 = cast5[0]
    check("an audio file in an image role is refused",
          cast5["members"], [])
    check("and says why", "needs an image" in report5, True)

    # Wired references land after the uploads and keep counting on from them,
    # so a prompt that cites <Subject 3> means the third card OR the first
    # wired socket, whichever the board shows in that position.
    cast6, slots6, audios6, videos6, _ = build(
        [{"key": "hero", "role": "character", "file": "hero.png"}],
        ref_images={"ref_image_1": "B", "ref_image_0": "A"},
        ref_audios={"ref_audio_0": "TRACK"},
        ref_videos={"ref_video_0": "CLIP"})
    check("wired images are numbered on from the uploads",
          [(m["key"], m["tag"]) for m in cast6["members"]],
          [("hero", "<Subject 1>"), ("wired_image_1", "<Subject 2>"),
           ("wired_image_2", "<Subject 3>"), ("wired_audio_1", "<Audio 1>"),
           ("wired_video_1", "<Video 1>")])
    check("autogrow sockets are read in slot order, not dict order",
          list(slots6[:3]), [("IMAGE", "hero.png"), "A", "B"])
    check("wired audio reaches its own slot", audios6[0], "TRACK")
    check("wired video reaches its own slot", videos6[0], "CLIP")

    # More audio than there are slots is tagged and named, never dropped in
    # silence: the tag is what a written prompt already cites.
    for name in ("a1.mp3", "a2.mp3", "a3.mp3", "a4.mp3", "a5.mp3"):
        open(os.path.join(tmp, name), "wb").close()
    cast7, _, audios7, _, report7 = build(
        [{"key": "t%d" % i, "role": "audio", "file": "a%d.mp3" % i}
         for i in range(1, 6)])
    check("only as many audio cards as there are slots join the cast",
          len(cast7["members"]), NA)
    check("each of those is wired",
          len([a for a in audios7 if a is not None]), NA)
    # ---- trimming a reference ------------------------------------------
    check("no trim reads as the whole file", nodes_cast.trim_of({}), (0.0, None))
    check("a blank field is not a trim",
          nodes_cast.trim_of({"start": "", "end": None}), (0.0, None))
    check("zero is not a trim", nodes_cast.trim_of({"start": 0}), (0.0, None))
    check("seconds come through",
          nodes_cast.trim_of({"start": "2.5", "end": 9}), (2.5, 9.0))
    for bad, why in (({"start": 5, "end": 5}, "backwards"),
                     ({"start": "soon"}, "not a number")):
        try:
            nodes_cast.trim_of(bad)
            check("a %s trim is refused" % why, "accepted", "refused")
        except ValueError:
            check("a %s trim is refused" % why, "refused", "refused")

    CUTS.clear()
    cast8, _, audios8, videos8, report8 = build([
        {"key": "song", "role": "audio", "file": "track.mp3",
         "start": 4, "end": 12.5},
        {"key": "plate", "role": "video", "file": "plate.mp4", "start": 1.25},
    ])
    check("the audio trim reaches the decoder", CUTS["track.mp3"], (4.0, 12.5))
    check("the video trim reaches the decoder", CUTS["plate.mp4"], (1.25, None))
    check("both are still wired out",
          audios8[0] is not None and videos8[0] is not None, True)
    check("and the trim is on the member",
          cast8["members"][0]["trim"], [4.0, 12.5])
    check("the report says what was used",
          "trimmed 4.00s to 12.50s" in report8
          and "1.25s to the end" in report8, True)

    # A trim that cannot be honoured must not silently become the whole file.
    CUTS.clear()
    _, _, _, _, report9 = build([
        {"key": "song", "role": "audio", "file": "track.mp3",
         "start": 9, "end": 3}])
    check("an impossible trim is reported", "trim ignored" in report9, True)
    check("and the file is used whole", CUTS["track.mp3"], (0.0, None))

    check("and the overflow is named, not silently dropped",
          "past the %d audio" % NA in report7 and "t5" in report7, True)
    check("no tag is handed out with nothing behind it",
          [m["tag"] for m in cast7["members"]],
          ["<Audio %d>" % i for i in range(1, NA + 1)])

finally:
    shutil.rmtree(tmp, ignore_errors=True)

# --------------------------------------------------------------------------
print("\nshots per segment")

segs1, _, _ = splitter.split_treatment(TREATMENT, 60.0, 14.375, 2.0, 1)
check("one shot per segment gives one segment per shot", len(segs1), 7)
check("each carries exactly one shot",
      all(len(s["source"]["shot_numbers"]) == 1 for s in segs1), True)
check("a short tail is not merged away when the limit is 1",
      segs1[-1]["source"]["shot_numbers"], [7])
check("packing still works when allowed",
      len(splitter.split_treatment(TREATMENT, 60.0, 14.375, 2.0, 4)[0]) < 7, True)

print("\naudio windows")
check("windows follow the video clock",
      [s["audio_start"] for s in segs1][:4], [0.0, 5.0, 11.5, 24.0])
check("windows are contiguous — no gap, no overlap",
      all(abs(segs1[i]["audio_start"] + segs1[i]["target_duration"]
              - segs1[i + 1]["audio_start"]) < 1e-6
          for i in range(len(segs1) - 1)), True)
check("the last window ends at the track's end",
      round(segs1[-1]["audio_start"] + segs1[-1]["target_duration"], 3), 60.0)

print("\neven beats (no treatment)")
even = _even_segments(60.0, 10.0, 14.375)
check("six beats from 60s at 10s each", len(even), 6)
check("durations are even", {round(s["target_duration"], 3) for s in even}, {10.0})
check("audio windows still line up",
      [s["audio_start"] for s in even], [0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
check("a request over the ceiling is capped",
      all(s["target_duration"] <= 14.375
          for s in _even_segments(60.0, 30.0, 14.375)), True)
check("total is preserved",
      round(sum(s["target_duration"] for s in _even_segments(47.0, 10.0, 14.375)), 3),
      47.0)

print("\nrepeat guard")
from h3_planner.nodes_prompt import H3PlannerSegmentPrompter as _P  # noqa: E402
_a = "[Shot 1] wide on the rapper at the storefront, the camera pushes in."
check("an identical description is caught", _P._closest(_a, [_a])[0], 0)
check("a near-identical one is caught",
      _P._closest(_a.replace("pushes", "pushes slowly"), [_a])[0], 0)
check("a genuinely different one passes",
      _P._closest("[Shot 1] low angle on the crowd, handheld, cutting fast.",
                  [_a])[0], None)
check("nothing to compare against passes", _P._closest(_a, [])[0], None)

print("\nshot renumbering")
from h3_planner.nodes_prompt import renumber_shots  # noqa: E402

# A segment cut from shots 4-6 came back numbered 4, 5, 6, and H3 read that as
# a video starting on its fourth shot.
check("shots are renumbered from 1",
      renumber_shots("[Shot 4] a. [Shot 5] At 00:02.000 b. [Shot 6] At 00:04.000 c."),
      "[Shot 1] a. [Shot 2] At 00:02.000 b. [Shot 3] At 00:04.000 c.")
check("already-correct numbering is unchanged",
      renumber_shots("[Shot 1] a. [Shot 2] b."), "[Shot 1] a. [Shot 2] b.")
check("text without shot markers is untouched",
      renumber_shots("no markers here"), "no markers here")

print("\ninvented tags and asset names")
from h3_planner.nodes_prompt import (asset_names, strip_asset_names,  # noqa: E402
                                     strip_unknown_tags, enforce_audio_exact,
                                     _allowed_tags)

# Verbatim from a real generation: <Video 1> invented with no video connected,
# the track's file name printed after the tag, and sound described on top of a
# supplied audio asset.
REAL = {
    "subject_definitions": "<Subject 1> is the male character, as defined in <Picture 1>.",
    "summary": "[reference generation] A street market sequence featuring <Subject 1>.",
    "retention_analysis": (
        "<Subject 1> (appears in [Shot 1]): fully_preserved - consistent identity. "
        "<Picture 1> ([Shot 1] keyframe): fully_preserved - establishes composition. "
        "<Video 1> (cut structure): weak_reference - provides pacing but not direct continuity. "
        "<Audio 1> (rapping audio): reference - style and rhythm guide vocal "
        "performance without copying exact lyrics."),
    "detailed_description": (
        "[Shot 4] At 00:10.000, the shot moves to a metro platform, the camera "
        "zooming in on <Subject 1>. Background noise includes distant train "
        "sounds and chatter from commuters."),
    "overall_soundscape": ("Ambient street sounds blend with rhythmic beats from "
                           "<Audio 1>, featuring layered percussion and basslines."),
    "non_diegetic_music": ("<Audio 1> (Phase_3_Ka_Loha__Heavy_B) provides the core "
                           "musical foundation throughout the segment."),
}
CAST = {"members": [
    {"tag": "<Subject 1>", "kind": "Subject", "slot": 1, "key": "hero",
     "file": "hf_20260831_a3b2cc5d.png"},
    {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "key": "song",
     "file": "Phase_3_Ka_Loha__Heavy_B.mp3"}], "has_audio": True}

check("an image slot permits both readings",
      {k: sorted(v) for k, v in _allowed_tags(CAST).items()},
      {"Subject": [1], "Picture": [1], "Audio": [1]})
check("only upload file names are scrubbed, not typed keys",
      sorted(asset_names(CAST)), ["Phase_3_Ka_Loha__Heavy_B", "hf_20260831_a3b2cc5d"])

_p = {k: strip_asset_names(v, asset_names(CAST)) for k, v in REAL.items()}
_p, _invented = strip_unknown_tags(_p, _allowed_tags(CAST))
_p, _fixed = enforce_audio_exact(_p, "<Audio 1>")
_all = " ".join(_p.values())

check("the invented video tag is removed", _invented, ["<Video 1>"])
check("no video tag survives anywhere",
      bool(re.search(r"<\s*Video", _all, re.I)), False)
check("the track file name is gone", "Phase_3" in _all, False)
check("the image file name is gone", "hf_2026" in _all, False)
check("subject and picture are not false positives",
      ("<Subject 1>" in _all and "<Picture 1>" in _all), True)
check("the shot survives tag stripping", "[Shot 4]" in _p["detailed_description"], True)
check("audio retention becomes fully_copy",
      "<Audio 1>: fully_copy, all shots." in _p["retention_analysis"], True)
check("the hedging audio line is gone", "exact lyrics" in _all, False)
check("invented ambience is dropped from the description",
      "Background noise" in _p["detailed_description"], False)
check("the soundscape says the audio is used exactly",
      _p["overall_soundscape"],
      "All audio in this segment comes from <Audio 1> and is used exactly as "
      "supplied. No additional sound is present.")
check("no invented sound words remain",
      any(w in _all.lower() for w in
          ("layered percussion", "train sounds", "ambient street", "basslines")),
      False)

# a video asset genuinely in the cast must not be stripped
WITH_VIDEO = dict(CAST)
WITH_VIDEO["members"] = CAST["members"] + [
    {"tag": "<Video 1>", "kind": "Video", "slot": 0, "key": "plate", "file": "plate.mp4"}]
_kept, _removed = strip_unknown_tags(
    {"retention_analysis": "<Video 1> (source): weak_reference - pacing."},
    _allowed_tags(WITH_VIDEO))
check("a real video tag is left alone", _removed, [])
check("and its line survives", "<Video 1>" in _kept["retention_analysis"], True)

# no audio in the cast -> the sound sections are the model's own
_no_audio = {"overall_soundscape": "Footsteps on gravel and distant thunder.",
             "non_diegetic_music": "A low drone builds under the scene.",
             "retention_analysis": "<Subject 1>: fully_preserved.",
             "detailed_description": "[Shot 1] a wide."}
check("without an audio asset the soundscape is untouched",
      dict(_no_audio) == _no_audio, True)

print("\nformats stay general")
from h3_planner.nodes_prompt import FORMATS, FORMAT_GUIDANCE  # noqa: E402

check("auto adds nothing", FORMAT_GUIDANCE["auto"], "")
check("every format is covered", sorted(FORMATS),
      ["auto", "cinematic ad", "explainer", "music video", "product",
       "short film", "ugc"])
check("no format's guidance leaks into the others",
      all("music" not in FORMAT_GUIDANCE[f].lower()
          for f in FORMATS if f != "music video"), True)
check("the node exposes no format-specific inputs",
      [k for k in _P.INPUT_TYPES()["optional"]
       if k in ("lyrics", "music_track", "lyrics_language",
                "transcribe_track", "audio_retention")], [])

print("\none shared wording for the cast")
from h3_planner.nodes_prompt import canonical_subjects  # noqa: E402

# Each segment is written by its own model call, so each re-words the cast and
# H3 generates a slightly different person every time. One canonical block,
# filtered per segment, is the only way the wording can be identical.
CANON = ("<Subject 1> is the young man in a black hooded jacket with a thin "
         "gold chain.\n"
         "<Subject 2> is the red bottle with a matte finish and a white label.\n"
         "<Audio 1> is the supplied reference track.")

check("a segment gets only the tags it cites",
      canonical_subjects(CANON, {"Subject": {1}, "Audio": {1}}).splitlines(),
      ["<Subject 1> is the young man in a black hooded jacket with a thin gold chain.",
       "<Audio 1> is the supplied reference track."])
check("a different segment gets a different subset",
      canonical_subjects(CANON, {"Subject": {2}}).splitlines(),
      ["<Subject 2> is the red bottle with a matte finish and a white label."])
check("citing nothing yields nothing", canonical_subjects(CANON, {}), "")
check("the wording is byte-identical between segments",
      canonical_subjects(CANON, {"Subject": {1}}),
      canonical_subjects(CANON, {"Subject": {1}, "Picture": {9}}))

print("\nstall diagnosis")
from h3_planner import store  # noqa: E402
tl = store.normalize_timeline(
    {"segments": [{"id": c, "duration": 3.0, "prompt": c} for c in "abcd"]},
    {"name": "t", "fps": 24.0, "base_seed": 1, "default_duration": 6.0})
tl["segments"][0]["clips"] = {"draft": {"file": "a.mp4"}}
tl["segments"][0]["state"] = "draft"
for seg in tl["segments"][1:]:
    seg["state"] = store.RUNNING          # three runs that never came back
check("counts see the stuck ones",
      store.counts(tl, "draft"),
      {"done": 1, "pending": 0, "running": 3, "failed": 0, "locked": 0,
       "total": 4})
check("a stall does not report itself as finished",
      "still marked running" in store.select(tl, "draft")[1], True)
check("unstick frees them", sorted(store.unstick(tl)), ["b", "c", "d"])
check("and work resumes", store.select(tl, "draft")[0]["id"], "b")


print("")
print("shot timestamps, in every form a treatment actually writes them")

# The form that shipped: "At MM:SS.mmm". The form that broke a real 52s
# render: the same times with the "At" dropped. Both must parse, or every
# segment collapses to the 0.5s fallback in parse_shots and the last shot
# inherits the entire runtime.
FORMS = {
    "guide form, with At": "[Shot 1] cinematic wide.{n}[Shot 2] At 00:05.500 he turns.{n}[Shot 3] At 00:09.000 he leaves.",
    "no At, bare time": "[Shot 1] cinematic wide.{n}[Shot 2] 00:05.500 he turns.{n}[Shot 3] 00:09.000 he leaves.",
    "em dash after time": "[Shot 1] cinematic wide.{n}[Shot 2] 00:05.500 " + chr(8212) + " he turns.{n}[Shot 3] 00:09.000 " + chr(8212) + " he leaves.",
    "hyphen separator": "[Shot 1] cinematic wide.{n}[Shot 2] - 00:05.500 he turns.{n}[Shot 3] - 00:09.000 he leaves.",
    "at sign": "[Shot 1] cinematic wide.{n}[Shot 2] @ 00:05.500 he turns.{n}[Shot 3] @ 00:09.000 he leaves.",
    "bracketed": "[Shot 1] cinematic wide.{n}[Shot 2] (00:05.500) he turns.{n}[Shot 3] (00:09.000) he leaves.",
    "lowercase at": "[Shot 1] cinematic wide.{n}[Shot 2] at 00:05.500 he turns.{n}[Shot 3] at 00:09.000 he leaves.",
}
for label, raw in FORMS.items():
    shots = splitter.parse_shots(raw.format(n=chr(10)), 12.0)
    check("%s -> three shots" % label, len(shots), 3)
    check("%s -> times read" % label,
          [round(x["start"], 3) for x in shots], [0.0, 5.5, 9.0])

zero = splitter.parse_shots(
    "[Shot 0] 00:00.000 cold open.{n}[Shot 1] 00:02.000 the drop.".format(n=chr(10)), 6.0)
check("shots numbered from zero still parse", [x["number"] for x in zero], [0, 1])
check("and keep their times", [round(x["start"], 3) for x in zero], [0.0, 2.0])

# Descriptive text opening with a number must NOT read as a timestamp. The
# MM:SS colon is what makes loosening the "At" safe.
for opener in ("2d, illustration art style, a 100mm macro shot",
               "100mm macro on the chain",
               "24mm from ground, low angle",
               "18mm handheld rush-in"):
    body = "[Shot 1] wide.{n}[Shot 2] {o} he turns.".format(n=chr(10), o=opener)
    got = splitter.parse_shots(body, 10.0)
    check("%r is not mistaken for a time (read %.3f)"
          % (opener[:28], got[1]["start"]), got[1]["start"], 0.0)

untimed = ("detailed_description:{n}" + "{n}".join(
    "[Shot %d] he does a thing." % i for i in range(1, 6))).format(n=chr(10))
_, _, warns = splitter.split_treatment(untimed, 52.0, 9.417, 2.0, 2)
check("an untimed treatment warns loudly",
      any("NO TIMESTAMPS PARSED" in w for w in warns), True)

timed = ("detailed_description:{n}" + "{n}".join(
    "[Shot %d] 00:%02d.000 he does a thing." % (i, i * 5)
    for i in range(1, 6))).format(n=chr(10))
segs, _, warns = splitter.split_treatment(timed, 25.0, 9.417, 2.0, 2)
check("a timed one does not",
      any("NO TIMESTAMPS" in w for w in warns), False)
check("and its segments are real lengths (%s)"
      % [s["target_duration"] for s in segs],
      all(s["target_duration"] > 1.0 for s in segs), True)
check("that walk the song (%s)" % [s["audio_start"] for s in segs],
      [s["audio_start"] for s in segs]
      == sorted(s["audio_start"] for s in segs), True)

print("")
print("audio role - who makes the sound in the supplied track")

# The failure this pins: a rap track was connected, the prompt said the audio
# was reused exactly, and nothing anywhere said the subject was rapping. H3
# played the track over a man sitting silently.
base = {
    "subject_definitions": "<Subject 1> a young man.",
    "summary": "A segment.",
    "retention_analysis": "<Subject 1>: fully_preserved.",
    "detailed_description": "[Shot 1] <Subject 1> sits on scrap rims.",
    "overall_soundscape": "Ambient street sounds with layered percussion.",
    "non_diegetic_music": "A driving hip-hop beat.",
}

performed, _ = nodes_prompt.enforce_audio_exact(
    dict(base), "<Audio 1>", "performed on camera", "<Subject 1>")
sound = performed["overall_soundscape"]
check("a performed track names the performer",
      "<Subject 1>" in sound, True)
check("and says the mouth matches the words",
      "mouth matches the words" in sound, True)
check("while still forbidding extra sound",
      "No additional sound is present" in sound, True)
check("invented ambience is still gone",
      "layered percussion" in " ".join(performed.values()), False)

bg, _ = nodes_prompt.enforce_audio_exact(dict(base), "<Audio 1>",
                                         "background only")
check("background only says nothing about a performer",
      "mouth" in bg["overall_soundscape"], False)
check("voice-over says nobody on screen makes it",
      "no mouth matches it" in nodes_prompt.enforce_audio_exact(
          dict(base), "<Audio 1>", "voice-over (off camera)")[0]
      ["overall_soundscape"], True)

# Every role still pins the retention marker to fully_copy.
for role in nodes_prompt.AUDIO_ROLES:
    got, _ = nodes_prompt.enforce_audio_exact(dict(base), "<Audio 1>", role)
    check("%s -> audio retention is fully_copy" % role,
          "<Audio 1>: fully_copy, all shots." in got["retention_analysis"], True)

# A performance sentence must survive the invented-ambience strip. Before,
# "breath" plus "distant" got the whole line deleted.
perf = dict(base)
perf["detailed_description"] = (
    "[Shot 1] wide on the lane. His breath carries the rhythm, the vocal distant in the mix. Ambient traffic hums in the background.")
got, _ = nodes_prompt.enforce_audio_exact(perf, "<Audio 1>",
                                          "performed on camera", "<Subject 1>")
check("the performance line survives",
      "breath carries the rhythm" in got["detailed_description"], True)
check("the pure-ambience line does not",
      "Ambient traffic hums" in got["detailed_description"], False)

# The system block must actually carry the instruction to the model.
blk = nodes_prompt.audio_system_block("<Audio 1>", "performed on camera")
check("the performed block tells the model to say they are rapping",
      "rapping, singing" in blk, True)
check("and to keep the actual words", "in quotation marks" in blk, True)
check("it still bans invented ambience", "Do not invent ambience" in blk, True)
check("no audio asset means no audio block",
      nodes_prompt.audio_system_block("", "performed on camera"), "")
check("background only does not ask for a performance",
      "rapping, singing" in nodes_prompt.audio_system_block(
          "<Audio 1>", "background only"), False)

print("")
print("speaker IDs survive the file-name strip")
_names = {"my_track_02", "presenter_01"}
for _text, _want, _label in (
    ("<Subject 1> (S1) says, hello.", "(S1)", "a speaker id is kept"),
    ("<Subject 2> (S1,S2) speak.", "(S1,S2)", "a compound id is kept"),
):
    check(_label, _want in nodes_prompt.strip_asset_names(_text, _names), True)
for _text, _gone, _label in (
    ("<Audio 1> (my_track_02) drives it.", "my_track_02", "a track name still goes"),
    ("<Subject 1> (presenter_01) walks.", "presenter_01", "a known stem still goes"),
    ("<Picture 1> (hf_2026_0c12.png) anchors.", ".png", "a path still goes"),
):
    check(_label, _gone in nodes_prompt.strip_asset_names(_text, _names), False)
check("both at once: id kept, name dropped",
      nodes_prompt.strip_asset_names(
          "<Subject 1> (S1) and <Audio 1> (my_track_02).", _names),
      "<Subject 1> (S1) and <Audio 1>.")

print("")
print("tag binding - brackets the model forgot")
_allowed = {"Subject": {1, 2}, "Audio": {1}}
for _text, _want, _label in (
    ("Subject 1 walks toward the camera holding Subject 2.",
     "<Subject 1> walks toward the camera holding <Subject 2>.",
     "bare words are bound"),
    ("<Subject 1> is bracketed, Subject 2 is not.",
     "<Subject 1> is bracketed, <Subject 2> is not.",
     "an existing tag is not double-bracketed"),
    ("Subject 1's hand grips Subject 2.",
     "<Subject 1>'s hand grips <Subject 2>.", "possessives survive"),
    ("subject 1 and SUBJECT 2.", "<Subject 1> and <Subject 2>.",
     "case does not matter"),
    ("Subject 12 is not a cast member.", "Subject 12 is not a cast member.",
     "a longer number is not a partial match"),
    ("Subject 3 is not in the cast.", "Subject 3 is not in the cast.",
     "a number the cast lacks is left alone"),
    ("All audio comes from Audio 1.", "All audio comes from <Audio 1>.",
     "audio binds too"),
    ("The subject walks; a subjective shot.",
     "The subject walks; a subjective shot.", "ordinary words are untouched"),
):
    check(_label, nodes_prompt.bind_tags({"d": _text}, _allowed)[0]["d"], _want)

check("with no cast nothing is bound",
      nodes_prompt.bind_tags({"d": "Subject 1"}, {})[0]["d"], "Subject 1")
check("it reports what it bound",
      bool(nodes_prompt.bind_tags({"d": "Subject 1 here"}, _allowed)[1]), True)

# the real failure: metadata bracketed, description bare, so H3 bound nothing
_real = {
    "subject_definitions": "<Subject 1>, tall. <Subject 2>, a bottle.",
    "detailed_description": "[Shot 1] Subject 1 walks along the sand holding "
                            "Subject 2 in her right hand.",
}
_fixed, _ = nodes_prompt.bind_tags(_real, _allowed)
check("the description gets its brackets",
      "<Subject 1>" in _fixed["detailed_description"]
      and "<Subject 2>" in _fixed["detailed_description"], True)
check("and the metadata is unchanged",
      _fixed["subject_definitions"], _real["subject_definitions"])
print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
