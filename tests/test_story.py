"""The Story Planner's deterministic half, and the render estimate.

The model calls are stubbed: what is tested here is everything that must be
right whatever the model returns — durations landing on the ladder, the total
matching the brief, scenes and handover threading through, and the Segment
Prompter being left exactly as it was.

    python tests/test_story.py
"""

import io
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from h3_planner import ladder, nodes_story, store  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def ok(label, condition, detail=""):
    check(label + (" (%s)" % detail if detail else ""), bool(condition), True)


def close(label, got, want, tol=1e-6):
    ok(label, abs(got - want) <= tol, "%.4f vs %.4f" % (got, want))


PROJECT = {
    "name": "t", "fps": 24.0, "frame_modulus": 17, "frame_remainder": 5,
    "frame_minimum": 5, "snap": "up", "max_seconds": 10.5, "base_seed": 7,
    "default_duration": 8.0, "width": 544, "height": 960, "megapixels": 0.5,
}

tmp = tempfile.mkdtemp(prefix="h3story_")
try:
    # ----------------------------------------------------------------------
    print("\nthe clip-length menu")
    rungs = nodes_story.allowed_rungs(PROJECT, 5.0, 10.0)
    seconds = [round(s, 3) for _, s in rungs]
    check("only ladder values between 5s and 10s", seconds,
          [5.167, 5.875, 6.583, 7.292, 8.0, 8.708, 9.417])
    ok("every one is a real rung",
       all(f % 17 == 5 for f, _ in rungs), str([f for f, _ in rungs]))
    check("8.000s is on the menu", 8.0 in seconds, True)
    check("a 30s brief wants 4 clips at the top of the window",
          nodes_story.segment_count(30.0, seconds), 4)

    # ----------------------------------------------------------------------
    print("\nreconciling durations against the brief")
    menu = [s for _, s in rungs]

    for total, wanted in ((30.0, [8, 8, 8, 8]),
                          (60.0, [10, 10, 10, 10, 10, 10, 10]),
                          (15.0, [7.5, 7.5]),
                          (45.0, [6, 9, 5, 8, 9, 8])):
        got = nodes_story.reconcile(wanted, total, menu)
        ok("%.0fs brief -> every length is on the ladder" % total,
           all(any(abs(g - m) < 1e-9 for m in menu) for g in got), str(got))
        drift = abs(sum(got) - total)
        ok("%.0fs brief -> total lands within half a rung" % total,
           drift <= 0.36, "%.3fs drift, sum %.3f" % (drift, sum(got)))

    # a model that ignores the menu entirely still produces a valid plan
    messy = nodes_story.reconcile([8.4, 3.1, 12.9, 7.05], 30.0, menu)
    ok("off-menu values are snapped back",
       all(any(abs(g - m) < 1e-9 for m in menu) for g in messy), str(messy))
    ok("and the total still lands", abs(sum(messy) - 30.0) <= 0.36,
       "%.3f" % sum(messy))
    check("no segment is dropped", len(messy), 4)
    check("an empty plan stays empty", nodes_story.reconcile([], 30.0, menu), [])

    print("")
    print("an under-planned brief is detectable, not silent")
    short = nodes_story.reconcile([10] * 6, 60.0, menu)
    ok("six clips cannot reach 60s", abs(sum(short) - 60.0) > 0.36,
       "%.3fs" % sum(short))
    ok("but every value is still a legal rung",
       all(any(abs(g - m) < 1e-9 for m in menu) for g in short))
    check("segment_count would have asked for seven",
          nodes_story.segment_count(60.0, menu), 7)
    seven = nodes_story.reconcile([9] * 7, 60.0, menu)
    ok("and seven reaches it", abs(sum(seven) - 60.0) <= 0.36,
       "%.3fs" % sum(seven))

    # ----------------------------------------------------------------------
    print("\nlaying out the timeline")
    beats = {
        "style_prefix": "Cinematic 35mm anamorphic, warm daylight grade",
        "world": "A rooftop in Delhi at golden hour. He wears a black jacket.",
        "subject_definitions": "<Subject 1> a young man, close-cropped hair, black jacket.",
        "segments": [
            {"seconds": 8.0, "scene": 1, "scene_name": "rooftop",
             "action": "he walks to the ledge",
             "opens_from": "standing by the door",
             "ends_with": "at the ledge, hands on the rail, camera behind him"},
            {"seconds": 8.0, "scene": 1, "scene_name": "rooftop",
             "action": "he looks out over the city",
             "opens_from": "WRONG - should be overwritten",
             "ends_with": "turning back, sun behind him"},
            {"seconds": 8.0, "scene": 2, "scene_name": "stairwell",
             "action": "he descends",
             "opens_from": "top of the stairs",
             "ends_with": "at the bottom, pushing the door"},
        ],
    }
    menu2 = [s for _, s in nodes_story.allowed_rungs(PROJECT, 5.0, 10.0)]
    lengths = nodes_story.reconcile([b["seconds"] for b in beats["segments"]],
                                    24.0, menu2)

    segments, clock = [], 0.0
    for i, (row, secs) in enumerate(zip(beats["segments"], lengths)):
        prev = segments[-1] if segments else None
        same = prev is not None and prev["scene"] == int(row["scene"])
        segments.append({
            "id": "seg_%02d" % (i + 1), "scene": int(row["scene"]),
            "scene_name": row["scene_name"], "action": row["action"],
            "opens_from": row["opens_from"], "ends_with": row["ends_with"],
            "target_duration": round(secs, 3),
            "link": "continue" if same else "cut",
            "prompt": "", "audio_start": round(clock, 3),
        })
        clock += secs
    for prev, seg in zip(segments, segments[1:]):
        if seg["scene"] == prev["scene"] and prev["ends_with"]:
            seg["opens_from"] = prev["ends_with"]

    check("segment 2 opens on what segment 1 ended with",
          segments[1]["opens_from"], segments[0]["ends_with"])
    check("a scene change does NOT inherit the handover",
          segments[2]["opens_from"], "top of the stairs")
    check("chaining never crosses a scene boundary",
          [s["link"] for s in segments], ["cut", "continue", "cut"])
    close("the audio clock starts at zero", segments[0]["audio_start"], 0.0)
    close("and accumulates on the trimmed plan",
          segments[2]["audio_start"], segments[0]["target_duration"]
          + segments[1]["target_duration"], 0.002)

    timeline = store.normalize_timeline(
        {"name": "t", "context": {"style_prefix": beats["style_prefix"]},
         "segments": segments}, PROJECT)
    check("scenes survive normalisation",
          [s["scene"] for s in timeline["segments"]], [1, 1, 2])
    check("so does the handover", timeline["segments"][1]["opens_from"],
          segments[0]["ends_with"])
    ok("every segment got a seed",
       all(s["seed"] for s in timeline["segments"]))

    for seg in timeline["segments"]:
        frames = ladder.snap_frames(seg["target_duration"], 24.0)
        seg["render_frames"] = frames
        seg["render_duration"] = frames / 24.0
    ok("the plan is already on the ladder, so nothing is wasted",
       all(abs(s["render_duration"] - s["target_duration"]) < 1e-6
           for s in timeline["segments"]),
       str([(s["target_duration"], s["render_duration"])
            for s in timeline["segments"]]))


    print("")
    print("the checks that catch a bad plan before it renders")
    planner = nodes_story.H3PlannerStoryPlanner()

    # 26 words into a 5.167s clip is what the model actually returned.
    over = planner._overlong([
        {"id": "seg_01", "target_duration": 5.167,
         "dialogue": ("I have always believed that fragrance is not just about "
                      "scent it is about feeling it is about capturing a moment "
                      "and making it last forever")},
        {"id": "seg_02", "target_duration": 8.0,
         "dialogue": "Horizon. Bottled."},
        {"id": "seg_03", "target_duration": 5.0, "dialogue": ""},
    ])
    check("only the over-long line is flagged", len(over), 1)
    ok("and it names the segment", over[0].startswith("seg_01"), over[0][:60])
    ok("and says it will be cut off", "cut off" in over[0])

    untagged = planner._untagged([
        {"id": "seg_01", "action": "she lifts the bottle",
         "opens_from": "standing", "ends_with": "holding it"},
        {"id": "seg_02", "action": "<Subject 1> lifts <Subject 2>",
         "opens_from": "<Subject 1> stands", "ends_with": "<Subject 2> raised"},
    ], {"Subject": {1, 2}})
    check("prose-only beats are flagged", untagged, ["seg_01"])
    check("tagged beats are not", planner._untagged([
        {"id": "x", "action": "<Subject 1> walks", "opens_from": "",
         "ends_with": ""}], {"Subject": {1}}), [])
    check("with no cast there is nothing to flag", planner._untagged([
        {"id": "x", "action": "she walks", "opens_from": "",
         "ends_with": ""}], {}), [])

    # The Timeline node re-authors through a field whitelist, and dropped every
    # Story Planner field: the script and the scene grouping vanished the
    # moment the plan passed downstream.
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src = io.open(os.path.join(root, "h3_planner", "nodes_plan.py"),
                   encoding="utf-8").read()
    _whitelist = _src.split("authored = {")[1].split("for seg in timeline")[0]
    for _field in ("scene", "scene_name", "opens_from", "ends_with", "dialogue"):
        ok("the Timeline node carries %s downstream" % _field,
           (chr(34) + _field + chr(34)) in _whitelist)

    print("")
    print("the airlock and the reference images")
    for _txt, _label in ((nodes_story.BEATS_SYSTEM, "the plan"),
                         (nodes_story.PROSE_SYSTEM, "the prose")):
        ok("%s explains the airlock" % _label, "AIRLOCK" in _txt.upper())
        ok("%s ties the hold to the clip, not a flat 2s" % _label,
           "two seconds" not in _txt)
        ok("%s keeps the hold silent" % _label,
           "no dialogue" in _txt or "free of dialogue" in _txt)
        ok("%s warns a still hold reads as a freeze" % _label, "freeze" in _txt)
    ok("the plan warns that contradictions become additions",
       "addition" in nodes_story.BEATS_SYSTEM.lower())

    ok("the plan is told the supplied image is the truth",
       "THAT IMAGE IS THE TRUTH" in nodes_story.BEATS_SYSTEM)
    ok("and not to invent appearance",
       "inventing one from the brief" in nodes_story.BEATS_SYSTEM)

    # The reference is a casting card, not a location scout. Sending images to
    # the planner made it read the studio backdrop as the advert's setting.
    ok("the image is for the subject only",
       "NOTHING ELSE" in nodes_story.BEATS_SYSTEM)
    ok("the backdrop is explicitly excluded",
       "backdrop" in nodes_story.BEATS_SYSTEM)
    ok("a white studio wall is called out by name",
       "white studio wall" in nodes_story.BEATS_SYSTEM)
    ok("the planner is told to invent the world",
       "YOU INVENT THE WORLD" in nodes_story.BEATS_SYSTEM)
    ok("and never to take it from a photograph",
       "NEVER take the world from a reference photograph"
       in nodes_story.BEATS_SYSTEM)
    ok("and that a note outranks it",
       "outranks" in nodes_story.BEATS_SYSTEM)

    # no cast, no engine: the helper must degrade quietly, not raise
    check("no cast means no images", nodes_story.cast_reference_images(None), [])
    check("a cast with no files means no images",
          nodes_story.cast_reference_images(
              {"members": [{"kind": "Subject", "file": ""}]}), [])
    check("audio members are never sent as pictures",
          nodes_story.cast_reference_images(
              {"members": [{"kind": "Audio", "file": "t.mp3"}]}), [])

    print("")
    print("an advert script, not continuous narration")
    planner = nodes_story.H3PlannerStoryPlanner()

    brief = planner._dialogue_brief("spoken lines", rungs, 30.0)
    ok("the budget is for the whole piece, not per segment",
       "WHOLE piece" in brief and "45 words" in brief, brief[:120])
    ok("silence is called structural", "SILENCE IS STRUCTURAL" in brief)
    ok("a silent middle beat is required", "beauty shot" in brief)
    ok("the hero shot is protected", "six words" in brief)
    check("dialogue=none still says nothing at all",
          "nobody speaks" in planner._dialogue_brief("none", rungs, 30.0), True)

    # the script the planner actually produced: ~98 words over 30s, every
    # segment speaking, and a closing line over the hero shot
    talky = [
        {"id": "seg_0%d" % i, "target_duration": 6.0,
         "dialogue": " ".join(["word"] * 25)} for i in range(1, 5)]
    shape = planner._script_shape(talky, 30.0)
    ok("an over-written script is flagged",
       any("continuous narration" in x for x in shape), str(shape)[:100])
    ok("and it says how much to cut", any("cut roughly" in x for x in shape))
    ok("wall-to-wall speech is flagged",
       any("every segment speaks" in x for x in shape))
    ok("a talky closing shot is flagged",
       any("hero shot" in x for x in shape))

    lean = [
        {"id": "seg_01", "target_duration": 8.0, "dialogue": "Some mornings you get decided."},
        {"id": "seg_02", "target_duration": 8.0, "dialogue": ""},
        {"id": "seg_03", "target_duration": 8.0, "dialogue": "Amber. Musk. Nothing else."},
        {"id": "seg_04", "target_duration": 6.0, "dialogue": "Horizon."},
    ]
    check("a well-shaped script passes clean",
          planner._script_shape(lean, 30.0), [])
    check("no dialogue at all is never flagged",
          planner._script_shape([{"id": "a", "target_duration": 8.0,
                                  "dialogue": ""}], 30.0), [])

    print("")
    print("segment bleed - one film written into every clip")
    P = nodes_story.H3PlannerStoryPlanner()
    PFX = "Cinematic 35mm anamorphic, warm daylight grade"

    WHOLE = ("[Shot 1] %s, she stands at the ledge, arms out toward the "
             "skyline as the sun drops behind the towers and the shadows run "
             "long across the concrete. [Shot 2] At 00:01.033 she turns to "
             "camera and lifts the bottle into the light, thumb along the "
             "cap. [Shot 3] At 00:02.067 she sets it on the parapet and "
             "steps back, leaving it centred against the last of the sun." % PFX)
    same = [{"id": "seg_0%d" % i, "dialogue": "",
             "prompt": {"detailed_description": WHOLE}} for i in (1, 2, 3)]
    found = P._bleed(same, PFX)
    ok("identical descriptions are caught", len(found) == 3, str(sorted(found)))
    ok("and it names the twin",
       any("reads the same as" in x for v in found.values() for x in v))

    # a segment carrying another segment's spoken line
    carried = [
        {"id": "seg_01", "dialogue": "This scent is not just perfume.",
         "prompt": {"detailed_description": "[Shot 1] she turns to camera."}},
        {"id": "seg_02", "dialogue": "A whisper of Arabian magic.",
         "prompt": {"detailed_description":
                    "[Shot 1] a wide of the roof. She says, This scent is not "
                    "just perfume. and lowers the bottle."}},
    ]
    found2 = P._bleed(carried, PFX)
    ok("a stolen line is caught", "seg_02" in found2, str(found2))
    ok("and it names whose line it was",
       any("seg_01" in x for x in found2.get("seg_02", [])))

    # healthy plans must not trip it
    healthy = [
        {"id": "seg_01", "dialogue": "Some mornings you get decided.",
         "prompt": {"detailed_description":
                    "[Shot 1] %s, she stands still at the parapet, wind "
                    "moving her hair, the city hazy and far below while the "
                    "light drops and shadows stretch over the concrete "
                    "roof deck toward the stairwell door." % PFX}},
        {"id": "seg_02", "dialogue": "",
         "prompt": {"detailed_description":
                    "[Shot 1] %s, the camera pushes in slowly on her hands "
                    "as she turns the bottle over, catching the gradient in "
                    "the glass and the small engraved lettering along the "
                    "shoulder of the flask." % PFX}},
    ]
    check("two genuinely different segments are left alone",
          P._bleed(healthy, PFX), {})
    check("an unwritten timeline is not bleed",
          P._bleed([{"id": "a", "dialogue": "", "prompt": ""}], PFX), {})
    check("a single segment can never bleed",
          P._bleed([healthy[0]], PFX), {})

    # the shot budget is what stops a whole advert fitting in one clip
    check("a short clip gets two shots", P._shot_budget(5.167), 2)
    check("a middling clip gets three", P._shot_budget(7.292), 3)
    check("a long clip gets four", P._shot_budget(9.417), 4)

    # the style prefix is copied into every segment on purpose and must not
    # be what makes them look alike
    ok("the prefix is discounted when comparing",
       PFX.lower() not in P._comparable("[Shot 1] %s, a thing." % PFX, PFX))
    ok("so are shot markers and timestamps",
       "shot" not in P._comparable("[Shot 2] At 00:01.033 a thing.", ""))

    print("")
    print("the airlock hold, scaled to the clip")
    P2 = nodes_story.H3PlannerStoryPlanner()
    for _d, _lo, _hi in ((5.167, 1.0, 1.3), (8.0, 1.6, 1.9), (9.417, 2.0, 2.0)):
        _h = P2._hold_seconds(_d)
        ok("a %.3fs clip holds %.2fs (%.0f%%)" % (_d, _h, 100 * _h / _d),
           _lo <= _h <= _hi, "%.3f" % _h)
    ok("never longer than two seconds", P2._hold_seconds(60.0) == 2.0)
    ok("never shorter than 0.8s", P2._hold_seconds(1.0) == 0.8)

    # the creator respaces evenly when the model times are invalid, which put
    # the first cut at exactly half the clip whatever the prompt asked for
    _even = "[Shot 1] she stands. [Shot 2] At 00:02.583, she turns."
    _held = P2.enforce_hold(_even, P2._hold_seconds(5.167), 5.167)
    ok("the midpoint cut is pulled back to the hold",
       "00:01.137" in _held, _held)
    ok("shot 1 keeps no timestamp", "[Shot 1] At" not in _held, _held)

    _three = ("[Shot 1] a. [Shot 2] At 00:02.000, b. [Shot 3] At 00:04.000, c.")
    _h3 = P2.enforce_hold(_three, 1.137, 5.167)
    import re as _re
    _times = [int(m) * 60 + float(sec) for m, sec in
              _re.findall("At ([0-9]+):([0-9]+[.][0-9]+)", _h3)]
    ok("later shots stay increasing", all(b > a for a, b in zip(_times, _times[1:])),
       str(_times))
    ok("and stay inside the clip", all(t < 5.167 for t in _times), str(_times))
    check("a single shot is left alone",
          P2.enforce_hold("[Shot 1] only one.", 1.0, 5.0), "[Shot 1] only one.")

    print("")
    print("the style prefix is written exactly once")
    PFX2 = "Cinematic 35mm anamorphic, warm daylight grade"
    for _body, _label in (
        ("[Shot 1] she walks the sand.", "a plain body gets the prefix"),
        ("%s%s [Shot 1] %s, she walks." % (PFX2, chr(10), PFX2),
         "a restated leading line is removed"),
        ("[Shot 1] %s, she walks." % PFX2, "an existing prefix is not doubled"),
        ("%s [Shot 1] she walks." % PFX2, "a bare lead-in is absorbed"),
    ):
        _out = P2._force_prefix(_body, PFX2)
        ok(_label, _out.count("Cinematic 35mm") == 1, _out[:100])
        ok("%s - and still opens on a shot" % _label, _out.startswith("[Shot 1]"),
           _out[:60])

    print("")
    print("the plan is told to vary the shot")
    for _needle in ("VARY THE SHOT", "NO PERSON in it", "macro",
                    "camera height", "closing shot is the hero"):
        ok("the beats brief demands %r" % _needle,
           _needle in nodes_story.BEATS_SYSTEM)

    print("")
    print("the runtime you asked for is enforced, not requested")
    _rows = [{"seconds": 5.0, "action": "b%d" % i} for i in range(6)]

    # the exact case: a window holding ONE rung, so reconcile cannot help
    _one = [s for _, s in nodes_story.allowed_rungs(PROJECT, 4.5, 5.2)]
    check("a 4.5-5.2s window holds exactly one clip length", len(_one), 1)
    _kept, _dropped = nodes_story.fit_count(_rows, 15.0, _one)
    check("six beats are cut to three", len(_kept), 3)
    check("and three were dropped", _dropped, 3)
    _lens = nodes_story.reconcile([r["seconds"] for r in _kept], 15.0, _one)
    ok("the plan lands near 15s, not 31s", abs(sum(_lens) - 15.0) < 1.0,
       "%.3fs" % sum(_lens))

    ok("the opening beat is kept", _kept[0]["action"] == "b0")
    ok("the closing beat is kept", _kept[-1]["action"] == "b5")
    ok("the survivor comes from the middle, not the front",
       _kept[1]["action"] not in ("b1",), _kept[1]["action"])

    # a plan that already fits is never touched
    _wide = [s for _, s in nodes_story.allowed_rungs(PROJECT, 4.4, 8.0)]
    _fits = [{"seconds": 5.0, "action": "b%d" % i} for i in range(3)]
    check("a plan that fits is left alone",
          nodes_story.fit_count(_fits, 15.0, _wide), (_fits, 0))
    check("too few beats are never padded",
          nodes_story.fit_count(_fits[:2], 30.0, _wide)[1], 0)

    _lo, _hi = nodes_story.feasible_counts(15.0, _one)
    check("15s of 5.167s clips is three of them", (_lo, _hi), (3, 3))
    _lo2, _hi2 = nodes_story.feasible_counts(30.0, _wide)
    ok("a wider window allows a range of counts", _hi2 > _lo2,
       "%d..%d" % (_lo2, _hi2))
    check("no rungs at all is not a crash",
          nodes_story.feasible_counts(15.0, []), (1, 1))
    # ----------------------------------------------------------------------
    print("\nthe render estimate")
    path = os.path.join(tmp, "eta", "timeline.json")
    store.save(path, timeline)
    seconds_left, samples = store.eta(timeline, "draft")
    check("no estimate before anything has rendered", (seconds_left, samples),
          (None, 0))

    # 192 frames in 450s = 2.34 s/frame
    timeline["segments"][0]["render_elapsed"] = 450.0
    timeline["segments"][0]["clips"] = {"draft": {"file": "x.mp4"}}
    timeline["segments"][0]["state"] = "draft"
    seconds_left, samples = store.eta(timeline, "draft")
    check("one sample is enough to project", samples, 1)
    ok("two segments left at that rate is about 15 minutes",
       880 < seconds_left < 920, "%.0fs" % seconds_left)

    # a slower second segment moves the median
    timeline["segments"][1]["render_elapsed"] = 900.0
    timeline["segments"][1]["clips"] = {"draft": {"file": "y.mp4"}}
    timeline["segments"][1]["state"] = "draft"
    seconds_left, samples = store.eta(timeline, "draft")
    check("both samples counted", samples, 2)
    ok("one segment left, projected off the slower rate",
       seconds_left > 440, "%.0fs" % seconds_left)

    check("finished timelines report nothing left",
          store.eta({"segments": [
              {"render_elapsed": 100.0, "render_frames": 192,
               "clips": {"draft": {"file": "a.mp4"}}, "state": "draft"}]}, "draft")[0], 0.0)

    check("90 seconds reads as minutes", store.format_duration(900), "15m")
    check("and a long one as hours", store.format_duration(6800), "1h 53m")
    check("a short one stays in seconds", store.format_duration(45), "45s")


    # ----------------------------------------------------------------------
    print("\nthe world reaches the prompt")
    # Their real world block, copied off Final_Ad_test's timeline.
    WORLD = "\n".join([
        "- Location: A sun-drenched corner cafe with exposed brick walls, "
        "vintage coffee machines, and mismatched wooden tables",
        "- Time of day: Late afternoon, golden hour light",
        "- Weather: Clear skies, soft breeze through open windows",
        "- Wardrobe: <Subject 1> wears blue velvet crop top",
        "- Palette: Earthy tones with pops of blue and gold",
        "- Lighting: Natural window light with subtle fill from overhead lamps",
    ])
    fields = nodes_story.world_fields(WORLD)
    check("the labelled world parses", sorted(fields),
          ["lighting", "location", "palette", "time of day", "wardrobe",
           "weather"])
    ok("the location keeps its detail",
       "exposed brick walls" in fields["location"], fields["location"][:60])
    ok("free prose still yields something rather than nothing",
       nodes_story.world_fields("A rooftop in Delhi at golden hour. Then more.")
       == {"": "A rooftop in Delhi at golden hour."})
    check("an empty world is empty", nodes_story.world_fields(""), {})

    place = nodes_story.world_sentence(fields)
    ok("the sentence names place, time and light",
       "corner cafe" in place and "golden hour" in place
       and "window light" in place, place[:90])

    # ---- the summary, which was "N/A" in all five segments ---------------
    seg = {"opens_from": "she stands in the doorway",
           "ends_with": "she stops at the counter", "link": "cut"}
    tag = nodes_story.task_prefix(
        {"members": [{"kind": "Subject"}, {"kind": "Audio"}]},
        "performed on camera")
    check("the task type is one the H3 guide allows", tag,
          "[reference generation + audio reuse]")
    check("a voice sample is audio reference, not audio reuse",
          nodes_story.task_prefix({"members": [{"kind": "Audio"}]},
                                  nodes_story.VOICE_SAMPLE),
          "[audio reference]")
    check("a chained first frame is keyframe completion",
          nodes_story.task_prefix({"members": [{"kind": "Subject"}]},
                                  "background only", chained=True),
          "[keyframe completion + reference generation]")

    summary = nodes_story.build_summary(seg, 1, 5, fields, tag, "cinematic ad")
    ok("it is no longer N/A", summary != "N/A" and len(summary) > 80,
       summary[:70])
    ok("it says which part of how many", "part 2 of 5" in summary, summary[:90])
    ok("and that the piece is continuous",
       "one continuous" in summary, summary[:110])
    ok("it carries the room", "corner cafe" in summary)
    ok("it carries the handover both ways",
       "opens on she stands in the doorway" in summary
       and "ends with she stops at the counter" in summary)
    ok("and introduces no reference label of its own",
       "<" not in summary, summary[:80])

    # The real handover text is the director's prose and cites the cast, so
    # the summary is built BEFORE the tag pass and gets bound and checked with
    # every other section. A clean stub hid this: on their own timeline the
    # opens_from read "<Subject 1> holds a cup of coffee in her right hand".
    tagged = nodes_story.build_summary(
        {"opens_from": "<Subject 1> holds the cup", "ends_with": "he stands",
         "link": "cut"}, 0, 2, fields, tag, "cinematic ad")
    ok("a handover that cites the cast keeps its tag", "<Subject 1>" in tagged,
       tagged[-90:])
    from h3_planner import nodes_prompt
    checked, _ = nodes_prompt.strip_unknown_tags(
        {"summary": tagged}, {"Subject": {1}})
    check("and a tag the cast really has survives the tag pass",
          "<Subject 1>" in checked["summary"], True)
    gone, _ = nodes_prompt.strip_unknown_tags(
        {"summary": tagged}, {"Subject": {2}})
    check("while one it does not have is taken out",
          "<Subject 1>" in gone["summary"], False)

    second = nodes_story.build_summary(
        {"opens_from": "she stops at the counter", "ends_with": "he stands",
         "link": "cut"}, 2, 5, fields, tag, "cinematic ad")
    ok("two segments describe the same room in the same words",
       nodes_story.world_sentence(fields) in summary
       and nodes_story.world_sentence(fields) in second)
    ok("but are not the same summary", summary != second)

    # ---- the two repairs on the description ------------------------------
    naked = ("[Shot 1] Cinematic 35mm, a medium shot of him looking up from "
             "the table as she crosses toward him.")
    fixed = nodes_story.anchor_world(naked, nodes_story.world_clause(fields),
                                     "Cinematic 35mm")
    ok("a description that names no location gets one",
       "The location is A sun-drenched corner cafe" in fixed, fixed[:110])
    ok("and it goes BEHIND the style prefix, which must lead [Shot 1]",
       fixed.startswith("[Shot 1] Cinematic 35mm, The location is"), fixed[:60])
    already = ("[Shot 1] Cinematic 35mm, the corner cafe, exposed brick behind "
               "the vintage coffee machines and mismatched wooden tables.")
    check("one that already names it is left alone",
          nodes_story.anchor_world(already, nodes_story.world_clause(fields),
                                   "Cinematic 35mm"),
          already)

    opens = "she stops at the counter, hands resting on the wood"
    landed = ("[Shot 1] Cinematic 35mm, she stops at the counter with her "
              "hands resting on the wood surface, then turns.")
    check("a shot that already opens where the last one ended is untouched",
          nodes_story.enforce_opening(landed, opens, "Cinematic 35mm"), landed)
    drifted = ("[Shot 1] Cinematic 35mm, a wide of the street outside, traffic "
               "passing in both directions.")
    joined = nodes_story.enforce_opening(drifted, opens, "Cinematic 35mm")
    ok("one that ignored the handover is given it",
       "The shot opens on %s." % opens in joined, joined[:130])
    ok("and the style prefix still leads the shot",
       joined.startswith("[Shot 1] Cinematic 35mm,"), joined[:40])

    # The bleed detector must not read the injected sentences as duplication:
    # they are identical on purpose, and the opening state is by definition a
    # repeat of the segment before it.
    planner = nodes_story.H3PlannerStoryPlanner
    one = planner._comparable(
        "[Shot 1] P, The location is W. The shot opens on state 0. he walks "
        "left past the window and stops.", "P")
    two = planner._comparable(
        "[Shot 1] P, The location is W. The shot opens on state 4. she turns "
        "the mug around twice and laughs.", "P")
    ok("the injected sentences are excluded from the comparison",
       "location is" not in one and "opens on" not in one, one[:60])
    import difflib
    ok("so two genuinely different segments do not read as bleed",
       difflib.SequenceMatcher(None, one, two).ratio() < 0.8,
       "%.2f" % difflib.SequenceMatcher(None, one, two).ratio())

    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    print("\na segment with no line must not be told anyone speaks")
    VOICES = [{"audio": "<Audio 1>", "subject": "<Subject 2>",
               "speaker": "(S2)"},
              {"audio": "<Audio 2>", "subject": "<Subject 1>",
               "speaker": "(S1)"}]
    base = {"overall_soundscape": "", "non_diegetic_music": "",
            "retention_analysis": "<Subject 1>: fully_preserved."}

    loud, _ = nodes_story.enforce_voice_reference(dict(base), VOICES, True)
    quiet, _ = nodes_story.enforce_voice_reference(dict(base), VOICES, False)
    ok("a speaking segment says the mouths match the words",
       "mouths match the words" in loud["overall_soundscape"])
    ok("a silent one says nobody speaks",
       quiet["overall_soundscape"].startswith("Nobody speaks in this segment"),
       quiet["overall_soundscape"][:60])
    ok("and that no mouth moves",
       "no mouth moves" in quiet["overall_soundscape"])
    ok("the silent one does not claim the dialogue is spoken on camera",
       "spoken on camera" not in quiet["overall_soundscape"])
    ok("a speaking segment marks the audio as reference",
       "<Audio 1>: reference" in loud["retention_analysis"],
       loud["retention_analysis"][:80])
    ok("a silent one marks it weak_reference, the guide's word for unused",
       "<Audio 1>: weak_reference" in quiet["retention_analysis"]
       and "<Audio 2>: weak_reference" in quiet["retention_analysis"],
       quiet["retention_analysis"][-90:])

    # ----------------------------------------------------------------------
    print("\na segment must not run into the next one")
    # Their seg_04, shortened: briefed to end on her standing up, written with
    # her walking out of the door as well, which is seg_05's whole job. The
    # exit then rendered twice, once at the end of one clip and again at the
    # start of the next.
    ENDS = "<Subject 1> stands up from the table, hands resting on the edge."
    overran = (
        "[Shot 1] Cinematic 35mm, close on <Subject 1> as she smiles, then a "
        "medium shot of her standing up. The shot ends with <Subject 1> "
        "standing up from the table, hands resting on the edge. "
        "[Shot 2] At 00:02.667, wide shot showing <Subject 1> walking out the "
        "front door into the sunlight. "
        "[Shot 3] At 00:04.000, the empty table she left behind.")
    check("three shots go in", len(nodes_story.shot_spans(overran)), 3)
    ok("the last shot does not land the planned ending",
       not nodes_story.ending_landed(overran, ENDS))
    check("only the first shot is kept", nodes_story.overran(overran, ENDS), 1)
    trimmed, dropped = nodes_story.trim_after_ending(overran, ENDS)
    check("two shots are dropped", dropped, 2)
    ok("what is left ends on the planned state",
       trimmed.rstrip().endswith("hands resting on the edge."), trimmed[-60:])
    ok("and the exit that belonged to the next segment is gone",
       "walking out the front door" not in trimmed)
    ok("the spoken line inside the kept shot survives untouched",
       "she smiles" in trimmed)

    # A segment that mentions its ending early AND closes on it is written
    # correctly: trimming that would delete good shots.
    fine = ("[Shot 1] Cinematic 35mm, <Subject 1> stands up from the table. "
            "[Shot 2] At 00:02.000, she gathers her coat, then settles again "
            "with her hands resting on the edge of the table as she stands up "
            "to leave.")
    check("a correctly closed segment is left alone",
          nodes_story.overran(fine, ENDS), 0)
    check("and nothing is dropped from it",
          nodes_story.trim_after_ending(fine, ENDS), (fine, 0))
    check("a single-shot segment is never trimmed",
          nodes_story.trim_after_ending("[Shot 1] anything at all.", ENDS),
          ("[Shot 1] anything at all.", 0))
    check("no planned ending, nothing to enforce",
          nodes_story.trim_after_ending(overran, ""), (overran, 0))

    # ----------------------------------------------------------------------
    print("\nthe audio role is checked against the brief")
    # Adding the voice-sample role did nothing on its own: their saved
    # workflow kept "performed on camera", so two voice samples came back as
    # one reused track and the report said nothing about it.
    P = nodes_story.H3PlannerStoryPlanner
    one = {"members": [{"kind": "Audio", "tag": "<Audio 1>"}]}
    two = {"members": [{"kind": "Audio", "tag": "<Audio 1>"},
                       {"kind": "Audio", "tag": "<Audio 2>"}]}
    BRIEF = "use <audio 1> for male and <audio 2> for female characters"
    said = P._audio_role_mismatch(one, "performed on camera", BRIEF)

    ok("their own brief is recognised, and it never says the word voice",
       "AUDIO ROLE LOOKS WRONG" in said, said[:60])
    ok("two audio references under a soundtrack role is wrong on its own",
       "Only one can be the soundtrack" in
       P._audio_role_mismatch(two, "performed on camera", "a cafe story"))
    ok("the note names the value to pick",
       nodes_story.VOICE_SAMPLE in
       P._audio_role_mismatch(two, "performed on camera", "x"))
    check("a music video with one track and no voice talk is left alone",
          P._audio_role_mismatch(
              one, "performed on camera",
              "a rapper on a rooftop, the track is <Audio 1>"), "")
    check("and so is a correctly set voice-sample run",
          P._audio_role_mismatch(two, nodes_story.VOICE_SAMPLE, BRIEF), "")
    check("no audio at all, nothing to say",
          P._audio_role_mismatch({"members": []}, "performed on camera",
                                 BRIEF), "")

    # That pattern was written with a word-boundary escape once and the escape
    # collapsed into a literal backspace, so it matched nothing and looked
    # perfect in every editor. Cheap to assert, impossible to see by reading.
    import io as _io
    raw = _io.open(nodes_story.__file__, "rb").read()
    for code in (7, 8, 11, 12):
        check("no stray control character %d in the source" % code,
              bytes([code]) in raw, False)

    print("\nthe Segment Prompter is untouched")
    from h3_planner import nodes_prompt
    ok("it still exports its node",
       "H3PlannerSegmentPrompter" in nodes_prompt.NODE_CLASS_MAPPINGS)
    ok("the planner reuses its enforcement rather than copying it",
       nodes_story.strip_unknown_tags is nodes_prompt.strip_unknown_tags
       and nodes_story.canonical_subjects is nodes_prompt.canonical_subjects
       and nodes_story.enforce_audio_exact is nodes_prompt.enforce_audio_exact)
    ok("both nodes are registered",
       {"H3PlannerStoryPlanner"} <= set(nodes_story.NODE_CLASS_MAPPINGS))

    import h3_planner
    for name in ("H3PlannerStoryPlanner", "H3PlannerSegmentPrompter",
                 "H3PlannerDispatcher", "H3PlannerVaultWrite"):
        ok("%s is loadable from the pack" % name,
           name in h3_planner.NODE_CLASS_MAPPINGS)

    planner = h3_planner.NODE_CLASS_MAPPINGS["H3PlannerStoryPlanner"]
    types = planner.INPUT_TYPES()
    ok("the planner asks for a brief", "idea" in types["required"])
    ok("and for the segment-length window",
       {"min_segment_seconds", "max_segment_seconds"} <= set(types["required"]))
    check("chaining is off by default",
          types["required"]["chain_first_frames"][1]["default"], False)
    check("one call up to 8 segments",
          types["required"]["single_call_max"][1]["default"], 8)
    check("it returns a timeline, a beat sheet and a report",
          planner.RETURN_NAMES, ("timeline", "beat_sheet", "report"))

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
