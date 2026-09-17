"""Run the nodes' real entry points, with the model and disk stubbed.

The unit tests exercise helper functions directly, which is exactly why two
NameErrors reached ComfyUI: the helpers were correct and the code that calls
them was not. `python -m py_compile` cannot catch an undefined name inside a
function body, so this file calls every node's FUNCTION for real.

    python tests/test_end_to_end.py
"""

import os
import re
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = tempfile.mkdtemp(prefix="h3planner_e2e_")
os.environ["USERPROFILE"] = os.environ["HOME"] = WORK

from h3_planner import engine, nodes_plan, nodes_prompt, nodes_run, store  # noqa: E402
from h3_planner import vocals as vocals_mod  # noqa: E402

try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:            # standalone: stand in for ComfyUI's blocker
    class ExecutionBlocker:
        def __init__(self, x=None):
            self.x = x
    nodes_run.ExecutionBlocker = ExecutionBlocker

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def ok(label, condition, detail=""):
    check(label + (("  " + detail) if detail and not condition else ""),
          bool(condition), True)


# -- stub the prompt engine so no model is needed ---------------------------

CALLS = {"n": 0, "systems": []}


def fake_generate(cfg, system, user, schema, required_keys=(), min_words=0,
                  images=None):
    CALLS["n"] += 1
    CALLS["systems"].append(system)
    i = CALLS["n"]
    return ({
        "subject_definitions": "<Subject 1> is the presenter, from <Picture 1>.",
        "summary": "[reference generation] Segment %d of the film." % i,
        # deliberately dirty: an invented tag, a file name, invented ambience
        "retention_analysis": (
            "<Subject 1>: fully_preserved. "
            "<Video 1> (cut structure): weak_reference - pacing. "
            "<Audio 1> (my_track_02): reference - guides the performance."),
        "detailed_description": (
            "<Subject 1> (S1) says, <d>[English] This city never asked me "
            "for permission.</d> "
            "[Shot %d] a %s shot, the camera %s toward <Subject 1>. "
            "Background noise includes distant traffic and chatter."
            % (i + 3, ["wide", "close", "low-angle", "tracking"][i % 4],
               ["pushes in", "tilts up", "tracks left", "arcs right"][i % 4])),
        "overall_soundscape": "Ambient room tone with layered percussion.",
        "non_diegetic_music": "<Audio 1> (my_track_02) drives the rhythm.",
    }, "stub")


engine.available = lambda: True
engine.backend_config = lambda **kw: types.SimpleNamespace(**kw)
engine.json_schema = lambda p, r: {}
engine.full_ref_system = lambda: "GUIDE"
engine.generate = fake_generate
engine.clean = lambda v: "" if v is None else str(v).strip()
engine.dedupe_subjects = lambda t: t
engine.fix_shot_times = lambda t, d: t
engine.normalize_labels = lambda t, d=0.0: t
engine.render_full_ref = lambda o, d=0.0: "\n\n".join(
    "%s:\n%s" % (k, o.get(k, "N/A")) for k in nodes_prompt.SECTIONS)

TREATMENT = (
    "detailed_description:\n"
    "[Shot 1] cinematic, wide on the presenter at the counter.\n"
    "[Shot 2] At 00:05.000 the camera cuts to the product in her hands.\n"
    "[Shot 3] At 00:10.000 the shot cuts to the logo on the wall.\n")

CAST = {
    "members": [
        {"tag": "<Subject 1>", "kind": "Subject", "slot": 1, "key": "presenter",
         "role": "character", "note": "", "file": "presenter_01.png"},
        {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "key": "vo",
         "role": "audio", "note": "", "file": "my_track_02.mp3"},
    ],
    "size": 1, "has_audio": True, "audio_tag": "<Audio 1>",
    "counts": {"Subject": 1, "Picture": 0, "Video": 0, "Audio": 1},
}


def project(name, render_pass="draft"):
    return nodes_plan.H3PlannerProject().build(
        name, 24.0, "9:16", render_pass, 0.3, 1.2, 32, 4242, 17, 5, 5, "up",
        6.0, 6.0, 10)[0]


# --------------------------------------------------------------------------
print("\nH3 Project")
proj = project("e2e")
ok("returns a project dict", isinstance(proj, dict))
check("both passes share an aspect",
      round(proj["width"] / proj["height"], 6),
      round(nodes_plan.resolution.pass_sizes(0.3, 1.2, "9:16", 32)[1][0]
            / nodes_plan.resolution.pass_sizes(0.3, 1.2, "9:16", 32)[1][1], 6))

print("\nH3 Segment Prompter — every format, with a cast that has audio")
for fmt in nodes_prompt.FORMATS:
    CALLS["n"] = 0
    CALLS["systems"] = []
    shutil.rmtree(os.path.join(WORK, "h3_planner_output"), ignore_errors=True)
    proj = project("e2e_" + fmt.replace(" ", "_"))
    timeline, report = nodes_prompt.H3PlannerSegmentPrompter().write(
        proj, TREATMENT, 15.0, 10.0, 1, fmt, "performed on camera",
        "Ollama (Local)",
        "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
    body = " ".join(
        " ".join(s["prompt"].values()) for s in timeline["segments"])
    problems = []
    if len(timeline["segments"]) != 3:
        problems.append("%d segments" % len(timeline["segments"]))
    if "<Video" in body:
        problems.append("invented video tag survived")
    if "my_track_02" in body or "presenter_01" in body:
        problems.append("file name survived")
    if "Background noise" in body or "layered percussion" in body:
        problems.append("invented ambience survived")
    if "fully_copy" not in body:
        problems.append("audio retention not enforced")
    if "[Shot 1]" not in body:
        problems.append("shots not renumbered")
    if fmt != "auto" and nodes_prompt.FORMAT_GUIDANCE[fmt].strip() \
            not in CALLS["systems"][0]:
        problems.append("format guidance not sent")
    if nodes_prompt.AUDIO_EXACT not in CALLS["systems"][0]:
        problems.append("audio rules not sent")
    ok("%-13s writes 3 clean segments" % fmt, not problems,
       "-> " + "; ".join(problems))

print("\nH3 Segment Prompter — no cast at all")
shutil.rmtree(os.path.join(WORK, "h3_planner_output"), ignore_errors=True)
CALLS["n"] = 0
CALLS["systems"] = []
proj = project("e2e_nocast")
timeline, report = nodes_prompt.H3PlannerSegmentPrompter().write(
    proj, TREATMENT, 15.0, 10.0, 1, "auto", "background only",
    "Ollama (Local)",
    "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0)
ok("runs with cast=None", len(timeline["segments"]) == 3)
ok("no audio rules without an audio asset",
   nodes_prompt.AUDIO_EXACT not in CALLS["systems"][0])
ok("nothing is stripped without a cast",
   "<Video 1>" in " ".join(timeline["segments"][0]["prompt"].values()))

print("\nH3 Segment Prompter — from an idea, no treatment")
shutil.rmtree(os.path.join(WORK, "h3_planner_output"), ignore_errors=True)
proj = project("e2e_idea")
timeline, report = nodes_prompt.H3PlannerSegmentPrompter().write(
    proj, "", 20.0, 5.0, 1, "product", "background only",
    "Ollama (Local)",
    "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST,
    idea="a countertop blender in a bright kitchen")
ok("plans even beats from an idea", len(timeline["segments"]) == 4)
ok("every beat gets a prompt",
   all(s["prompt"] for s in timeline["segments"]))

print("\nH3 Timeline and H3 Shot Dispatcher")
shutil.rmtree(os.path.join(WORK, "h3_planner_output"), ignore_errors=True)
proj = project("e2e_run")
timeline, _ = nodes_prompt.H3PlannerSegmentPrompter().write(
    proj, TREATMENT, 15.0, 10.0, 1, "auto", "background only",
    "Ollama (Local)",
    "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
result = nodes_plan.H3PlannerTimeline().build(
    proj, "inline", '{"segments":[]}', "merge", True, timeline=timeline,
    cast=CAST)
stored = result["result"][0]
ok("the timeline node accepts an upstream timeline", len(stored["segments"]) == 3)
ok("and hands the cards their JSON back",
   bool(result["ui"]["h3_timeline"][0]["authored"]))
check("it hands out a segment count to wire into the stitcher",
      result["result"][2], len(stored["segments"]))
check("one value per declared output", len(result["result"]),
      len(nodes_plan.H3PlannerTimeline.RETURN_TYPES))

dispatched = nodes_run.H3PlannerDispatcher().dispatch(
    proj, stored, "next_pending", 1, 1, 0, False, "full_reference_6", False,
    cast=CAST)
ok("the dispatcher reports on the node", bool(dispatched["ui"]["text"][0]))
out = dispatched["result"]
ok("the dispatcher claims a segment", out[0] is not None)
ok("and hands over a real prompt", "<Subject 1>" in out[1])
ok("with a legal frame count", out[2] % 17 == 5, "-> %r" % (out[2],))
check("ten outputs, as wired", len(out), 10)

# A blocked run finishes in 0.05s with no error and no output. From the canvas
# that is indistinguishable from nothing happening, so the reason has to reach
# the node itself.
print("\nnothing left to render says so, on the node")
finished = store.load(proj["timeline_path"])
for _seg in finished["segments"]:
    _seg["clips"] = {"draft": {"file": "x.mp4"}}
    _seg["state"] = "draft"
store.save(proj["timeline_path"], finished)

idle = nodes_run.H3PlannerDispatcher().dispatch(
    proj, finished, "next_pending", 1, 1, 0, False, "full_reference_6", False,
    cast=CAST)
msg = idle["ui"]["text"][0]
ok("a blocked run still returns a message", bool(msg))
ok("it counts what is done", "3 done of 3" in msg, "-> %r" % msg)
ok("and says what to do next", "render_pass" in msg, "-> %r" % msg)
ok("while still blocking every output",
   all(isinstance(v, nodes_run.ExecutionBlocker) for v in idle["result"]))

print("\nH3 Pass Gate — which branch runs")
try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:                                      # standalone
    class ExecutionBlocker:                            # noqa: D401
        def __init__(self, x=None):
            self.x = x
    nodes_run.ExecutionBlocker = ExecutionBlocker


def blocked(value):
    return isinstance(value, nodes_run.ExecutionBlocker)


LATENT = {"samples": "stub"}
for mode, want_draft, want_final in (("draft", True, False),
                                     ("final", False, True),
                                     ("one_go", True, True)):
    proj = project("e2e_gate_" + mode, mode)
    shot = {"segment_id": "seg_01", "pass": proj["render_pass"],
            "pass_mode": proj["pass_mode"], "take": 0}
    d_lat, d_shot, f_lat, f_shot, info = nodes_run.H3PlannerPassGate().gate(
        shot, LATENT)
    problems = []
    if blocked(d_lat) == want_draft:
        problems.append("draft latent")
    if blocked(d_shot) == want_draft:
        problems.append("draft shot")
    if blocked(f_lat) == want_final:
        problems.append("final latent")
    if blocked(f_shot) == want_final:
        problems.append("final shot")
    if want_draft and d_shot["pass"] != "draft":
        problems.append("draft pass not stamped")
    if want_final and f_shot["pass"] != "final":
        problems.append("final pass not stamped")
    ok("%-7s draft=%-5s final=%-5s" % (mode, want_draft, want_final),
       not problems, "-> " + ", ".join(problems))

check("draft mode claims the draft pass", project("e2e_p1", "draft")["render_pass"], "draft")
check("final mode claims the final pass", project("e2e_p2", "final")["render_pass"], "final")
check("one_go claims the final pass too",
      project("e2e_p3", "one_go")["render_pass"], "final")

print("\nbase resolution never changes with the pass")
sizes = {m: (project("e2e_r_" + m, m)["width"], project("e2e_r_" + m, m)["height"])
         for m in ("draft", "final", "one_go")}
check("every mode samples at the same base size",
      len(set(sizes.values())), 1)
check("and that base is the draft size", sizes["final"], (416, 736))
check("the upscale target is the final size",
      round(project("e2e_r2", "final")["upscale_megapixels"], 3),
      round(832 * 1472 / 1e6, 3))

print("\nboth passes land in the vault in one_go")
_tl = {"segments": [{"id": "a", "duration": 3.0, "prompt": "x", "clips": {},
                     "state": "pending", "spec_hash": "h", "index": 0,
                     "target_duration": 3.0}]}
_path = os.path.join(WORK, "one_go.json")
_tl = store.normalize_timeline(
    {"segments": [{"id": "a", "duration": 3.0, "prompt": "x"}]},
    project("e2e_onego", "one_go"))
store.save(_path, _tl)
store.claim(_path, "a", "final")
store.complete(_path, "a", "draft", {"file": "a_draft.mp4"})
store.complete(_path, "a", "final", {"file": "a_final.mp4"})
_seg = store.find(store.load(_path), "a")
check("both clips are stored", sorted(_seg["clips"]), ["draft", "final"])
check("the final is what stitches", _seg["chosen"], "final")
# and the order the two branches finish in must not matter
store.complete(_path, "a", "draft", {"file": "a_draft2.mp4"})
check("a late draft does not steal chosen",
      store.find(store.load(_path), "a")["chosen"], "final")

print("\nIS_CHANGED must never repeat")
# float("nan") is the usual way to force a re-run; on this ComfyUI build the
# dispatcher was served from cache anyway and the same segment was re-rendered
# forever. A token that cannot compare equal is the fix.
seen = {nodes_run.H3PlannerDispatcher.IS_CHANGED() for _ in range(500)}
check("500 calls give 500 distinct tokens", len(seen), 500)
ok("none of them is a float NaN", all(isinstance(v, str) for v in seen))
ok("every stateful node uses it",
   all(n.IS_CHANGED() != n.IS_CHANGED() for n in (
       nodes_run.H3PlannerDispatcher, nodes_run.H3PlannerVaultWrite,
       nodes_run.H3PlannerStitch, nodes_plan.H3PlannerTimeline)))

print("\nauto-advance cannot loop on one segment")
vw = nodes_run.H3PlannerVaultWrite()
tl_loop = {"segments": [{"id": "seg_02", "state": "draft", "clips": {},
                         "target_duration": 3.0}]}
key = ("/x.json", "seg_02", "draft", 0)
first = vw._auto_advance(tl_loop, "draft", None, 24, key=key)
second = vw._auto_advance(tl_loop, "draft", None, 24, key=key)
ok("the first advance is allowed", "STOPPED" not in first)
ok("a repeat of the same segment is refused", "STOPPED" in second)
ok("and it names the segment", "seg_02" in second)
ok("a different segment still advances",
   "STOPPED" not in vw._auto_advance(
       tl_loop, "draft", None, 24, key=("/x.json", "seg_03", "draft", 0)))

print("\nH3 Stitch Timeline — nothing rendered yet")
res = nodes_run.H3PlannerStitch().stitch(
    proj, stored, "chosen", True, "h3planner", 2, False)
ok("waits instead of raising", res["result"][0] == "")
ok("and says what it is waiting for", "waiting" in res["result"][3])

# The node grew a VIDEO output so the finished render can be wired to Save
# Video. While it is still waiting there is no video, and handing a None to
# a Save Video node would throw on every early queue of the render graph.
check("every output slot is filled", len(res["result"]),
      len(nodes_run.H3PlannerStitch.RETURN_TYPES))
check("the last one is the video", nodes_run.H3PlannerStitch.RETURN_TYPES[-1],
      "VIDEO")
ok("and that branch is blocked rather than fed a None",
   res["result"][-1] is None or type(res["result"][-1]).__name__
   == "ExecutionBlocker",
   repr(res["result"][-1]))

# min_clips = 0 means "every segment", so the node can sit in the render
# graph and stitch itself once the last card is done, with no number to keep
# in step with the timeline by hand.
whole = nodes_run.H3PlannerStitch().stitch(
    proj, stored, "chosen", True, "h3planner", 0, False)
ok("min_clips 0 waits for the whole timeline",
   "need %d to stitch" % len(stored["segments"]) in whole["result"][3],
   whole["result"][3].splitlines()[0])
ok("which is more than the default of 2", len(stored["segments"]) > 2,
   str(len(stored["segments"])))

# ...and the count to wire into it comes off the Timeline node.
check("the timeline reports its segment count",
      nodes_plan.H3PlannerTimeline.RETURN_NAMES[-1], "segment_count")

# A stitcher asked for finals must not quietly join drafts. vault.resolve
# falls back to the other pass on purpose, so the wait has to count clips of
# the pass actually asked for, or the end of the draft pass produces a draft
# video the report calls final.
from h3_planner import vault as _vault
_drafted = {s["id"]: {"pass": "draft", "take": 0, "duration": 3.0,
                      "width": 544, "height": 960, "has_audio": True,
                      "file": "x.mp4"} for s in stored["segments"]}
_real_resolve, _real_abs, _real_thumb = (_vault.resolve, _vault.abs_path,
                                         _vault.thumb_path)
_vault.resolve = lambda project, seg, prefer: _drafted.get(seg["id"])
_vault.abs_path = lambda project, clip: "/nowhere/x.mp4"
_vault.thumb_path = lambda project, clip: None
try:
    fin = nodes_run.H3PlannerStitch().stitch(
        proj, stored, "final", True, "h3planner", 0, False)
    ok("a final stitch does not join drafts", fin["result"][0] == "",
       fin["result"][3].splitlines()[0])
    ok("and it says which pass it is short of",
       "have a final clip" in fin["result"][3],
       fin["result"][3].splitlines()[0])
    # The same clips satisfy a draft stitch, which then really joins them.
    _real_concat = nodes_run.media.concat_trimmed
    nodes_run.media.concat_trimmed = (
        lambda clips, out, fps, w, h, include_audio=True:
        open(out, "wb").write(b"0" * 2048))
    try:
        dr = nodes_run.H3PlannerStitch().stitch(
            proj, stored, "draft", True, "h3planner", 0, False)
    finally:
        nodes_run.media.concat_trimmed = _real_concat
    ok("while a draft stitch is satisfied by the same clips",
       dr["result"][0].endswith(".mp4"), dr["result"][3].splitlines()[0])
    ok("and a final-preferring run would have named the borrowed passes",
       "MIXED PASSES" not in dr["result"][3])
finally:
    _vault.resolve, _vault.abs_path = _real_resolve, _real_abs
    _vault.thumb_path = _real_thumb

# ...and when it does fall back, it says so instead of passing a draft off
# as a final. Rule: never drop or substitute input silently.
_mixed = {s["id"]: {"pass": "draft", "take": 0, "duration": 3.0,
                    "width": 544, "height": 960, "has_audio": True,
                    "file": "x.mp4"} for s in stored["segments"]}
_mixed[stored["segments"][0]["id"]] = dict(_mixed[stored["segments"][0]["id"]],
                                           **{"pass": "final"})
_vault.resolve = lambda project, seg, prefer: _mixed.get(seg["id"])
_vault.abs_path = lambda project, clip: "/nowhere/x.mp4"
_vault.thumb_path = lambda project, clip: None
_real_concat = nodes_run.media.concat_trimmed
nodes_run.media.concat_trimmed = (
    lambda clips, out, fps, w, h, include_audio=True:
    open(out, "wb").write(b"0" * 2048))
try:
    mix = nodes_run.H3PlannerStitch().stitch(
        proj, stored, "final", True, "h3planner", 1, False)
    ok("one final is enough to start a final stitch",
       mix["result"][0].endswith(".mp4"), mix["result"][3].splitlines()[0])
    ok("and the borrowed drafts are named in the report",
       "MIXED PASSES" in mix["result"][3] and "draft" in mix["result"][3])
finally:
    nodes_run.media.concat_trimmed = _real_concat
    _vault.resolve, _vault.abs_path = _real_resolve, _real_abs
    _vault.thumb_path = _real_thumb

print("")
print("H3 Story Planner - one brief, the whole timeline")
from h3_planner import nodes_story

STORY = {"n": 0, "beats": 0, "prose": 0, "batches": []}


def story_generate(cfg, system, user, schema, required_keys=(), min_words=0,
                   images=None):
    STORY["n"] += 1
    if "You are the director" in system:
        STORY["beats"] += 1
        return ({
            "style_prefix": "Cinematic 35mm anamorphic, warm grade",
            "world": "A rooftop in Delhi at golden hour.",
            "subject_definitions": "<Subject 1> a young man in a black jacket.",
            "segments": [
                {"seconds": 8.0, "scene": 1, "scene_name": "rooftop",
                 "action": "he crosses to the ledge",
                 "dialogue": "This city never asked me for permission.",
                 "opens_from": "by the door", "ends_with": "hands on the rail"},
                {"seconds": 8.0, "scene": 1, "scene_name": "rooftop",
                 "action": "he looks out", "opens_from": "ignored",
                 "ends_with": "turning back, sun behind him"},
                {"seconds": 8.0, "scene": 2, "scene_name": "stairwell",
                 "action": "he descends", "opens_from": "top of the stairs",
                 "ends_with": "pushing the door open"},
            ],
        }, "stub")
    STORY["prose"] += 1
    count = user.count("--- SEGMENT ")
    STORY["batches"].append(count)
    # Distinct per segment, and only the segment that was GIVEN a line carries
    # one. A stub that repeated segment 1's dialogue in every object is the
    # exact bleed the planner now detects and rewrites.
    beats = [
        ("<Subject 1> (S1) says, <d>[English] This city never asked me for "
         "permission.</d> [Shot 1] a wide shot, the camera pushes in toward "
         "<Subject 1> as he crosses the roof. Background noise includes "
         "distant traffic and chatter."),
        ("[Shot 1] a close shot, the camera tilts up the ledge to find "
         "<Subject 1> looking out over the rooftops, hands loose at his "
         "sides while the light drops behind the water tanks."),
        ("[Shot 1] a low-angle tracking shot following <Subject 1> down the "
         "stairwell, shoulder leading, the door swinging shut above him and "
         "the stripe of daylight narrowing to nothing."),
        ("[Shot 1] a handheld shot arcing right around <Subject 1> at the "
         "parapet, the skyline sliding behind his shoulder as he turns his "
         "head slowly to follow something out of frame."),
    ]
    out = []
    for i in range(count):
        obj, _ = fake_generate(cfg, system, user, schema)
        obj = dict(obj)
        obj["detailed_description"] = "".join(beats[i % len(beats)])
        out.append(obj)
    return ({"segments": out}, "stub")


engine.generate = story_generate
sp = project("story")
timeline, sheet, report = nodes_story.H3PlannerStoryPlanner().plan(
    sp, "A rapper crosses a Delhi rooftop at golden hour.", 24.0, 5.0, 10.0,
    "cinematic ad", "spoken lines", "performed on camera", 4, 8, False,
    "Ollama (Local)",
    "http://x", "m", 0.3,
    False, 0, cast=CAST)

ok("it returns a timeline", isinstance(timeline, dict))
check("three segments were planned", len(timeline["segments"]), 3)
check("the beat sheet was one call", STORY["beats"], 1)
check("and the prose was one more", STORY["prose"], 1)
check("all three written in a single batch", STORY["batches"], [3])
ok("that is 2 calls, not 1 per segment", STORY["n"] == 2, str(STORY["n"]))

check("scenes came through", [s["scene"] for s in timeline["segments"]],
      [1, 1, 2])
check("the handover was threaded", timeline["segments"][1]["opens_from"],
      "hands on the rail")
check("chaining stayed off by default",
      [s["link"] for s in timeline["segments"]], ["cut", "cut", "cut"])
ok("every segment got a full prompt",
   all(isinstance(s["prompt"], dict) and
       set(s["prompt"]) == set(nodes_prompt.SECTIONS)
       for s in timeline["segments"]))
ok("durations are on the ladder",
   all(abs(s["render_duration"] - s["target_duration"]) < 1e-6
       for s in timeline["segments"]),
   str([(s["target_duration"], s["render_duration"])
        for s in timeline["segments"]]))

flat = nodes_plan.flatten_prompt(timeline["segments"][0]["prompt"])
ok("the invented <Video 1> was stripped", "<Video 1>" not in flat, flat[:160])
ok("the file name was stripped", "my_track_02" not in flat)
ok("the audio is declared exact", "used exactly as supplied" in flat)
ok("the style prefix is in the description",
   "Cinematic 35mm anamorphic" in timeline["segments"][0]["prompt"]
   ["detailed_description"])
ok("the cast wording is shared verbatim",
   len({s["prompt"]["subject_definitions"] for s in timeline["segments"]}) == 1)

ok("the beat sheet names the scenes", "SCENE 1" in sheet and "SCENE 2" in sheet)
ok("and shows the handover", "hands on the rail" in sheet)
ok("the report counts the calls", "2 model call(s)" in report, report[:200])
ok("the timeline was persisted",
   store.load(sp["timeline_path"]) is not None)

print("")
print("H3 Story Planner - a long piece is windowed")
STORY["beats"] = STORY["prose"] = STORY["n"] = 0
STORY["batches"] = []


def many_beats(cfg, system, user, schema, required_keys=(), min_words=0,
               images=None):
    if "You are the director" in system:
        STORY["beats"] += 1
        STORY["n"] += 1
        return ({"style_prefix": "P", "world": "W",
                 "subject_definitions": "<Subject 1> a man.",
                 "segments": [{"seconds": 8.0, "scene": 1 + i // 3,
                               "action": "beat %d" % i,
                               "ends_with": "state %d" % i}
                              for i in range(9)]}, "stub")
    return story_generate(cfg, system, user, schema)


engine.generate = many_beats
lp = project("story_long")
long_tl, _, long_report = nodes_story.H3PlannerStoryPlanner().plan(
    lp, "A longer piece.", 72.0, 5.0, 10.0, "short film",
    "none", "background only", 4, 8, True,
    "Ollama (Local)", "http://x", "m", 0.3, False, 0, cast=CAST)
check("nine segments", len(long_tl["segments"]), 9)
check("beats stayed one call", STORY["beats"], 1)
check("prose was windowed in fours", STORY["batches"], [4, 4, 1])
ok("that is 4 calls for 9 segments, not 9", STORY["n"] == 4,
   str((STORY["n"], STORY["prose"])))
ok("chaining was requested and applied inside scenes",
   any(s["link"] == "continue" for s in long_tl["segments"]),
   str([s["link"] for s in long_tl["segments"]]))
ok("but never across a scene change",
   all(not (s["link"] == "continue" and s["scene"] != p["scene"])
       for p, s in zip(long_tl["segments"], long_tl["segments"][1:])))

engine.generate = fake_generate


print("")
print("H3 Story Planner - the spoken script")
seg1 = timeline["segments"][0]
check("the planned line is kept on the segment", seg1["dialogue"],
      "This city never asked me for permission.")
body = seg1["prompt"]["detailed_description"]
ok("it reached the prompt as H3 dialogue", "<d>" in body, body[:120])
ok("with the language tag", "[English]" in body)
ok("and a speaker id", "(S1)" in body)
ok("the beat sheet prints the script", "says:" in sheet, sheet[:200])
ok("the report counts it", "dialogue" in report and "speak" in report)

# a line planned but dropped by the model must be reported, not swallowed
planner = nodes_story.H3PlannerStoryPlanner()
ok("a missing <d> is caught",
   not planner._dialogue_landed({"detailed_description": "she walks away."},
                                "Your morning, bottled."))
ok("a present one passes",
   planner._dialogue_landed(
       {"detailed_description": "<d>[English] Your morning, bottled.</d>"},
       "Your morning, bottled."))
ok("no line planned is never a failure",
   planner._dialogue_landed({"detailed_description": "she walks."}, ""))

# "none" must actually suppress the line even when the model writes one
ok("dialogue=none strips the planned line",
   all(not x.get("dialogue") for x in long_tl["segments"]),
   str([x.get("dialogue") for x in long_tl["segments"]][:3]))

print("")
print("H3 Story Planner - a second queue costs nothing")
STORY["n"] = STORY["beats"] = STORY["prose"] = 0
STORY["batches"] = []
engine.generate = story_generate
again_tl, again_sheet, again_report = nodes_story.H3PlannerStoryPlanner().plan(
    sp, "A rapper crosses a Delhi rooftop at golden hour.", 24.0, 5.0, 10.0,
    "cinematic ad", "spoken lines", "performed on camera", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)
check("re-running made no model calls at all", STORY["n"], 0)
check("and returned the same segments", len(again_tl["segments"]), 3)
ok("every prompt is still there",
   all(isinstance(x["prompt"], dict) for x in again_tl["segments"]))
ok("the report says why it did nothing", "unchanged" in again_report,
   again_report[:90])
ok("the beat sheet still renders", "SCENE" in again_sheet)

# changing an input must plan again
STORY["n"] = 0
nodes_story.H3PlannerStoryPlanner().plan(
    sp, "A completely different brief about a kitchen.", 24.0, 5.0, 10.0,
    "cinematic ad", "spoken lines", "performed on camera", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)
ok("a changed brief replans", STORY["n"] > 0, str(STORY["n"]))

# and reuse_existing off always replans
STORY["n"] = 0
nodes_story.H3PlannerStoryPlanner().plan(
    sp, "A completely different brief about a kitchen.", 24.0, 5.0, 10.0,
    "cinematic ad", "spoken lines", "performed on camera", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, False, 0, cast=CAST)
ok("reuse_existing off forces a replan", STORY["n"] > 0, str(STORY["n"]))

print("")
print("H3 Story Planner - a locked hand-written prompt is never touched")
lock_p = project("story_lock")
engine.generate = story_generate
lk1, _, _ = nodes_story.H3PlannerStoryPlanner().plan(
    lock_p, "the first brief", 24.0, 5.0, 10.0, "cinematic ad",
    "spoken lines", "background only", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)

MINE = {k: "MY OWN HAND WRITTEN PROMPT" for k in nodes_prompt.SECTIONS}
saved = store.load(lock_p["timeline_path"])
saved["segments"][0]["prompt"] = dict(MINE)
saved["segments"][0]["state"] = store.LOCKED
store.save(lock_p["timeline_path"], saved)

# changing the brief must replan the rest and leave the locked one alone
STORY["n"] = 0
lk2, _, lock_rep = nodes_story.H3PlannerStoryPlanner().plan(
    lock_p, "a COMPLETELY different brief", 24.0, 5.0, 10.0, "cinematic ad",
    "spoken lines", "background only", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)
first = lk2["segments"][0]
check("the locked segment stays locked", first["state"], store.LOCKED)
ok("and is not blanked", bool(first.get("prompt")), repr(first.get("prompt"))[:60])
check("it keeps the hand-written text", first["prompt"]["summary"],
      "MY OWN HAND WRITTEN PROMPT")
ok("while the others were replanned", STORY["n"] > 0, str(STORY["n"]))
ok("and they are not the hand-written text",
   lk2["segments"][1]["prompt"]["summary"] != "MY OWN HAND WRITTEN PROMPT")

# and re-queueing with nothing changed still costs no calls
STORY["n"] = 0
nodes_story.H3PlannerStoryPlanner().plan(
    lock_p, "a COMPLETELY different brief", 24.0, 5.0, 10.0, "cinematic ad",
    "spoken lines", "background only", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)
check("a lock does not force a replan every queue", STORY["n"], 0)

print("")
print("H3 Story Planner - nothing left to write is not a crash")
empty_p = project("story_all_locked")
engine.generate = story_generate
ep1, _, _ = nodes_story.H3PlannerStoryPlanner().plan(
    empty_p, "a brief to lock", 24.0, 5.0, 10.0, "cinematic ad",
    "none", "background only", 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, True, 0, cast=CAST)
locked_all = store.load(empty_p["timeline_path"])
for _s in locked_all["segments"]:
    _s["state"] = store.LOCKED
store.save(empty_p["timeline_path"], locked_all)

# reuse_existing OFF with every segment locked left `todo` empty, which made
# the window size zero and raised "range() arg 3 must not be zero".
try:
    ep2, _, ep_rep = nodes_story.H3PlannerStoryPlanner().plan(
        empty_p, "a brief to lock", 24.0, 5.0, 10.0, "cinematic ad",
        "none", "background only", 4, 8, False,
        "Ollama (Local)", "http://x", "m", 0.3, False, 0, cast=CAST)
    ok("every segment locked does not raise", True)
    ok("the timeline still comes back whole",
       all(x.get("prompt") for x in ep2["segments"]),
       str(len(ep2["segments"])))
    ok("and the report explains the no-op",
       "nothing needed writing" in ep_rep, ep_rep[:90])
except Exception as _ex:
    ok("every segment locked does not raise", False, repr(_ex))

# a window of zero from a wired input must not raise either
try:
    nodes_story.H3PlannerStoryPlanner().plan(
        project("story_zero_window"), "another brief", 24.0, 5.0, 10.0,
        "cinematic ad", "none", "background only", 0, 0, False,
        "Ollama (Local)", "http://x", "m", 0.3, False, 0, cast=CAST)
    ok("a zero window does not raise", True)
except Exception as _ex:
    ok("a zero window does not raise", False, repr(_ex))

print("")
print("H3 Story Planner - a short reply leaves no segment unwritten")
SHORT = {"n": 0, "sizes": []}


def stingy_generate(cfg, system, user, schema, required_keys=(), min_words=0,
                    images=None):
    """Returns ONE object however many segments were asked for."""
    if "You are the director" in system:
        return ({"style_prefix": "Cinematic 35mm", "world": "A beach.",
                 "subject_definitions": "<Subject 1> a woman.",
                 "segments": [{"seconds": 8.0, "scene": 1,
                               "action": "beat %d" % i,
                               "ends_with": "e%d" % i, "dialogue": ""}
                              for i in range(4)]}, "stub")
    SHORT["n"] += 1
    SHORT["sizes"].append(user.count("--- SEGMENT "))
    body = ("[Shot 1] a distinct beat %d, the light dropping behind her "
            "shoulder as she crosses the sand toward the water."
            % SHORT["n"])
    return ({"segments": [{k: ("".join(body)
                               if k == "detailed_description" else "x")
                           for k in nodes_prompt.SECTIONS}]}, "stub")


engine.generate = stingy_generate
short_tl, _, short_rep = nodes_story.H3PlannerStoryPlanner().plan(
    project("story_short"), "a perfume ad", 32.0, 5.0, 10.0, "cinematic ad",
    "none", "background only", 4, 8, False, "Ollama (Local)", "http://x",
    "m", 0.3, False, 0, cast=CAST)

check("four segments were planned", len(short_tl["segments"]), 4)
ok("every one ends up with a prompt",
   all(isinstance(x.get("prompt"), dict) for x in short_tl["segments"]),
   str([x["id"] for x in short_tl["segments"] if not x.get("prompt")]))
ok("the missing ones were written singly",
   SHORT["sizes"][0] == 4 and SHORT["sizes"][1:] == [1, 1, 1],
   str(SHORT["sizes"]))
ok("the report says the reply was short",
   "returned 1 of 4" in short_rep, short_rep[:100])
ok("and no segment is reported as unwritten",
   "NO prompt" not in short_rep)

# an impossible length window must say so instead of quietly using a fallback
engine.generate = story_generate
_, _, win_rep = nodes_story.H3PlannerStoryPlanner().plan(
    project("story_window"), "a brief", 15.0, 5.0, 5.0, "cinematic ad",
    "none", "background only", 4, 8, False, "Ollama (Local)", "http://x",
    "m", 0.3, False, 0, cast=CAST)
ok("an impossible clip-length window is reported",
   "no clip length exists" in win_rep, win_rep[:120])
ok("and it names the real lengths nearby",
   "4.458s" in win_rep or "5.167s" in win_rep, win_rep[:200])


# --------------------------------------------------------------------------
print("")
print("H3 Story Planner - two voice samples, two people talking")

# Their real brief: a cafe two-hander where <Audio 1> is the MALE voice and
# the male character is <Subject 2>, so pairing the tags in order is wrong.
DUO = {
    "members": [
        {"tag": "<Subject 1>", "kind": "Subject", "slot": 1, "key": "she",
         "role": "character", "note": "", "file": "she_01.png"},
        {"tag": "<Subject 2>", "kind": "Subject", "slot": 2, "key": "he",
         "role": "character", "note": "", "file": "he_01.png"},
        {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "key": "male_vo",
         "role": "audio", "note": "male voice", "file": "male_01.mp3"},
        {"tag": "<Audio 2>", "kind": "Audio", "slot": 0, "key": "female_vo",
         "role": "audio", "note": "female voice", "file": "female_01.mp3"},
    ],
    "size": 2, "has_audio": True, "audio_tag": "<Audio 1>",
    "counts": {"Subject": 2, "Picture": 0, "Video": 0, "Audio": 2},
}

VOICE = {"beats_system": "", "prose_system": "", "beats_user": ""}


def voice_generate(cfg, system, user, schema, required_keys=(), min_words=0,
                   images=None):
    if "You are the director" in system:
        VOICE["beats_system"] = system
        VOICE["beats_user"] = user
        return ({
            "style_prefix": "Cinematic 35mm, warm daylight grade",
            "world": "A corner cafe in late afternoon light.",
            "subject_definitions": "<Subject 1> platinum ponytails. "
                                   "<Subject 2> dark hair, mustache.",
            # Misbehaving the way a real reply does: one pair names a subject
            # that does not exist, and one segment names a speaker nobody
            # cast. Neither may reach the prompt.
            "voices": [{"audio": "<Audio 1>", "subject": "<Subject 2>"},
                       {"audio": "<Audio 2>", "subject": "<Subject 7>"}],
            "segments": [
                {"seconds": 5.875, "scene": 1, "scene_name": "cafe",
                 "action": "she comes through the door",
                 "dialogue": "You're late.", "speaker": "<Subject 1>",
                 "opens_from": "street", "ends_with": "at the counter"},
                {"seconds": 5.875, "scene": 1, "scene_name": "cafe",
                 "action": "he looks up from the table",
                 "dialogue": "The bus never came.", "speaker": "<Subject 2>",
                 "opens_from": "at the counter", "ends_with": "he stands"},
                {"seconds": 5.875, "scene": 1, "scene_name": "cafe",
                 "action": "she sets the cup down",
                 "dialogue": "It never does.", "speaker": "<Subject 9>",
                 "opens_from": "he stands", "ends_with": "both seated"},
            ],
        }, "stub")

    VOICE["prose_system"] = system
    count = user.count("--- SEGMENT ")
    out = []
    for i in range(count):
        obj, _ = fake_generate(cfg, system, user, schema)
        obj = dict(obj)
        obj["detailed_description"] = (
            "[Shot 1] Cinematic 35mm, warm daylight grade, <Subject 1> (S1) "
            "pushes the door open and says, <d>[English] You're late.</d> "
            "her mouth matching the words.")
        # The two failures this feature exists to stop, written in by hand:
        # a soundscape that forbids generated speech, and fully_copy.
        obj["overall_soundscape"] = (
            "All audio in this segment comes from <Audio 1> and is used "
            "exactly as supplied. No additional sound is present.")
        obj["non_diegetic_music"] = "N/A - the audio is supplied by <Audio 1>."
        obj["retention_analysis"] = (
            "<Subject 1>: fully_preserved. <Audio 1>: fully_copy, all shots.")
        out.append(obj)
    return ({"segments": out}, "stub")


engine.generate = voice_generate
vp = project("story_voices")
v_timeline, v_sheet, v_report = nodes_story.H3PlannerStoryPlanner().plan(
    vp, "A cafe two-hander. Use <audio 1> for the male and <audio 2> for the "
        "female character.", 18.0, 5.0, 6.0,
    "cinematic ad", "spoken lines", nodes_story.VOICE_SAMPLE, 4, 8, False,
    "Ollama (Local)", "http://x", "m", 0.3, False, 0, cast=DUO)

seg = v_timeline["segments"][0]
sound = seg["prompt"]["overall_soundscape"]
ok("the copy-the-track soundscape is gone",
   "used exactly as supplied" not in sound and "No additional sound" not in sound,
   sound[:80])
ok("and it says the dialogue is spoken on camera",
   "spoken on camera" in sound, sound[:80])
ok("both voices are named in it",
   "<Audio 1>" in sound and "<Audio 2>" in sound, sound[-120:])
ok("the music section no longer defers to the track",
   seg["prompt"]["non_diegetic_music"].startswith("N/A - there is no scored"),
   seg["prompt"]["non_diegetic_music"])

ret = seg["prompt"]["retention_analysis"]
ok("fully_copy is replaced by reference", "fully_copy" not in ret, ret)
ok("every audio tag gets a retention line, not just the first",
   "<Audio 1>: reference" in ret and "<Audio 2>: reference" in ret, ret)
ok("the subject definitions still survive", "fully_preserved" in ret, ret)

# The casting the director chose, checked against the cast that exists.
ok("the brief's casting is honoured over pairing in order",
   "<Audio 1> supplies the voice of <Subject 2>" in VOICE["prose_system"],
   VOICE["prose_system"][:200])
ok("a pair naming a subject that does not exist is repaired",
   "<Subject 7>" not in VOICE["prose_system"]
   and "<Audio 2> supplies the voice of <Subject 1>" in VOICE["prose_system"])
ok("the two speakers get different ids",
   "<Subject 2>, who speaks as (S2)" in VOICE["prose_system"]
   and "<Subject 1>, who speaks as (S1)" in VOICE["prose_system"],
   VOICE["prose_system"][:260])

ok("the director is told the audio is a voice sample, not a soundtrack",
   "VOICE SAMPLES" in VOICE["beats_system"]
   and "used exactly as given" not in VOICE["beats_system"])
ok("and the beat sheet is briefed as a conversation",
   "THIS IS A CONVERSATION" in VOICE["beats_user"], VOICE["beats_user"][:80])
ok("the silence rules for a voice-over ad are not applied to it",
   "The final segment is silent" not in VOICE["beats_user"])

check("who speaks is carried on the segment",
      [s.get("speaker") for s in v_timeline["segments"]],
      ["<Subject 1>", "<Subject 2>", ""])
ok("a speaker nobody cast is dropped rather than trusted",
   "<Subject 9>" not in str(v_timeline["segments"]))
ok("the report shows the casting and the line count",
   "<Audio 1> = <Subject 2> (S2)" in v_report, [
       l for l in v_report.splitlines() if l.startswith("voices")])

# and the same planner on the same cast with the old role is untouched
labelled = nodes_story.H3PlannerStoryPlanner._speaker_label(
    {"speaker": "<Subject 2>"},
    nodes_story.cast_voices(DUO, [{"audio": "<Audio 1>",
                                   "subject": "<Subject 2>"}]))
check("the prose call names the right speaker for a line", labelled,
      "<Subject 2> (S2)")
check("a line with no speaker falls back to the first cast voice",
      nodes_story.H3PlannerStoryPlanner._speaker_label(
          {}, nodes_story.cast_voices(DUO)), "<Subject 1> (S1)")

idle = nodes_story.H3PlannerStoryPlanner._voices_used(
    [{"speaker": "<Subject 1>"}, {"speaker": "<Subject 1>"}],
    nodes_story.cast_voices(DUO))
ok("a cast voice that never speaks is reported", len(idle) == 1
   and "<Subject 2>" in idle[0], str(idle))
check("and nothing is reported when both speak",
      nodes_story.H3PlannerStoryPlanner._voices_used(
          [{"speaker": "<Subject 1>"}, {"speaker": "<Subject 2>"}],
          nodes_story.cast_voices(DUO)), [])


# --------------------------------------------------------------------------
print("")
print("H3 Timeline - a refined prompt survives the planner re-running")
import json
# The user's exact loop: a planner wired into the Timeline, a segment
# rendered, the card refined, the graph queued again. The planner handed back
# its original prompt, the refine was overwritten, and the same shot rendered
# again. Adversarial the way the real pipeline is: the planner also saves its
# own prompts over the timeline on disk BEFORE the Timeline node runs.
TL = nodes_plan.H3PlannerTimeline()
kp = project("keep_edits")


def planned(p2="[Shot 1] a wide of the cafe as she walks in."):
    return {"name": kp["name"], "context": {},
            "segments": [
                {"id": "seg_01", "target_duration": 5.0,
                 "prompt": "[Shot 1] he looks up from the table."},
                {"id": "seg_02", "target_duration": 5.0, "prompt": p2}]}


def queue(upstream, widget):
    store.save(kp["timeline_path"], store.merge(
        store.load(kp["timeline_path"]),
        store.normalize_timeline(upstream, kp)))           # the planner's save
    out = TL.build(kp, "inline", widget, "merge", True, timeline=upstream)
    return out, out["ui"]["h3_timeline"][0]["authored"]


first, widget = queue(planned(), "{}")
doc = json.loads(widget)
ok("the planner's delivery is recorded on each card",
   all(s.get("planned_prompt_hash") for s in doc["segments"]))

# seg_02 renders
disk = store.load(kp["timeline_path"])
s2 = store.find(disk, "seg_02")
s2["clips"] = {"draft": {"file": "old.mp4", "pass": "draft", "take": 0}}
s2["state"] = "draft"
store.save(kp["timeline_path"], disk)

# refine seg_02: what the route does on disk, and what the card does in the
# widget
REFINED = "[Shot 1] a LOW ANGLE of her walking away from camera."
disk = store.load(kp["timeline_path"])
s2 = store.find(disk, "seg_02")
s2["prompt"] = REFINED
s2["refine_note"] = "make it a low angle, she walks away"
s2["spec_hash"] = store.spec_hash(s2)
s2["clips"] = {}
s2["state"] = store.PENDING
store.save(kp["timeline_path"], disk)
for s in doc["segments"]:
    if s["id"] == "seg_02":
        s["prompt"] = REFINED
        s["refine_note"] = "make it a low angle, she walks away"
widget = json.dumps(doc)

second, widget = queue(planned(), widget)
live = store.load(kp["timeline_path"])
check("the refined prompt stays on the timeline",
      store.find(live, "seg_02")["prompt"], REFINED)
check("and in the card strip's JSON",
      [s["prompt"] for s in json.loads(widget)["segments"]
       if s["id"] == "seg_02"][0], REFINED)
ok("the segment renders again rather than reusing the old clip",
   store.find(live, "seg_02")["state"] == store.PENDING
   and not store.find(live, "seg_02").get("clips"))
ok("the report says the edit was kept",
   "kept your edited prompt on seg_02" in second["result"][1],
   second["result"][1][:160])
check("an untouched segment still takes the planner's prompt",
      store.find(live, "seg_01")["prompt"], "[Shot 1] he looks up from the table.")

third, widget = queue(planned(), widget)
check("it is still there on the queue after that",
      store.find(store.load(kp["timeline_path"]), "seg_02")["prompt"], REFINED)

# the planner genuinely re-writes seg_02: its new prompt wins, loudly
NEW = "[Shot 1] a new plan: she sits down opposite him."
fourth, widget = queue(planned(NEW), widget)
back = store.find(store.load(kp["timeline_path"]), "seg_02")
check("a real re-plan of that segment replaces the edit", back["prompt"], NEW)
ok("and the report names the lost edit",
   "REPLACED your edited prompt on seg_02" in fourth["result"][1])
check("the stale refine note goes with it", back.get("refine_note"), "")

# a hand edit on the card, no refine note, is kept the same way
doc = json.loads(widget)
for s in doc["segments"]:
    if s["id"] == "seg_01":
        s["prompt"] = "[Shot 1] HAND EDIT: he stands to greet her."
fifth, widget = queue(planned(NEW), json.dumps(doc))
check("a hand edit on the card is kept too",
      store.find(store.load(kp["timeline_path"]), "seg_01")["prompt"],
      "[Shot 1] HAND EDIT: he stands to greet her.")

# a card from before the planned prompt was recorded
old = {"name": kp["name"], "segments": [
    {"id": "seg_01", "prompt": "[Shot 1] REFINED EARLIER.",
     "refine_note": "earlier note"},
    {"id": "seg_02", "prompt": "[Shot 1] stale copy, never refined."}]}
sixth, _ = queue(planned(NEW), json.dumps(old))
after = store.load(kp["timeline_path"])
check("an older card that was refined keeps its refine",
      store.find(after, "seg_01")["prompt"], "[Shot 1] REFINED EARLIER.")
check("an older card that was never refined takes the planner's prompt",
      store.find(after, "seg_02")["prompt"], NEW)

# a card strip that belongs to another project leaks nothing across
foreign = {"name": "some_other_project", "segments": [
    {"id": "seg_02", "prompt": "[Shot 1] FROM ANOTHER PROJECT.",
     "refine_note": "x"}]}
seventh, _ = queue(planned(NEW), json.dumps(foreign))
check("edits from another project's card strip are ignored",
      store.find(store.load(kp["timeline_path"]), "seg_02")["prompt"], NEW)

# --------------------------------------------------------------------------
print("")
print("H3 Segment Prompter - an idea is planned once, and every segment is complete")
# The user's run: an 86-second music video from an idea alone, a stage picture
# and a character sheet. Each segment was written blind and on its own: the
# looks were invented, <Subject 1> was the singer in some segments and the
# guitarist in others, only segment 1's definitions reached the rest, and six
# summaries came out "N/A". The stub misbehaves the way that run did.
IDEA = ("<Picture 1> is the main anchor image where the band is performing, "
        "use <Picture 2> as character sheets of band characters. <Audio 1> is "
        "the audio. first 28 seconds have only instrumental (no lip sync). "
        "then the lyrics comes. the main lead singer is singing this and "
        "others are playing their instruments. dont use <Picture 2> in the "
        "video.")
BAND = {
    "members": [
        {"tag": "<Picture 1>", "kind": "Picture", "slot": 1, "number": 1,
         "key": "main_stage", "role": "composition", "note": "",
         "file": "main_stage.png"},
        {"tag": "<Picture 2>", "kind": "Picture", "slot": 2, "number": 2,
         "key": "character sheet", "role": "character", "note": "",
         "file": "sheet_0916.png"},
        {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "number": 1,
         "key": "wired_audio_1", "role": "audio", "note": "", "file": ""},
    ],
    "size": 2, "has_audio": True, "audio_tag": "<Audio 1>",
    "counts": {"Subject": 0, "Picture": 2, "Video": 0, "Audio": 1},
}
PLAN_CAST = [
    {"tag": "<Subject 1>", "who": "lead singer", "from": "<Picture 2>",
     "looks": "long dark messy hair. A scar on the left cheek. Black jacket "
              "with chains"},
    {"tag": "Subject 2", "who": "guitarist", "from": "<Picture 2>",
     "looks": "short dark hair, black sleeveless top, red guitar"},
    {"tag": "<Subject 3>", "who": "drummer", "from": "<Picture 9>",
     "looks": "shaved head, tattooed arms"},
    {"tag": "<Subject 4>", "who": "bassist", "from": "<Picture 2>",
     "looks": "long braids, leather vest"},
]
IP = {"plan_calls": [], "seg_calls": [], "plan_mode": "short_then_full",
      "sections": [], "shots": 2}


def plan_reply(count):
    return {
        "cast": PLAN_CAST,
        "segments": [{"index": n + 1,
                      "action": "PLANNED ACTION %d: the camera finds the band "
                                "from a new angle." % (n + 1),
                      "featured": ["<Subject %d>" % (n % 4 + 1), "<Subject 8>"],
                      "vocals": "instrumental" if n < 3 else "sung"}
                     for n in range(count)],
        "performance": "sings",
        "performer": "",                 # left empty, as qwen did
        "vocal_sections": IP["sections"],
    }


def idea_generate(cfg, system, user, schema, required_keys=(), min_words=0,
                  images=None):
    if system == nodes_prompt.PLAN_SYSTEM:
        IP["plan_calls"].append({"user": user, "images": list(images or []),
                                 "max": cfg.max_output_tokens})
        if IP["plan_mode"] == "raise":
            raise RuntimeError("connection refused")
        if IP["plan_mode"] == "short_then_full" and len(IP["plan_calls"]) == 1:
            return plan_reply(7), "stub"            # 7 of the 11 asked for
        return plan_reply(11), "stub"
    IP["seg_calls"].append({"user": user, "images": list(images or []),
                            "system": system,
                            "keys": sorted((schema or {}).get("properties", {}))
                            if isinstance(schema, dict) else []})
    n = len(IP["seg_calls"])
    if "LEFT summary" in user:                      # the section retry
        return ({"summary": ""} if n % 2 else
                {"summary": "[reference generation] The band plays on, "
                            "call %d." % n}), "stub"
    return {
        # its own, different wording, and only for one subject
        # ... citing the character sheet first, as the real model did
        "subject_definitions": "<Subject 1> is a man with short hair and a "
                               "leather jacket, from <Picture 2>.",
        # the whole summary names subjects this segment does not define;
        # every third call leaves it empty
        "summary": "" if n % 3 == 0 else (
            "%s The band <Subject 1>, <Subject 2>, <Subject 3> and "
            "<Subject 4> perform, call %d."
            % ("[video continuation]" if n % 4 == 1 else
               "[reference generation]", n)),
        "retention_analysis": "<Subject 1> (appears in [Shot 1], [Shot 7]): "
                              "fully_preserved - identity.<Picture 2> ([Shot "
                              "1] first frame): fully_preserved - composition "
                              "anchor for the stage.",
        "detailed_description": (
            "[Shot 1] <Subject 3> hits the drums hard under a red light, "
            "call %d. The scene opens on <Picture 2>, a wide of the main "
            "stage. [Shot 2] At 00:03.000, <Subject 7> waves at the camera "
            "while <Subject 2> plays a riff. <Subject 1> from <Picture 2> "
            "raps into the mic, his mouth matching the words. <Subject 2> "
            "(S2) turns and says, <d>'[lyrics]'</d>, head banging. <Subject 1> "
            "(S1) roars <d>[English] Burn it all down</d>. <Subject 3> pounds "
            "the kit, mouth matching the words as he sings. %s %s"
            % (n, " ".join("[Shot %d] At 00:0%d.500, a cut to the crowd."
                           % (k, k) for k in range(3, IP["shots"] + 1)),
               " ".join("Beat %d.%d: the %s camera %s past %s." % (
                n, k, ["handheld", "crane", "dolly", "static", "whip"][(n + k) % 5],
                ["drifts", "pushes", "arcs", "tilts", "tracks"][(n * k) % 5],
                ["the amps", "the kit", "the crowd", "the lights"][(n + 2 * k) % 4])
                for k in range(14)))),
        "overall_soundscape": "Crowd noise.",
        "non_diegetic_music": "<Audio 1> drives it.",
    }, "stub"


engine.generate = idea_generate
_real_image_b64 = engine.image_b64
engine.image_b64 = lambda path: "IMG:" + os.path.basename(path)


def idea_project(name):
    return nodes_plan.H3PlannerProject().build(
        name, 24.0, "16:9", "draft", 0.6, 1.2, 32, 12345, 17, 5, 5, "up",
        10.0, 6.0, 10)[0]


def run_idea(proj, reuse=False):
    return nodes_prompt.H3PlannerSegmentPrompter().write(
        proj, "", 86.0, 8.0, 2, "music video", "performed on camera",
        "Ollama (Local)", "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25,
        reuse, 371069875, cast=BAND, idea=IDEA, max_output_tokens=3072)


ip = idea_project("idea_plan")
tl, rep = run_idea(ip)
segs = tl["segments"]
check("the 86-second idea is 11 segments", len(segs), 11)

check("the plan is asked for twice when it comes back short",
      len(IP["plan_calls"]), 2)
ok("and the retry says why", "7 segment(s), 11 were asked for"
   in IP["plan_calls"][1]["user"])
check("the planner sees both pictures", IP["plan_calls"][0]["images"],
      ["IMG:main_stage.png", "IMG:sheet_0916.png"])
ok("so does every segment writer",
   all(c["images"] == ["IMG:main_stage.png", "IMG:sheet_0916.png"]
       for c in IP["seg_calls"]))
ok("the plan gets room for its long reply",
   IP["plan_calls"][0]["max"] >= 4096)
ok("the planner is told each picture's job",
   "<Picture 2> (character sheet) = IDENTITY reference"
   in IP["plan_calls"][0]["user"])

canon = tl["context"]["subject_definitions"]
check("the plan's cast becomes the shared wording, one line per subject",
      len(nodes_prompt.subject_lines(canon)), 4)
ok("with the looks read from the picture, full stops and all",
   "<Subject 1> is lead singer, from <Picture 2>: long dark messy hair; A scar"
   " on the left cheek; Black jacket with chains." in canon, canon[:200])
ok("a picture the cast does not have is not credited",
   "<Picture 9>" not in canon and "<Subject 3> is drummer:" in canon)
ok("and the report says so",
   "<Subject 3> was said to come from <Picture 9>" in rep)
ok("the plan is kept on the timeline for the next queue",
   tl["context"]["plan"]["plan"]["beats"][10]["action"].startswith(
       "PLANNED ACTION 11"))
ok("a featured subject the plan never defined is dropped",
   all(8 not in b["featured"] for b in tl["context"]["plan"]["plan"]["beats"]))
ok("the backend is saved for refine at last", tl["context"].get("backend"))

third = [c["user"] for c in IP["seg_calls"]
         if "SEGMENT 3 OF 11" in c["user"]][0]
ok("segment 3 is told where it sits in the track",
   "15.64s to 23.45s of the whole 86.00s video" in third, third[:300])
ok("and what the plan says happens in it",
   "PLANNED ACTION 3:" in third and "PLANNED ACTION 2:" in third
   and "PLANNED ACTION 4:" in third)
ok("and who is on screen", "<Subject 3> (drummer)" in third)
ok("and which verb the performer takes, and who that is",
   "<Subject 1> (lead singer) sings" in third
   and "Nobody else's mouth moves to the words" in third)
ok("the singer is found from the cast when the plan leaves performer empty",
   tl["context"]["plan"]["plan"]["performer"] == 1)
ok("and the soundscape names the singer, not 'the subject on screen'",
   all("performed on camera by <Subject 1>" in s["prompt"]["overall_soundscape"]
       for s in segs if "performed on camera" in s["prompt"]["overall_soundscape"]))
ok("a spoken line with no words in it is removed",
   all("[lyrics]" not in json.dumps(s["prompt"]) for s in segs))
ok("the guitarist it was given to does not keep a speaker id",
   all("<Subject 2> (S2)" not in s["prompt"]["detailed_description"] for s in segs))
ok("a real line is kept where the song is sung",
   any("<d>[English] Burn it all down</d>" in s["prompt"]["detailed_description"]
       for s in segs[4:]))
ok("only the performer sings: the drummer's singing is taken away",
   all("<Subject 3> pounds the kit." in s["prompt"]["detailed_description"]
       for s in segs))
ok("and the guitarist given an empty line keeps playing with his mouth shut",
   all("<Subject 2> keeps playing, mouth closed." in s["prompt"]["detailed_description"]
       for s in segs[4:]))
check("a summary claims no keyframe task without a frame picture",
      nodes_prompt.fix_summary_type("[keyframe completion] The band plays.",
                                    BAND, "performed on camera"),
      "[reference generation + audio reuse] The band plays.")
ok("a summary never claims a video task with no video connected",
   all("[video continuation]" not in s["prompt"]["summary"] for s in segs)
   and all(s["prompt"]["summary"].startswith("[") for s in segs))
ok("glued retention lines are separated",
   all(".<" not in s["prompt"]["retention_analysis"] for s in segs))
ok("and the cast numbers are fixed", "never swap them" in third)

problems = []
for s in segs:
    p = s["prompt"]
    defined = set(nodes_prompt.defining_lines(p["subject_definitions"]))
    cited = nodes_prompt.cited_numbers(p, "Subject")
    if cited - defined:
        problems.append("%s cites %s undefined" % (s["id"], sorted(cited - defined)))
    for n in cited:
        line = nodes_prompt.defining_lines(canon).get(n)
        if line and line not in p["subject_definitions"]:
            problems.append("%s: <Subject %d> not in the shared wording" % (s["id"], n))
    if "short hair and a leather jacket" in p["subject_definitions"]:
        problems.append("%s kept its own invented looks" % s["id"])
    for key in ("summary", "retention_analysis", "detailed_description"):
        if p[key].strip() in ("", "N/A"):
            problems.append("%s: %s is empty" % (s["id"], key))
    for n in cited:
        if "<Subject %d>" % n not in p["retention_analysis"]:
            problems.append("%s: no retention for <Subject %d>" % (s["id"], n))
    if "<Subject 7>" in json.dumps(p):
        problems.append("%s: the undefined subject survived" % s["id"])
ok("every segment defines every subject it cites, in the shared words, with "
   "a summary and a retention line for each", not problems,
   "; ".join(problems[:6]))
ok("a sentence naming an undefined subject keeps its action",
   all("someone waves at the camera" in s["prompt"]["detailed_description"]
       for s in segs))
ok("a retention line only claims shots the subject is really in",
   all("<Subject 4>: fully_preserved" in s["prompt"]["retention_analysis"]
       for s in segs if "<Subject 4>" in s["prompt"]["summary"]
       and "<Subject 4>" not in s["prompt"]["detailed_description"]))
ok("the whole-band summary survives",
   any("<Subject 3> and <Subject 4> perform" in s["prompt"]["summary"]
       for s in segs))
ok("the undefined subject was asked about before being renamed",
   any("CITED <Subject 7>, WHICH NOTHING DEFINES" in c["user"]
       for c in IP["seg_calls"]))
ok("an empty summary is asked for again",
   any("LEFT summary EMPTY" in c["user"] for c in IP["seg_calls"]))
ok("and written in code when the retry is empty too",
   any(s["prompt"]["summary"].startswith(
       "[reference generation + audio reuse] The target video shows this:")
       for s in segs), [s["prompt"]["summary"][:60] for s in segs])
ok("the report lists the repairs", "REPAIRED" in rep and "summary empty" in rep)
ok("the report shows the plan and the pictures",
   "plan          one plan for the whole video, made for this run: 11 of 11"
   in rep and "shown to the model: <Picture 1>, <Picture 2>" in rep, rep[:600])

# reuse_existing: nothing is asked again, nothing changes
before = json.dumps([s["prompt"] for s in segs], sort_keys=True)
store.save(ip["timeline_path"], tl)
IP["plan_calls"], IP["seg_calls"] = [], []
tl2, rep2 = run_idea(ip, reuse=True)
check("with reuse_existing on, the plan is not asked for again",
      len(IP["plan_calls"]), 0)
check("and no segment is rewritten", len(IP["seg_calls"]), 0)
check("so the prompts are exactly the same",
      json.dumps([s["prompt"] for s in tl2["segments"]], sort_keys=True), before)
ok("the report says the plan was reused", "reused from the saved timeline" in rep2)

# a treatment is unaffected by all of this
IP["plan_calls"], IP["seg_calls"] = [], []
nodes_prompt.H3PlannerSegmentPrompter().write(
    idea_project("idea_plan_treatment"), TREATMENT, 15.0, 10.0, 1, "auto",
    "background only", "Ollama (Local)", "http://127.0.0.1:11434",
    "qwen3-vl:8b", 0.25, False, 0, cast=CAST)
check("a treatment makes no plan call", len(IP["plan_calls"]), 0)
ok("and sends no pictures", all(not c["images"] for c in IP["seg_calls"]))

# the plan call failing does not stop the node
IP["plan_mode"] = "raise"
IP["plan_calls"], IP["seg_calls"] = [], []
tl3, rep3 = run_idea(idea_project("idea_plan_down"))
ok("a failed plan still writes every segment",
   all(s["prompt"] for s in tl3["segments"]))
ok("and says loudly that it had no plan",
   "plan          FAILED — the plan request failed: connection refused" in rep3)
IP["plan_mode"] = "short_then_full"

# --------------------------------------------------------------------------
print("")
print("H3 Segment Prompter - the idea's timing, shot count and picture jobs hold")
# Same run. The idea said the first 28 seconds are instrumental, and three
# clips inside them sang. It said the character sheet is not to be used in the
# video, and seven clips opened on it. shots_per_segment was 2, and one clip
# had 25 shots.
first = segs[0]["prompt"]
ok("a character sheet is never the opening frame",
   all("opens on <Picture 2>" not in s["prompt"]["detailed_description"]
       for s in segs))
ok("the place it was used for becomes the stage picture",
   "opens on <Picture 1>, a wide of the main stage"
   in first["detailed_description"], first["detailed_description"][:300])
ok("a subject named 'from' the sheet keeps only its tag",
   all("from <Picture 2>" not in s["prompt"]["detailed_description"]
       for s in segs))
ok("the sheet is still credited where it belongs, in the definitions",
   all("<Picture 2>" in s["prompt"]["subject_definitions"] for s in segs))
ok("its retention line says identity only, never first frame",
   all("<Picture 2>: reference - the appearance of the subjects taken from it "
       "only" in s["prompt"]["retention_analysis"]
       and "first frame" not in s["prompt"]["retention_analysis"]
       for s in segs))
ok("the stage picture is in every segment",
   all("<Picture 1>" in s["prompt"]["detailed_description"] for s in segs))
ok("and retained as the setting",
   all("<Picture 1>" in s["prompt"]["retention_analysis"] for s in segs))


def performed(body):
    """The description minus the instrumental cue, which says nobody sings."""
    return vocals_mod.ANY_CUE_RE.sub("", body)


sung = [s for s in segs
        if "sings into the mic" in s["prompt"]["detailed_description"]]
ok("a metal singer sings: 'raps' is not left in",
   all("raps" not in performed(s["prompt"]["detailed_description"])
       for s in segs) and sung)
ok("the picture jobs reach the writer",
   all("PICTURE JOBS" in c["system"] for c in IP["seg_calls"]))

check("shots_per_segment caps each clip at 2 shots",
      [len(re.findall(r"\[Shot \d+\]", s["prompt"]["detailed_description"]))
       for s in segs], [2] * 11)
ok("and the writer is told so",
   all("at most 2 shot(s)" in c["user"] for c in IP["seg_calls"]
       if "SEGMENT" in c["user"] and "LEFT" not in c["user"]))
ok("retention points at no shot the clip does not have",
   all("[Shot 7]" not in s["prompt"]["retention_analysis"] for s in segs))
check("a long budget is still bounded by the clip's length",
      nodes_prompt.shot_budget(10, 7.818), 5)
check("and never below one shot", nodes_prompt.shot_budget(0, 1.0), 1)

# the model writes eight shots into a two-shot clip
IP["shots"] = 8
IP["plan_calls"], IP["seg_calls"] = [], []
tl5, rep5 = run_idea(idea_project("idea_many_shots"))
body5 = tl5["segments"][0]["prompt"]["detailed_description"]
check("eight shots written, two kept",
      len(re.findall(r"\[Shot \d+\]", body5)), 2)
ok("the folded shots keep their action", body5.count("a cut to the crowd") == 6,
   body5[:400])
ok("and the report names it", "6 extra shot(s) folded into shot 2" in rep5)
IP["shots"] = 2

# the plan did not place the vocals itself: its per-segment states are used
check("with no vocal times, the plan's segment states become the timeline",
      tl["context"].get("vocal_timeline"),
      [{"start": 0.0, "end": 23.454, "kind": "instrumental"},
       {"start": 23.454, "end": 85.998, "kind": "vocals"}])

# the plan places them: first 28 seconds instrumental, as the idea says
IP["sections"] = [{"start": 0, "end": 28, "kind": "instrumental"}]
IP["plan_calls"], IP["seg_calls"] = [], []
tl6, rep6 = run_idea(idea_project("idea_vocals"))
check("the idea's 28 instrumental seconds become the vocal timeline",
      tl6["context"]["vocal_timeline"],
      [{"start": 0.0, "end": 28.0, "kind": "instrumental"},
       {"start": 28.0, "end": 85.998, "kind": "vocals"}])
by_seg = {}
for c in IP["seg_calls"]:
    m = re.search(r"SEGMENT (\d+) OF 11", c["user"])
    if m and "LEFT" not in c["user"]:
        by_seg[int(m.group(1))] = c
instrumental = nodes_prompt.INSTRUMENTAL_GUIDANCE.strip()
check("clips 1-3 are written as instrumental, 4 on are not",
      [instrumental in by_seg[n]["system"] for n in range(1, 12)],
      [True, True, True] + [False] * 8)
ok("clip 4 is told exactly where the singing starts inside it",
   "sung only from 4.545s to 7.818s" in by_seg[4]["user"],
   [l for l in by_seg[4]["user"].splitlines() if "sung" in l])
ok("nobody sings in the instrumental clips",
   all(not re.search(r"\b(sings|raps|singing|mouth matching)\b",
                     performed(s["prompt"]["detailed_description"]))
       for s in tl6["segments"][:3]))
ok("the singer still sings once the vocals start",
   all("sings into the mic" in s["prompt"]["detailed_description"]
       for s in tl6["segments"][4:]))
ok("the report shows the timeline and where it came from",
   "vocals        from your idea (the idea's own times): 00:00.000-00:28.000 "
   "instrumental, 00:28.000-01:25.998 vocals — 7 sung, 3 instrumental, "
   "1 mixed clip(s)" in rep6, [l for l in rep6.splitlines() if "vocals" in l])

# an idea that says nothing about the song gets no timeline, whatever the
# model claims
IDEA_SAVED = IDEA
IDEA = "a gritty metal band on <Picture 1>, faces from <Picture 2>."
IP["plan_calls"], IP["seg_calls"] = [], []
tl7, rep7 = run_idea(idea_project("idea_no_timing"))
check("no timing in the idea, no vocal timeline",
      tl7["context"].get("vocal_timeline"), None)
ok("so every clip follows audio_role, as before",
   all(instrumental not in c["system"] for c in IP["seg_calls"]))
IDEA = IDEA_SAVED
IP["sections"] = []
# --------------------------------------------------------------------------
print("")
print("H3 Segment Prompter - the picture and subject numbers reach H3 unchanged")
# The prompt engine renumbers every tag by first appearance, and its dedupe
# renumbers definitions from 1. subject_definitions named the character sheet,
# <Picture 2>, first, so in every prompt the sheet became <Picture 1> and the
# stage <Picture 2>: the text described the two wired images the wrong way
# round. The stub below renumbers exactly as the real engine does.
_LABEL = re.compile(r"<\s*(Subject|Picture|Video|Audio)\s*(\d+)\s*>")


def renumbering_normalize(text, duration=0.0):
    order = {}
    for kind, n in _LABEL.findall(text):
        seen = order.setdefault(kind, [])
        if n not in seen:
            seen.append(n)
    return _LABEL.sub(lambda m: "<%s %d>" % (
        m.group(1), order[m.group(1)].index(m.group(2)) + 1), text)


def renumbering_dedupe(text):
    parts = [p.strip() for p in re.split(r"(?=<Subject\s*\d+>\s*(?:=|is\s))", text)
             if p.strip().startswith("<Subject")]
    if not parts:
        return text
    return "\n".join("<Subject %d>%s" % (i + 1, re.sub(r"^<Subject\s*\d+>", "", p))
                     for i, p in enumerate(parts))


def renumbering_render(obj, duration=0.0):
    obj = dict(obj)
    obj["subject_definitions"] = renumbering_dedupe(
        obj.get("subject_definitions") or "")
    return renumbering_normalize("\n\n".join(
        "%s:\n%s" % (k, obj.get(k) or "N/A") for k in nodes_prompt.SECTIONS))


_saved = (engine.render_full_ref, engine.normalize_labels, engine.dedupe_subjects)
engine.render_full_ref = renumbering_render
engine.normalize_labels = renumbering_normalize
engine.dedupe_subjects = renumbering_dedupe
IP["plan_calls"], IP["seg_calls"] = [], []
tl8, _ = run_idea(idea_project("idea_numbers"))
engine.render_full_ref, engine.normalize_labels, engine.dedupe_subjects = _saved

canon8 = nodes_prompt.defining_lines(tl8["context"]["subject_definitions"])
wrong = []
for s in tl8["segments"]:
    p = s["prompt"]
    if "Staged in the setting of <Picture 1>" not in p["detailed_description"] \
            and "<Picture 1>" not in p["detailed_description"]:
        wrong.append("%s lost the stage picture" % s["id"])
    if "<Picture 2>, a wide" in p["detailed_description"]:
        wrong.append("%s put the sheet back in the frame" % s["id"])
    for n, line in nodes_prompt.defining_lines(p["subject_definitions"]).items():
        if line != canon8.get(n):
            wrong.append("%s: <Subject %d> is defined as %r" % (s["id"], n, line[:50]))
ok("every prompt cites the stage as <Picture 1> and the sheet as <Picture 2>, "
   "and every subject keeps its number", not wrong, "; ".join(wrong[:4]))

seg_subset = {"subject_definitions": canon8[4] + "\n" + canon8[2],
              "summary": "[reference generation] x",
              "retention_analysis": "<Subject 2>: fully_preserved.",
              "detailed_description": "[Shot 1] <Subject 2> and <Subject 4> play.",
              "overall_soundscape": "x", "non_diegetic_music": "y"}
engine.render_full_ref = renumbering_render
engine.normalize_labels = renumbering_normalize
kept = nodes_prompt.render_keeping_labels(seg_subset, 8.0)
engine.render_full_ref, engine.normalize_labels, engine.dedupe_subjects = _saved
check("a segment defining only subjects 4 and 2 still defines 4 and 2",
      sorted(nodes_prompt.defining_lines(kept["subject_definitions"])), [2, 4])
check("and its description still cites 2 and 4",
      re.findall(r"<Subject \d>", kept["detailed_description"]),
      ["<Subject 2>", "<Subject 4>"])

engine.image_b64 = _real_image_b64

engine.generate = fake_generate
shutil.rmtree(WORK, ignore_errors=True)
print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
