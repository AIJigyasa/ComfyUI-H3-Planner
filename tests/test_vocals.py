"""Instrumental clips must not show anyone singing.

Run from the pack root:

    python tests/test_vocals.py

The symptom: a music video whose song opens on a long instrumental showed the
singer mouthing words over it, because "performed on camera" was applied to
every segment alike. The timeline used here is the one measured on the user's
real 86-second track.
"""

import glob
import importlib.util
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = tempfile.mkdtemp(prefix="h3vocals_")
os.environ["USERPROFILE"] = os.environ["HOME"] = WORK

from h3_planner import (engine, nodes_plan, nodes_prompt, nodes_slice,  # noqa: E402
                        splitter, vocals)

FAILED = []


def ok(label, condition, detail=""):
    if condition:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s%s" % (label, ("\n       " + str(detail)) if detail else ""))


def project(name):
    return nodes_plan.H3PlannerProject().build(
        name, 24.0, "9:16", "draft", 0.3, 1.2, 32, 4242, 17, 5, 5, "up",
        10.5, 8.0, 10)[0]


# Written by the prompt creator for "AI Jigyasa (1).mp3", first 86 seconds.
LINE = ("Vocal timeline of <Audio 1>: 00:00.000-00:25.300 instrumental; "
        "00:25.300-00:27.100 vocals; 00:27.100-00:30.000 instrumental; "
        "00:30.000-01:18.200 vocals; 01:18.200-01:20.500 instrumental; "
        "01:20.500-01:23.000 vocals; 01:23.000-01:26.000 instrumental.")

TREATMENT = "\n".join([
    "subject_definitions:",
    "<Subject 1> the lead singer, long black hair, torn denim jacket.",
    "",
    "summary:",
    "[reference generation] A gritty metal performance.",
    "",
    "retention_analysis:",
    "<Subject 1>: fully_preserved. <Audio 1>: fully_copy, all shots.",
    "",
    "overall_soundscape:",
    "<Audio 1> is used exactly as supplied. " + LINE,
    "",
    "non_diegetic_music:",
    "N/A.",
    "",
    "detailed_description:",
    "[Shot 1] gritty handheld, <Subject 1> screams into the mic as the band "
    "thrashes. The drummer pounds the kit. <Audio 1> is instrumental here: "
    "nobody sings or raps, and every mouth stays closed.",
    "[Shot 2] At 00:08.000, <Subject 1> sings the opening line to the crowd. "
    "Smoke rolls across the stage. <Audio 1> is instrumental here: nobody "
    "sings or raps, and every mouth stays closed.",
    "[Shot 3] At 00:16.000, the guitarist leans into a riff under red light.",
    "[Shot 4] At 00:30.000, <Subject 1> grips the stand and sings. <Audio 1> "
    "carries the vocal here, <d>[English] We're running outward</d>.",
    "[Shot 5] At 00:38.000, a wide of the crowd surging. <Audio 1> carries "
    "the vocal here, <d>[English] no money left</d>.",
])

CAST = {
    "members": [
        {"tag": "<Subject 1>", "kind": "Subject", "slot": 1, "key": "singer",
         "role": "character", "note": "", "file": "lead_singer.png"},
        {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "key": "song",
         "role": "audio", "note": "", "file": "ai_jigyasa_song.mp3"},
    ],
    "size": 1, "has_audio": True, "audio_tag": "<Audio 1>",
    "counts": {"Subject": 1, "Audio": 1},
}

print("\nreading the timeline")
label, sections = vocals.parse("Some prose. " + LINE + " More prose after.")
ok("the prompt creator's line parses", label == "<Audio 1>" and len(sections) == 7,
   (label, sections[:2]))
ok("minutes are read correctly", sections[3] == {"start": 30.0, "end": 78.2, "kind": "vocals"},
   sections[3])
ok("no line means no timeline", vocals.parse("<Audio 1> is used as supplied.") == ("", []))

# Both packs must agree on the format. When the prompt creator is installed
# beside this pack, write the line with ITS formatter and read it with ours.
creator = None
for candidate in glob.glob(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "*", "h3_vocal_timeline.py")) + \
        glob.glob(r"D:\Claude work\ComfyUI-H3-Prompt-Creator\h3_vocal_timeline.py"):
    spec = importlib.util.spec_from_file_location("creator_vocal_timeline", candidate)
    creator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(creator)
    break
if creator is not None:
    written = creator.format_vocal_timeline(sections, "<Audio 1>")
    ok("the prompt creator's own formatter writes a line this pack reads back",
       vocals.parse(written) == ("<Audio 1>", sections), written)
else:
    print("  skip the prompt creator is not installed beside this pack")

print("\nwhat each clip holds")
ok("a clip in the intro is instrumental", vocals.window(sections, 2, 8)["state"] == "instrumental")
ok("a clip in the verse is sung", vocals.window(sections, 40, 46)["state"] == "vocals")
mixed = vocals.window(sections, 22, 30)
ok("a clip with the shout in it is mixed, rebased to the clip",
   mixed["state"] == "mixed" and mixed["spans"] == [(3.3, 5.1)], mixed)
ok("a sliver of vocal over a cut does not make a clip sung",
   vocals.window(sections, 78.0, 80.4)["state"] == "instrumental")

print("\nthe splitter carries it")
segments, context, warnings = splitter.split_treatment(TREATMENT, 46.0, 10.5, 2.0, 1)
ok("the context holds the timeline", len(context.get("vocal_timeline") or []) == 7)
ok("a timeline covering the video raises no warning",
   not any("vocal timeline" in w for w in warnings), warnings)
_s, _c, short_warnings = splitter.split_treatment(TREATMENT, 120.0, 10.5, 2.0, 1)
ok("a timeline shorter than the video is reported",
   any("vocal timeline covers 86.0s but the video is 120.0s" in w for w in short_warnings),
   short_warnings)
_s, beat_context, _w = splitter.split_on_cuts(
    TREATMENT, [{"start": 0.0, "end": 8.0}, {"start": 8.0, "end": 16.0}], 46.0)
ok("the beat-map path carries it too", len(beat_context.get("vocal_timeline") or []) == 7)

citing = ("[Shot 1] the singer steps up to the mic.\n"
          "[Shot 2] At 00:05.000, a wide shot matching the framing of [Shot 1] as smoke rolls.\n"
          "[Shot 3] At 00:09.000, the crowd surges.")
cited = splitter.parse_shots(citing, 12.0)
ok("a mid-sentence citation of a shot is not taken for a new shot",
   [s["number"] for s in cited] == [1, 2, 3], [(s["number"], s["start"]) for s in cited])
ok("the citation stays in the text of the shot that makes it",
   "matching the framing of [Shot 1] as smoke rolls" in cited[1]["text"], cited[1]["text"])
ok("a timed marker mid-sentence is still a shot",
   [s["number"] for s in splitter.parse_shots(
       "[Shot 1] a wide. The next shot, [Shot 2] At 00:04.000, a close-up.", 8.0)] == [1, 2])

print("\nenforcing one clip")
dirty = {
    "subject_definitions": "<Subject 1> the lead singer.",
    "summary": "A clip.",
    "retention_analysis": "<Subject 1>: fully_preserved.",
    "detailed_description": (
        "[Shot 1] gritty handheld, <Subject 1> sings <d>[English] welcome</d> to "
        "the crowd. The drummer pounds the kit. [Shot 2] At 00:04.000, <Subject 1> "
        "lip-syncs the chorus. <Audio 1> is instrumental here: nobody sings or "
        "raps, and every mouth stays closed."),
    "overall_soundscape": "x", "non_diegetic_music": "x",
}
inst = vocals.window(sections, 2, 8)
out, changed = nodes_prompt.enforce_audio_exact(
    dict(dirty), "<Audio 1>", "performed on camera", "<Subject 1>", inst)
body = out["detailed_description"]
ok("removing shot 1's singing sentence keeps the look it opened with",
   body.startswith("[Shot 1] gritty handheld, The drummer"), body)
ok("singing is removed from an instrumental clip",
   "sings <d>" not in body and "lip-syncs" not in body and "<d>" not in body, body)
ok("a removed sentence keeps its shot marker",
   "[Shot 1]" in body and "[Shot 2] At 00:04.000," in body, body)
ok("non-singing direction is kept", "drummer pounds the kit" in body, body)
ok("the treatment's whole-video cue is replaced by the clip's own",
   body.count("is instrumental") == 1
   and body.endswith("<Audio 1> is instrumental throughout this clip: nobody sings "
                     "or raps, and every mouth stays closed."), body)
ok("the soundscape no longer says the vocal is performed on camera",
   "instrumental" in out["overall_soundscape"]
   and "performed on camera" not in out["overall_soundscape"], out["overall_soundscape"])
ok("the change is reported", any("singing sentence" in c for c in changed), changed)

sung = vocals.window(sections, 40, 46)
out, _ = nodes_prompt.enforce_audio_exact(
    dict(dirty), "<Audio 1>", "performed on camera", "<Subject 1>", sung)
ok("a sung clip keeps its singing", "lip-syncs the chorus" in out["detailed_description"],
   out["detailed_description"])
ok("a sung clip's soundscape is the performed-on-camera line, as before",
   out["overall_soundscape"] == nodes_prompt.soundscape_line(
       "<Audio 1>", "performed on camera", "<Subject 1>"), out["overall_soundscape"])
ok("a sung clip drops the treatment's instrumental cue for a shot it no longer is",
   "is instrumental here" not in out["detailed_description"], out["detailed_description"])

out, _ = nodes_prompt.enforce_audio_exact(
    dict(dirty), "<Audio 1>", "performed on camera", "<Subject 1>", mixed)
ok("a mixed clip says when, on the clip's own clock",
   "carries the vocal only from 00:03.300 to 00:05.100 of this clip" in out["detailed_description"],
   out["detailed_description"])
ok("a mixed clip's soundscape gives the same window",
   "only from 00:03.300 to 00:05.100" in out["overall_soundscape"], out["overall_soundscape"])

before, _ = nodes_prompt.enforce_audio_exact(dict(dirty), "<Audio 1>", "performed on camera", "<Subject 1>")
ok("without a timeline nothing changes from the old behaviour",
   "lip-syncs the chorus" in before["detailed_description"]
   and "performed on camera" in before["overall_soundscape"], before)

print("\nthe Segment Slicer, end to end")
slicer = nodes_slice.H3PlannerSegmentSlicer()
timeline, report = slicer.slice_treatment(
    project("vocals_slice"), TREATMENT, 46.0, 1, 2.0, "performed on camera", True, cast=CAST)
by_start = {s["source"]["start"]: s for s in timeline["segments"]}
first = by_start[0.0]["prompt"]
ok("the intro clip has no singing in it",
   "screams into the mic" not in first["detailed_description"], first["detailed_description"])
ok("the intro clip keeps its style prefix after the singing is removed",
   first["detailed_description"].startswith("[Shot 1] gritty handheld,"), first["detailed_description"])
ok("the intro clip says so", "instrumental throughout this clip" in first["detailed_description"],
   first["detailed_description"])
verse = by_start[30.0]["prompt"]
ok("the verse clip still sings", "grips the stand and sings" in verse["detailed_description"],
   verse["detailed_description"])
ok("the verse clip keeps its lyric", "<d>[English] We're running outward</d>" in verse["detailed_description"],
   verse["detailed_description"])
ok("no clip carries a whole-video timestamp from the treatment",
   not any("00:25.300" in s["prompt"]["detailed_description"] for s in timeline["segments"]))
ok("the report counts sung and instrumental clips",
   "vocals        timeline from the treatment" in report and "instrumental" in report, report[:900])
ok("each row names its clip's state", "[instrumental]" in report and "[vocals]" in report, report)

print("\nthe Segment Prompter, end to end")
SYSTEMS, USERS = [], []


def fake_generate(cfg, system, user, schema, required_keys=(), min_words=0, images=None):
    SYSTEMS.append(system)
    USERS.append(user)
    # Misbehaves the way the model does: sings in every clip regardless.
    return ({
        "subject_definitions": "<Subject 1> the lead singer.",
        "summary": "[reference generation] Clip %d." % len(SYSTEMS),
        "retention_analysis": "<Subject 1>: fully_preserved.",
        "detailed_description": (
            "[Shot 1] gritty handheld, <Subject 1> belts out the chorus, mouth "
            "matching the words, number %d. The band thrashes." % len(SYSTEMS)),
        "overall_soundscape": "Ambient crowd noise.",
        "non_diegetic_music": "N/A",
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

prompter = nodes_prompt.H3PlannerSegmentPrompter()
proj = project("vocals_prompt")
timeline, report = prompter.write(
    proj, TREATMENT, 46.0, 10.0, 1, "music video", "performed on camera",
    "Ollama (Local)", "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
starts = [s["source"]["start"] for s in timeline["segments"]]
intro_i = starts.index(0.0)
verse_i = starts.index(30.0)


def last_call_for(index):
    """The system and user prompt of the LAST model call for one segment.

    One call per segment is not guaranteed: two halves of one long shot read
    alike, and the repeat check asks again, which shifts every later index.
    """
    tag = "SEGMENT %d OF %d" % (index + 1, len(timeline["segments"]))
    hits = [i for i, u in enumerate(USERS) if tag in u]
    return SYSTEMS[hits[-1]], USERS[hits[-1]]


INTRO_SYSTEM, INTRO_USER = last_call_for(intro_i)
VERSE_SYSTEM, _VERSE_USER = last_call_for(verse_i)
ok("the intro clip's system prompt is the instrumental guidance",
   nodes_prompt.INSTRUMENTAL_GUIDANCE.strip() in INTRO_SYSTEM
   and "THE AUDIO IS PERFORMED ON CAMERA" not in INTRO_SYSTEM)
ok("the verse clip's system prompt still says the vocal is performed on camera",
   "THE AUDIO IS PERFORMED ON CAMERA" in VERSE_SYSTEM)
ok("the intro clip's brief says there is no vocal",
   "VOCALS IN THIS CLIP" in INTRO_USER and "none. The audio is instrumental" in INTRO_USER)
ok("the brief does not show the treatment's whole-video cues",
   "is instrumental here" not in INTRO_USER, INTRO_USER[-600:])
intro_body = timeline["segments"][intro_i]["prompt"]["detailed_description"]
ok("a model that sings anyway is corrected in the intro clip",
   "belts out" not in intro_body and "instrumental throughout this clip" in intro_body, intro_body)
ok("the verse clip keeps the model's singing",
   "belts out" in timeline["segments"][verse_i]["prompt"]["detailed_description"])
ok("the report states the vocal plan", "vocals        timeline from the treatment" in report, report[:700])

# A prompt written before the timeline existed must not be reused as unchanged.
plain = TREATMENT.replace(LINE, "")
SYSTEMS.clear()
t1, _ = prompter.write(project("vocals_reuse"), plain, 46.0, 10.0, 1, "music video",
                       "performed on camera", "Ollama (Local)",
                       "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
first_pass = len(SYSTEMS)
_t2, r2 = prompter.write(project("vocals_reuse"), TREATMENT, 46.0, 10.0, 1, "music video",
                         "performed on camera", "Ollama (Local)",
                         "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
ok("adding a timeline rewrites clips instead of reusing the old ones",
   len(SYSTEMS) > first_pass and "unchanged, reused" not in r2.split("segments:")[1].split("[instrumental]")[0],
   r2[-600:])
_t3, r3 = prompter.write(project("vocals_reuse"), plain, 46.0, 10.0, 1, "music video",
                         "performed on camera", "Ollama (Local)",
                         "http://127.0.0.1:11434", "qwen3-vl:8b", 0.25, True, 0, cast=CAST)
ok("with no timeline the report says what that risks and how to fix it",
   "no vocal timeline in the treatment" in r3 and "Full-Reference" in r3, r3[:700])

print()
if FAILED:
    print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all vocal checks passed")
