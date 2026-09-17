"""The planner's post-processing through the REAL prompt engine.

Run from the pack root:

    python tests/test_bridge.py

test_end_to_end.py replaces every engine function with a stub, so it cannot
see the prompt creator's renderer, timing fixer or label normaliser undoing
what this pack enforced. This test loads the real prompt creator from beside
this pack and pushes a clip through it. Skipped, loudly, when it is not there.
"""

import glob
import importlib
import os
import sys
import types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FAILED = []


def ok(label, condition, detail=""):
    if condition:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s%s" % (label, ("\n       " + str(detail)) if detail else ""))


def find_creator():
    candidates = glob.glob(os.path.join(os.path.dirname(HERE), "*", "h3_prompt_creator.py"))
    candidates += glob.glob(r"D:\Claude work\ComfyUI-H3-Prompt-Creator\h3_prompt_creator.py")
    for path in candidates:
        pkg_dir = os.path.dirname(path)
        name = "bridge_creator_pkg"
        pkg = types.ModuleType(name)
        pkg.__path__ = [pkg_dir]
        sys.modules[name] = pkg
        try:
            return importlib.import_module(name + ".h3_prompt_creator"), pkg_dir
        except Exception as ex:
            print("  (could not load %s: %s)" % (path, ex))
    return None, None


creator, where = find_creator()
if creator is None:
    print("SKIPPED: ComfyUI-H3-Prompt-Creator not found beside this pack, so the "
          "real engine could not be checked. Install it next to this pack.")
    sys.exit(0)
print("real engine from %s" % where)

from h3_planner import engine, nodes_plan, nodes_prompt, vocals  # noqa: E402

ok("the planner's bridge finds the loaded prompt creator", engine.creator() is creator)
ok("both packs recognise singing with the very same pattern",
   vocals.SINGING_RE.pattern == creator._SINGING_RE.pattern
   and vocals.SINGING_RE.flags == creator._SINGING_RE.flags)

import tempfile  # noqa: E402
os.environ["USERPROFILE"] = os.environ["HOME"] = tempfile.mkdtemp(prefix="h3bridge_")
proj = nodes_plan.H3PlannerProject().build(
    "bridge", 24.0, "9:16", "draft", 0.3, 1.2, 32, 4242, 17, 5, 5, "up", 10.5, 8.0, 10)[0]

SECTIONS = [{"start": 0.0, "end": 25.3, "kind": "instrumental"},
            {"start": 25.3, "end": 27.1, "kind": "vocals"},
            {"start": 27.1, "end": 30.0, "kind": "instrumental"},
            {"start": 30.0, "end": 78.2, "kind": "vocals"}]

model_reply = {
    "subject_definitions": "<Subject 1> the lead singer.",
    "summary": "[reference generation] The intro.",
    "retention_analysis": "<Subject 1> (appears in [Shot 1]): fully_copy - identity kept.",
    "detailed_description": (
        "[Shot 1] gritty handheld, <Subject 1> screams into the mic. The drummer "
        "pounds the kit. [Shot 2] At 00:03.000, <Subject 1> lip-syncs the chorus "
        "as smoke rolls in."),
    "overall_soundscape": "Crowd noise.",
    "non_diegetic_music": "N/A",
}
seg = {"id": "seg_01", "target_duration": 6.0, "render_duration": 6.375,
       "source": {"start": 2.0, "end": 8.0, "shots": []}}
allowed = nodes_prompt._allowed_tags({
    "members": [{"tag": "<Subject 1>", "kind": "Subject", "slot": 1},
                {"tag": "<Audio 1>", "kind": "Audio", "slot": 0}]})
clip = vocals.window(SECTIONS, 2.0, 8.0)

prompt, invented = nodes_prompt.H3PlannerSegmentPrompter()._post_process(
    dict(model_reply), seg, "gritty handheld", allowed, (), "<Audio 1>",
    "performed on camera", "<Subject 1>", clip)
body = prompt["detailed_description"]

ok("after the real renderer, the intro clip has no singing",
   "screams into the mic" not in body and "lip-syncs" not in body, body)
ok("after the real renderer, the clip says nobody sings",
   "<Audio 1> is instrumental throughout this clip" in body, body)
ok("the real timing fixer kept both shots", "[Shot 1]" in body and "[Shot 2]" in body, body)
ok("the clip's look survives", "gritty handheld" in body, body)
ok("the soundscape tells H3 the passage is instrumental",
   "instrumental" in prompt["overall_soundscape"]
   and "performed on camera" not in prompt["overall_soundscape"], prompt["overall_soundscape"])
ok("the real renderer's retention fix leaves the audio line valid",
   "<Audio 1>: fully_copy" in prompt["retention_analysis"], prompt["retention_analysis"])
ok("a visible subject marked with an audio word is corrected by the real engine",
   "<Subject 1> (appears in [Shot 1]): fully_preserved" in prompt["retention_analysis"],
   prompt["retention_analysis"])

# And a clip where the vocal enters partway, through the same real path.
mixed = vocals.window(SECTIONS, 22.0, 28.0)
prompt, _ = nodes_prompt.H3PlannerSegmentPrompter()._post_process(
    dict(model_reply), seg, "gritty handheld", allowed, (), "<Audio 1>",
    "performed on camera", "<Subject 1>", mixed)
ok("a mixed clip's timing sentence survives the real renderer untouched",
   "carries the vocal only from 00:03.300 to 00:05.100 of this clip" in prompt["detailed_description"],
   prompt["detailed_description"])

print()
if FAILED:
    print("%d FAILED: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("all bridge checks passed")
