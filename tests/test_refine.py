"""Refining one segment from a note: only the shot description may move.

    python tests/test_refine.py

The model is stubbed, and stubbed adversarially — a well-behaved stub would
prove nothing here, since the whole job is surviving a model that drops tags,
loses the spoken line, or rewrites sections it was told not to touch.
"""

import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = tempfile.mkdtemp(prefix="h3refine_")
os.environ["USERPROFILE"] = os.environ["HOME"] = WORK

from h3_planner import engine, refine, store  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def ok(label, condition, detail=""):
    check(label + (" (%s)" % detail if detail else ""), bool(condition), True)


SECS = ("subject_definitions", "summary", "retention_analysis",
        "detailed_description", "overall_soundscape", "non_diegetic_music")

ORIGINAL = {
    "subject_definitions": "<Subject 1> a young man in a black jacket. "
                           "<Subject 2> a glass bottle with a gold cap.",
    "summary": "A rooftop at golden hour.",
    "retention_analysis": "<Subject 1>: fully_preserved. <Audio 1>: fully_copy.",
    "detailed_description": (
        "[Shot 1] Cinematic 35mm, <Subject 1> walks toward camera holding "
        "<Subject 2>. He says, <d>[English] This is the one.</d> "
        "[Shot 2] At 00:02.500, he lifts <Subject 2> to the light."),
    "overall_soundscape": "All audio comes from <Audio 1>, used exactly as "
                          "supplied. No additional sound is present.",
    "non_diegetic_music": "N/A - the audio is supplied by <Audio 1>.",
}

CONTEXT = {
    "style_prefix": "Cinematic 35mm",
    "world": "A rooftop at golden hour.",
    "backend": {"provider": "Ollama (Local)", "ollama_model": "qwen3-vl:8b",
                "ollama_url": "http://127.0.0.1:11434", "temperature": 0.3},
}


def seg_of(prompt=None, link="cut", state=store.PENDING):
    return {"id": "seg_02", "index": 1, "target_duration": 5.167,
            "render_duration": 5.167, "link": link, "state": state,
            "prompt": dict(prompt if prompt is not None else ORIGINAL),
            "seed": 1, "refine_seed": 2, "clips": {}}


def timeline_of(seg):
    return {"version": 1, "project": "t", "context": dict(CONTEXT),
            "segments": [seg]}


REPLIES = []


def stub(cfg, system, user, schema, required_keys=(), min_words=0, images=None):
    STATE["calls"] += 1
    STATE["last_user"] = user
    reply = REPLIES.pop(0) if REPLIES else None
    if reply is None:
        return None, "stub"
    return ({"detailed_description": reply}, "stub")


STATE = {"calls": 0, "last_user": ""}

engine.available = lambda: True
engine.backend_config = lambda **kw: types.SimpleNamespace(**kw)
engine.json_schema = lambda p, r: {}
engine.clean = lambda v: "" if v is None else str(v).strip()
engine.fix_shot_times = lambda t, d: t
engine.generate = stub

GOOD = ("[Shot 1] Cinematic 35mm, <Subject 1> walks AWAY from a low camera "
        "holding <Subject 2>. He says, <d>[English] This is the one.</d> "
        "[Shot 2] At 00:02.500, he lifts <Subject 2> to the light.")

try:
    # ----------------------------------------------------------------------
    print("\na good revision")
    STATE["calls"] = 0
    REPLIES[:] = [GOOD]
    seg = seg_of()
    msg = refine.refine_segment(timeline_of(seg), seg, "make it a low angle, "
                                "he should walk away from camera")
    ok("it reports what it did", "seg_02" in msg, msg[:80])
    check("one call was enough", STATE["calls"], 1)
    ok("the description changed",
       "walks AWAY" in seg["prompt"]["detailed_description"])

    for key in SECS:
        if key == "detailed_description":
            continue
        check("%s is untouched" % key, seg["prompt"][key], ORIGINAL[key])

    ok("the note reached the model", "low angle" in STATE["last_user"])
    ok("the model was told which tags must survive",
       "<Subject 1>" in STATE["last_user"] and "<Subject 2>" in STATE["last_user"])
    ok("and the spoken line", "This is the one" in STATE["last_user"])
    ok("the note is remembered on the segment",
       "low angle" in seg.get("refine_note", ""))
    ok("the spec hash changed, so the stored clip is invalidated",
       seg["spec_hash"] != store.spec_hash(
           dict(seg, prompt=ORIGINAL)))

    # ----------------------------------------------------------------------
    print("\nthe model misbehaving")

    # drops a tag -> rejected, retried, accepted on the second reply
    STATE["calls"] = 0
    REPLIES[:] = ["[Shot 1] Cinematic 35mm, a man walks away holding a bottle.",
                  GOOD]
    seg = seg_of()
    refine.refine_segment(timeline_of(seg), seg, "low angle")
    check("a dropped tag is retried", STATE["calls"], 2)
    ok("and the tags are back",
       "<Subject 1>" in seg["prompt"]["detailed_description"]
       and "<Subject 2>" in seg["prompt"]["detailed_description"])

    # drops the spoken line, twice -> refused, segment untouched
    STATE["calls"] = 0
    silent = ("[Shot 1] Cinematic 35mm, <Subject 1> walks away from a low "
              "camera holding <Subject 2>. [Shot 2] At 00:02.500, he lifts it.")
    REPLIES[:] = [silent, silent]
    seg = seg_of()
    before = dict(seg["prompt"])
    try:
        refine.refine_segment(timeline_of(seg), seg, "low angle")
        ok("losing the spoken line is refused", False, "it was accepted")
    except RuntimeError as ex:
        ok("losing the spoken line is refused", "spoken line" in str(ex),
           str(ex)[:90])
    check("and the segment is exactly as it was", seg["prompt"], before)

    # returns nothing at all
    STATE["calls"] = 0
    REPLIES[:] = [None, None]
    seg = seg_of()
    before = dict(seg["prompt"])
    try:
        refine.refine_segment(timeline_of(seg), seg, "low angle")
        ok("an empty reply is refused", False, "it was accepted")
    except RuntimeError:
        ok("an empty reply is refused", True)
    check("the segment survives that too", seg["prompt"], before)

    # tries to introduce a tag the segment never had
    STATE["calls"] = 0
    REPLIES[:] = [GOOD.replace("<Subject 2>", "<Subject 2> and <Video 1>")]
    seg = seg_of()
    refine.refine_segment(timeline_of(seg), seg, "add a video reference")
    ok("a tag the segment never cited is stripped",
       "<Video 1>" not in seg["prompt"]["detailed_description"],
       seg["prompt"]["detailed_description"][:110])

    # rewrites the style prefix away
    STATE["calls"] = 0
    REPLIES[:] = [GOOD.replace("[Shot 1] Cinematic 35mm, ", "[Shot 1] ")]
    seg = seg_of()
    refine.refine_segment(timeline_of(seg), seg, "low angle")
    ok("the style prefix is put back",
       seg["prompt"]["detailed_description"].startswith(
           "[Shot 1] Cinematic 35mm"),
       seg["prompt"]["detailed_description"][:60])

    # ----------------------------------------------------------------------
    print("\nrefusals")
    for kwargs, needle, label in (
        ({"note": ""}, "describe what should change", "an empty note"),
        ({"note": "x", "seg": seg_of(prompt={k: "" for k in SECS})},
         "no written prompt", "a segment with no prompt"),
        ({"note": "x", "seg": seg_of(state=store.LOCKED)},
         "locked", "a locked segment"),
    ):
        target = kwargs.get("seg") or seg_of()
        REPLIES[:] = [GOOD]
        try:
            refine.refine_segment(timeline_of(target), target, kwargs["note"])
            ok("refuses %s" % label, False, "it proceeded")
        except RuntimeError as ex:
            ok("refuses %s" % label, needle in str(ex), str(ex)[:80])

    # A Slicer or Beat Map timeline records no provider, because neither
    # uses a model. Refusing those made refine useless for the whole
    # music-video path, so it borrows local Ollama instead.
    seg = seg_of()
    tl = timeline_of(seg)
    tl["context"].pop("backend")
    REPLIES[:] = [GOOD]
    msg = refine.refine_segment(tl, seg, "low angle")
    ok("a timeline with no provider still refines", "revised" in msg,
       msg[:90])
    ok("and it says the model was borrowed", "local Ollama" in msg,
       msg[:120])
    ok("the revision still landed",
       "walks AWAY" in seg["prompt"]["detailed_description"])
    cfg, borrowed = refine.backend_config({})
    ok("the fallback is flagged as borrowed", borrowed)
    cfg2, borrowed2 = refine.backend_config(
        {"backend": {"provider": "Google Gemini"}})
    ok("a recorded provider is not borrowed", not borrowed2)

    # ----------------------------------------------------------------------
    print("\nthe API key is never written to disk")
    ctx = {}
    store.remember_backend(ctx, {
        "provider": "Google Gemini", "api_model": "gemini-2.0-flash",
        "api_key": "AIzaSyTOTALLY-SECRET", "temperature": 0.3,
        "ollama_url": "http://127.0.0.1:11434"})
    ok("the provider is remembered", ctx["backend"]["provider"] == "Google Gemini")
    ok("the model is remembered",
       ctx["backend"]["api_model"] == "gemini-2.0-flash")
    check("the key is absent", "api_key" in ctx["backend"], False)
    ok("and nowhere in the block at all",
       "SECRET" not in repr(ctx), repr(ctx)[:80])

finally:
    shutil.rmtree(WORK, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
