"""The H3 Segment Slicer: no model, and nothing must leak.

Run from the pack root:

    python tests/test_slice.py

Everything here calls the node's real FUNCTION. There is no provider to stub,
which is the point of the node.
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

WORK = tempfile.mkdtemp(prefix="h3slice_")
os.environ["USERPROFILE"] = os.environ["HOME"] = WORK

from h3_planner import nodes_plan, nodes_slice, splitter, store  # noqa: E402

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


TREATMENT = "\n".join([
    "subject_definitions:",
    "<Subject 1> a young man, close-cropped hair, black jacket.",
    "<Audio 1> the supplied track.",
    "",
    "summary:",
    "A rooftop film in one continuous mood.",
    "",
    "retention_analysis:",
    "<Subject 1>: fully_preserved. <Audio 1>: fully_copy, all shots.",
    "",
    "overall_soundscape:",
    "The supplied track, heard throughout.",
    "",
    "non_diegetic_music:",
    "N/A.",
    "",
    "detailed_description:",
    "[Shot 1] cinematic 35mm, <Subject 1> steps onto the roof at dawn.",
    "[Shot 2] 00:02.500 — he crosses to the ledge, <Subject 1> in profile.",
    "[Shot 3] 00:05.000 — a wide of the skyline behind him.",
    "[Shot 4] 00:07.500 — he turns back toward the door.",
])

CAST = {
    "members": [
        {"tag": "<Subject 1>", "kind": "Subject", "slot": 1, "key": "hero",
         "role": "character", "note": "", "file": "hero_headshot_01.png"},
        {"tag": "<Audio 1>", "kind": "Audio", "slot": 0, "key": "track",
         "role": "audio", "note": "", "file": "delhi_beat_final.mp3"},
    ],
    "size": 1, "has_audio": True, "audio_tag": "<Audio 1>",
    "counts": {"Subject": 1, "Audio": 1},
}

node = nodes_slice.H3PlannerSegmentSlicer()

try:
    # ----------------------------------------------------------------------
    print("\ntimestamps")
    check("zero formats as MM:SS.mmm", nodes_slice.format_timestamp(0), "00:00.000")
    check("seconds keep milliseconds", nodes_slice.format_timestamp(5.5), "00:05.500")
    check("past a minute rolls over", nodes_slice.format_timestamp(65.25), "01:05.250")
    check("negatives are clamped", nodes_slice.format_timestamp(-3), "00:00.000")

    times = nodes_slice.rebase_times(
        [{"at": 0.0}, {"at": 2.0}, {"at": 4.0}], 8.0)
    check("shot 1 always starts at zero", times[0], 0.0)
    ok("and the rest keep their spacing", times == [0.0, 2.0, 4.0], str(times))

    # a treatment that hands back equal or backwards times
    bad = nodes_slice.rebase_times(
        [{"at": 0.0}, {"at": 1.0}, {"at": 1.0}, {"at": 0.5}], 8.0)
    ok("equal and backwards times are made increasing",
       all(b > a for a, b in zip(bad, bad[1:])), str(bad))

    # a shot sitting past the rendered length would be a cue H3 never reaches
    over = nodes_slice.rebase_times(
        [{"at": 0.0}, {"at": 5.0}, {"at": 11.0}], 8.0)
    ok("nothing lands at or past the render", max(over) < 8.0, str(over))
    ok("and the run is still increasing",
       all(b > a for a, b in zip(over, over[1:])), str(over))
    check("no shots is not a crash", nodes_slice.rebase_times([], 8.0), [])

    # ----------------------------------------------------------------------
    print("\nslicing a real treatment")
    proj = project("slice_e2e")
    timeline, report = node.slice_treatment(
        proj, TREATMENT, 10.0, 2, 2.0, "performed on camera", True,
        cast=CAST)

    ok("it produced segments", len(timeline["segments"]) >= 2,
       str(len(timeline["segments"])))
    ok("every segment has all six sections",
       all(set(s["prompt"]) == set(nodes_slice.SECTIONS)
           for s in timeline["segments"]))
    ok("no section is left blank",
       all(str(v).strip() for s in timeline["segments"]
           for v in s["prompt"].values()))
    ok("the report says no model was used", "no model" in report, report[:80])

    first = timeline["segments"][0]["prompt"]["detailed_description"]
    ok("shot 1 carries no timestamp",
       "[Shot 1]" in first and "[Shot 1] At" not in first, first[:90])
    ok("the style prefix is on shot 1", "cinematic 35mm" in first, first[:90])

    ok("the style prefix is not doubled",
       first.count("cinematic 35mm") == 1, first[:110])
    ok("later shots are renumbered from 1", "[Shot 2] At" in first, first[:200])
    ok("the absolute clock is gone", "00:05.500" not in first, first[:200])

    for seg in timeline["segments"]:
        body = seg["prompt"]["detailed_description"]
        stamps = splitter.SHOT_RE.findall(body)
        nums = [int(n) for n, _, _ in stamps]
        ok("%s numbers its shots from 1" % seg["id"],
           nums == list(range(1, len(nums) + 1)), str(nums))
        secs = [int(m) * 60 + float(s) for _, m, s in stamps if m]
        ok("%s timestamps increase" % seg["id"],
           all(b > a for a, b in zip(secs, secs[1:])), str(secs))
        ok("%s stays inside its render" % seg["id"],
           all(t < seg["render_duration"] for t in secs),
           "%s vs %.3f" % (secs, seg["render_duration"]))

    # ----------------------------------------------------------------------
    print("\nnothing leaks")
    blob = " ".join(v for s in timeline["segments"]
                    for v in s["prompt"].values())
    ok("no upload file name reaches the prompt",
       "hero_headshot_01" not in blob and "delhi_beat_final" not in blob)
    ok("no cast key reaches the prompt", "\"hero\"" not in blob)
    ok("no tag outside the cast is cited",
       "<Video" not in blob and "<Picture 2>" not in blob and
       "<Subject 2>" not in blob)
    ok("the audio is declared exact", "used exactly as supplied" in blob)
    ok("and named as performed on camera", "mouth matches the words" in blob)
    ok("the cast wording is identical in every segment",
       len({s["prompt"]["subject_definitions"]
            for s in timeline["segments"]}) <= 2)

    # a cast tag the treatment cites but the cast does not have
    dirty = TREATMENT.replace("<Subject 1> steps",
                              "<Video 1> and <Subject 4> step")
    tl2, _ = node.slice_treatment(
        project("slice_dirty"), dirty, 10.0, 2, 2.0, "background only", True,
        cast=CAST)
    blob2 = " ".join(v for s in tl2["segments"] for v in s["prompt"].values())
    ok("an invented video tag is stripped", "<Video 1>" not in blob2)
    ok("a cast number that does not exist is stripped",
       "<Subject 4>" not in blob2)
    ok("but the shot survives the strip", "[Shot 1]" in blob2)

    # ----------------------------------------------------------------------
    print("\nrefusals and edge cases")
    for text, needle in (("", "connect the h3_prompt"),
                         ("   ", "connect the h3_prompt"),
                         ("a treatment with no markers", "no [Shot N] markers")):
        try:
            node.slice_treatment(project("slice_bad"), text, 20.0, 2, 2.0,
                                 "background only", True)
            ok("refuses %r" % (text[:24] or "empty"), False, "it returned")
        except ValueError as ex:
            ok("refuses %r" % (text[:24] or "empty"), needle in str(ex),
               str(ex)[:70])

    tl3, rep3 = node.slice_treatment(
        project("slice_nocast"), TREATMENT, 10.0, 2, 2.0,
        "performed on camera", True)
    ok("it works with no cast at all", len(tl3["segments"]) >= 2)
    ok("and says so in the report", "none connected" in rep3)
    blob3 = " ".join(v for s in tl3["segments"] for v in s["prompt"].values())
    ok("with no cast, tags are left alone", "<Subject 1>" in blob3)

    tl4, _ = node.slice_treatment(
        project("slice_minimal"), TREATMENT, 10.0, 2, 2.0, "background only",
        False, cast=CAST)
    ok("carry_sections off still fills every section",
       all(str(v).strip() and v != "" for s in tl4["segments"]
           for v in s["prompt"].values()))
    ok("and does not copy the treatment summary",
       "one continuous mood" not in " ".join(
           s["prompt"]["summary"] for s in tl4["segments"]))

    # ----------------------------------------------------------------------
    print("\nit is deterministic and it persists")
    p5 = project("slice_repeat")
    a1, _ = node.slice_treatment(p5, TREATMENT, 10.0, 2, 2.0,
                                 "performed on camera", True, cast=CAST)
    prompts_a = [s["prompt"] for s in a1["segments"]]
    fps_a = [s["prompt_fingerprint"] for s in a1["segments"]]
    a2, _ = node.slice_treatment(p5, TREATMENT, 10.0, 2, 2.0,
                                 "performed on camera", True, cast=CAST)
    check("the same inputs give byte-identical prompts",
          [s["prompt"] for s in a2["segments"]], prompts_a)
    check("and the same fingerprints",
          [s["prompt_fingerprint"] for s in a2["segments"]], fps_a)

    b1, _ = node.slice_treatment(p5, TREATMENT, 10.0, 2, 2.0,
                                 "background only", True, cast=CAST)
    ok("changing a setting changes the fingerprint",
       b1["segments"][0]["prompt_fingerprint"] != fps_a[0])

    saved = store.load(p5["timeline_path"])
    ok("the timeline is on disk", saved is not None)
    ok("with its prompts", all(s.get("prompt") for s in saved["segments"]))

    # a locked segment is never rewritten
    saved["segments"][0]["state"] = store.LOCKED
    saved["segments"][0]["prompt"] = {k: "LOCKED CONTENT"
                                      for k in nodes_slice.SECTIONS}
    store.save(p5["timeline_path"], saved)
    c1, rep_c = node.slice_treatment(p5, TREATMENT, 10.0, 2, 2.0,
                                     "performed on camera", True, cast=CAST)
    check("a locked segment keeps its prompt",
          c1["segments"][0]["prompt"]["summary"], "LOCKED CONTENT")
    ok("and the report says it was left alone", "locked, left alone" in rep_c)
    ok("while the others were still written",
       c1["segments"][1]["prompt"]["summary"] != "LOCKED CONTENT")

    # The two sound sections are deliberately NOT the treatment's, because a
    # supplied track already contains every sound. Silently swapping them read
    # as the node losing them, so the report has to name the swap.
    d1, rep_d = node.slice_treatment(p5, TREATMENT, 10.0, 2, 2.0,
                                     "performed on camera", True, cast=CAST)
    seg = next(s for s in d1["segments"] if s["state"] != store.LOCKED)
    ok("overall_soundscape is the exact-audio line, not the treatment's",
       "used exactly as supplied" in seg["prompt"]["overall_soundscape"],
       seg["prompt"]["overall_soundscape"][:80])
    ok("non_diegetic_music defers to the track",
       seg["prompt"]["non_diegetic_music"].startswith("N/A"),
       seg["prompt"]["non_diegetic_music"][:60])
    ok("and the report says both were replaced",
       "REPLACED in every segment" in rep_d,
       [l for l in rep_d.splitlines() if l.startswith("soundscape")][:1])

    # With no audio connected there is nothing to protect, so the treatment's
    # own sound writing must survive untouched.
    p6 = project("slice_no_audio")
    e1, rep_e = node.slice_treatment(p6, TREATMENT, 10.0, 2, 2.0,
                                     "background only", True)
    ok("without an audio asset the treatment's soundscape is kept",
       "kept as written" in rep_e,
       [l for l in rep_e.splitlines() if l.startswith("soundscape")][:1])

finally:
    shutil.rmtree(WORK, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
