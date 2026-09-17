"""Cut one long H3 treatment into segment-sized groups of shots.

The prompt creator already writes `[Shot 1]`, `[Shot 2] At 00:04.500`, … and
enforces that Shot 1 carries no timestamp while later shots strictly increase
and stay inside the duration. That makes the shot markers a reliable cut list,
so choosing segment boundaries needs no model at all — just arithmetic.

The one rule that matters: **never split a shot.** A shot is the smallest unit
H3 was told to render as one continuous idea; cutting through the middle of one
produces two fragments that each describe half a movement.

Pure functions, no ComfyUI imports.
"""

import re

from . import vocals

SECTION_KEYS = (
    "subject_definitions", "summary", "retention_analysis",
    "detailed_description", "overall_soundscape", "non_diegetic_music",
    "integrated_multimodal_description",
)

# The guide writes "[Shot 2] At 00:04.500", but real treatments come back with
# the "At" dropped, or replaced by a dash, an @ or a bracket:
#
#     [Shot 2] At 00:05.500 - ...      [Shot 2] 00:05.500 — ...
#     [Shot 2] @ 00:05.500 ...         [Shot 2] (00:05.500) ...
#
# Requiring the literal "At" meant a whole treatment parsed with every shot at
# 0.000, which collapses every segment to the half-second fallback in
# parse_shots and hands the last shot the entire runtime. The MM:SS colon is
# what makes this safe to loosen: descriptive text that opens a shot ("2d,
# illustration art style", "100mm macro", "24mm from ground") has no colon in
# that position, so it cannot be mistaken for a timestamp.
SHOT_RE = re.compile(
    r"\[\s*Shot\s+(\d+)\s*\]"
    r"\s*(?:at|@|[-‒–—―]|\()?"
    r"\s*(?:(\d{1,2}):(\d{1,2}(?:\.\d+)?))?",
    re.IGNORECASE)

# The guide's style vocabulary, plus the handful of near-synonyms models reach
# for. Used only to lift the prefix off Shot 1 so it can be repeated verbatim
# in every segment.
STYLE_WORDS = (
    "cinematic", "live-action", "live action", "2d-animated", "2d animated",
    "3d cg", "3d-cg", "3d animated", "claymation", "watercolor", "watercolour",
    "vintage film", "anime", "stop-motion", "stop motion", "photorealistic",
    "documentary", "2d illustration", "illustrated", "hand-drawn",
)


def parse_sections(prompt_text):
    """Split a rendered H3 prompt back into its named sections.

    Falls back to putting everything in detailed_description when the text has
    no section headers at all, so a hand-written prompt still splits.
    """
    text = (prompt_text or "").replace("\r\n", "\n")
    pattern = re.compile(
        r"^(%s)\s*:\s*$" % "|".join(SECTION_KEYS), re.IGNORECASE | re.MULTILINE)

    marks = list(pattern.finditer(text))
    if not marks:
        return {"detailed_description": text.strip()}

    sections = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections[m.group(1).lower()] = text[m.end():end].strip()
    if "detailed_description" not in sections and \
            "integrated_multimodal_description" in sections:
        sections["detailed_description"] = \
            sections["integrated_multimodal_description"]
    return sections


def _is_heading(text, match):
    """A marker that starts a shot, as opposed to one that cites a shot.

    "matching the framing of [Shot 1]" is a citation. Counting it as a shot
    cut the treatment there and gave the real Shot 1's body to a clip of half
    a second. A marker with a timestamp is always a heading, wherever it sits;
    one without must start the text, a line or a sentence — the same rule the
    prompt creator's timing pass uses.
    """
    if match.group(2) is not None:
        return True
    before = text[:match.start()].rstrip(" \t")
    return not before or before[-1] in ".!?;:\n"


def parse_shots(description, total_duration):
    """Return [{number, start, end, text}] from a description body."""
    text = description or ""
    matches = [m for m in SHOT_RE.finditer(text) if _is_heading(text, m)]
    if not matches:
        return []

    shots = []
    for i, m in enumerate(matches):
        start = 0.0
        if m.group(2) is not None:
            start = int(m.group(2)) * 60 + float(m.group(3))
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(description)
        shots.append({
            "number": int(m.group(1)),
            "start": start,
            "text": description[m.start():body_end].strip(),
        })

    shots[0]["start"] = 0.0
    total = float(total_duration)
    for i, shot in enumerate(shots):
        shot["end"] = shots[i + 1]["start"] if i + 1 < len(shots) else total
        if shot["end"] <= shot["start"]:
            # a model that emitted equal or backwards times; give it something
            shot["end"] = min(total, shot["start"] + 0.5)
    shots[-1]["end"] = max(shots[-1]["end"], total)
    return shots


def style_prefix(description):
    """The style clause at the head of Shot 1, to repeat in every segment."""
    shots = SHOT_RE.search(description or "")
    if not shots:
        return ""
    body = description[shots.end():].lstrip()
    parts = [p.strip() for p in body.split(",")]
    kept = []
    for part in parts[:4]:
        low = part.lower()
        if any(word in low for word in STYLE_WORDS) and len(part) < 40:
            kept.append(part)
        elif kept:
            break
        else:
            break
    return ", ".join(kept)


def group_shots(shots, max_seconds, min_seconds=2.0, max_shots=4,
                total_duration=None):
    """Group consecutive shots into segments that fit the H3 length limit.

    Greedy and left-to-right, which keeps it predictable: a shot joins the
    current segment while the segment stays under ``max_seconds`` and under
    ``max_shots``. A single shot longer than the limit becomes its own segment
    and is flagged — it cannot be split without breaking the shot.
    """
    if not shots:
        return [], ["the treatment has no [Shot N] markers to cut on"]

    groups, warnings = [], []
    current = []

    def span(group):
        return group[-1]["end"] - group[0]["start"]

    for shot in shots:
        if not current:
            current = [shot]
            if shot["end"] - shot["start"] > max_seconds:
                warnings.append(
                    "shot %d runs %.2fs, past the %.2fs limit — kept whole as "
                    "its own segment; shorten it in the treatment or let H3 "
                    "compress it" % (shot["number"],
                                     shot["end"] - shot["start"], max_seconds))
            continue
        trial = current + [shot]
        if span(trial) <= max_seconds and len(trial) <= max_shots:
            current = trial
        else:
            groups.append(current)
            current = [shot]
    if current:
        groups.append(current)

    # A short tail reads as a mistake rather than a beat — fold it back in,
    # but never past the shots-per-segment limit: at 1 shot per segment the
    # whole point is that a shot keeps its own clip.
    if len(groups) > 1 and span(groups[-1]) < min_seconds:
        merged = groups[-2] + groups[-1]
        if span(merged) <= max_seconds and len(merged) <= max_shots:
            groups[-2:] = [merged]
        else:
            warnings.append(
                "the last segment is only %.2fs and will not fit into the one "
                "before it" % span(groups[-1]))

    for group in groups:
        if span(group) < min_seconds:
            warnings.append("segment starting at shot %d is only %.2fs"
                            % (group[0]["number"], span(group)))

    return groups, warnings


def _read_vocal_timeline(context, total_duration, warnings):
    """Lift the prompt creator's vocal timeline into the shared context.

    A timeline that stops short of the video is reported rather than silently
    treated as instrumental past its end, which would stop the singer in the
    last segments with nothing said.
    """
    label, sections = vocals.parse(context.get("overall_soundscape", ""))
    context["vocal_timeline"] = sections
    context["vocal_label"] = label
    if sections:
        covered = sections[-1]["end"]
        if covered + 0.5 < float(total_duration):
            warnings.append(
                "the vocal timeline covers %.1fs but the video is %.1fs; "
                "segments past %.1fs are treated as instrumental. Re-run the "
                "prompt creator with the full-length audio connected."
                % (covered, float(total_duration), covered))


def fit_runtime(shots, total_duration, max_seconds, min_seconds=2.0):
    """Make a shot list fit the video's real length and segment length.

    Two failures, one symptom: a single segment tens of seconds long at the end
    of the timeline.

    - A treatment that stops short. An 86-second music video came back from the
      prompt creator with ten shots, the last at 00:27.000. The last shot has
      no next timestamp, so it was stretched to the end of the video: 59
      seconds of one shot, which no clip can render.
    - A shot longer than a segment. "Never split a shot" is right for a shot H3
      can render in one clip, and wrong for one it cannot.

    The shot list's own pace decides where the treatment really ends: its last
    shot is given the median length of the others. Whatever lies beyond that is
    returned as continuation spans, each no longer than ``max_seconds``, for
    the caller to fill. Any shot still longer than ``max_seconds`` is split into
    equal parts that fit.

    Returns ``(shots, continuation, notes)``.
    """
    import math

    total = float(total_duration)
    cap = float(max_seconds)
    shots = [dict(s) for s in shots]
    notes, continuation = [], []
    if not shots or cap <= 0:
        return shots, continuation, notes

    last = shots[-1]
    if len(shots) > 1:
        lengths = sorted(s["end"] - s["start"] for s in shots[:-1])
        natural = lengths[len(lengths) // 2]
    else:
        natural = cap
    natural = min(cap, max(1.0, natural))
    covered_to = round(min(total, last["start"] + natural), 3)
    gap = total - covered_to
    if gap > max(float(min_seconds), 0.5):
        last["end"] = covered_to
        count = max(1, int(math.ceil(gap / cap - 1e-9)))
        step = gap / count
        for k in range(count):
            # Whole milliseconds, so the rounded segment lengths still add up
            # to the video exactly rather than drifting a millisecond a cut.
            start = round(covered_to + k * step, 3)
            end = total if k == count - 1 else round(covered_to + (k + 1) * step, 3)
            continuation.append({
                "start": start, "end": end, "part": k + 1, "of": count,
                "after_shot": last["number"],
                "after_text": (last.get("text") or "")[:400],
            })
        notes.append(
            "THE TREATMENT ENDS EARLY. Its shots cover 0.00s to %.2fs of a "
            "%.2fs video; its last shot is [Shot %d] at %.2fs. The remaining "
            "%.2fs is filled with %d segment(s) of up to %.2fs that continue "
            "from it, rather than stretching that one shot to the end. Re-run "
            "the prompt creator if you want those written from the treatment."
            % (covered_to, total, last["number"], last["start"], gap, count,
               cap))

    fitted = []
    for shot in shots:
        length = shot["end"] - shot["start"]
        if length <= cap + 1e-6:
            fitted.append(shot)
            continue
        count = int(math.ceil(length / cap - 1e-9))
        step = length / count
        for k in range(count):
            part = dict(shot)
            part["start"] = round(shot["start"] + k * step, 3)
            part["end"] = (shot["end"] if k == count - 1
                           else round(shot["start"] + (k + 1) * step, 3))
            part["part"], part["of"] = k + 1, count
            fitted.append(part)
        notes.append(
            "[Shot %d] runs %.2fs, longer than the %.2fs segment length, so it "
            "is split into %d segments of %.2fs that each cover part of it."
            % (shot["number"], length, cap, count, step))
    return fitted, continuation, notes


def split_treatment(prompt_text, total_duration, max_seconds,
                    min_seconds=2.0, max_shots=4, enforce_runtime=False):
    """Full pass: text in, segment descriptors + shared context out.

    ``enforce_runtime`` (the Segment Prompter) guarantees no segment is longer
    than ``max_seconds`` and that the whole ``total_duration`` is planned, even
    when the treatment stops short. Off, the behaviour is unchanged.
    """
    sections = parse_sections(prompt_text)
    description = sections.get("detailed_description", "")
    shots = parse_shots(description, total_duration)
    continuation, runtime_notes = [], []
    if enforce_runtime:
        shots, continuation, runtime_notes = fit_runtime(
            shots, total_duration, max_seconds, min_seconds)
    groups, warnings = group_shots(shots, max_seconds, min_seconds, max_shots,
                                   total_duration)
    warnings = runtime_notes + warnings

    # Timestamps that all read 0.000 are not a plan, they are a parse failure:
    # every segment collapses to the half-second fallback and the last shot
    # inherits the whole runtime. Silently rendering that wastes a queue, so
    # say it at the top of the report rather than leaving it to be inferred
    # from a strip of 0.5s cards.
    timed = [s for s in shots if s["start"] > 0]
    if len(shots) > 1 and not timed:
        warnings.insert(0,
            "NO TIMESTAMPS PARSED. All %d shots read as starting at 0.000, so "
            "the durations below are fallbacks, not a plan. The treatment's "
            "shot markers need a time, as \"[Shot 2] At 00:05.500\" or "
            "\"[Shot 2] 00:05.500\". Re-run the prompt creator at the full "
            "video duration, or plan from `idea` instead." % len(shots))

    prefix = style_prefix(description)
    if not prefix:
        warnings.append(
            "no style prefix found at the head of Shot 1 — set one on the "
            "splitter, or every segment may drift to a different look")

    segments = []
    for i, group in enumerate(groups):
        start, end = group[0]["start"], group[-1]["end"]
        segments.append({
            "id": "seg_%02d" % (i + 1),
            "beat": ("shot %d" % group[0]["number"] if len(group) == 1
                     else "shots %d-%d" % (group[0]["number"],
                                           group[-1]["number"])),
            "target_duration": round(end - start, 3),
            "link": "cut",
            "prompt": "",
            # The treatment's clock and the reference track's clock are the
            # same clock, so a segment's position in the video is also its
            # window in the song. Track Slice reads this.
            "audio_start": round(start, 3),
            "source": {
                "start": round(start, 3),
                "end": round(end, 3),
                "shot_numbers": [s["number"] for s in group],
                # timestamps rebased so the prompter never has to subtract
                "shots": [dict({
                    "number": s["number"],
                    "at": round(s["start"] - start, 3),
                    "text": s["text"],
                }, **({"part": s["part"], "of": s["of"]} if s.get("of") else {}))
                    for s in group],
            },
        })

    # Time the treatment never reached. Each span is its own segment, so the
    # writer continues the piece instead of re-describing the last shot.
    for piece in continuation:
        index = len(segments)
        segments.append({
            "id": "seg_%02d" % (index + 1),
            "beat": "continues after shot %d (%d/%d)"
                    % (piece["after_shot"], piece["part"], piece["of"]),
            "target_duration": round(piece["end"] - piece["start"], 3),
            "link": "cut",
            "prompt": "",
            "audio_start": round(piece["start"], 3),
            "source": {
                "start": round(piece["start"], 3),
                "end": round(piece["end"], 3),
                "shot_numbers": [],
                "shots": [],
                "continuation": {
                    "part": piece["part"], "of": piece["of"],
                    "after_shot": piece["after_shot"],
                    "after_text": piece["after_text"],
                },
            },
        })

    context = {
        "style_prefix": prefix,
        "total_duration": float(total_duration),
        "shot_count": len(shots),
        "subject_definitions": sections.get("subject_definitions", ""),
        "summary": sections.get("summary", ""),
        "retention_analysis": sections.get("retention_analysis", ""),
        "overall_soundscape": sections.get("overall_soundscape", ""),
        "non_diegetic_music": sections.get("non_diegetic_music", ""),
    }
    _read_vocal_timeline(context, total_duration, warnings)
    return segments, context, warnings


def split_on_cuts(prompt_text, cuts, total_duration):
    """Cut a treatment on boundaries someone else decided.

    Used by the Beat Map path: the music says where the cuts fall, and each
    shot goes to whichever segment its own timestamp lands in. A segment that
    catches no shot of its own still exists — a gap in the shot list is not a
    gap in the video — and reuses the nearest preceding shot for its text.
    """
    sections = parse_sections(prompt_text)
    description = sections.get("detailed_description", "")
    shots = parse_shots(description, total_duration)
    warnings = []
    if not shots:
        return [], sections, ["the treatment has no [Shot N] markers to cut on"]

    segments, placed = [], set()
    for i, cut in enumerate(cuts):
        start, end = float(cut["start"]), float(cut["end"])
        inside = [s for s in shots if start - 1e-6 <= s["start"] < end - 1e-6]
        placed.update(s["number"] for s in inside)
        if not inside:
            earlier = [s for s in shots if s["start"] <= start + 1e-6]
            inside = earlier[-1:] or shots[:1]
            warnings.append(
                "seg_%02d at %.2fs covers no shot of its own; it reuses the "
                "one before it" % (i + 1, start))
        segments.append({
            "id": "seg_%02d" % (i + 1),
            "beat": cut.get("label") or ("%d bar(s)" % cut.get("bars", 0)),
            "target_duration": round(end - start, 3),
            "link": "cut",
            "prompt": "",
            "audio_start": round(start, 3),
            "source": {
                "start": round(start, 3),
                "end": round(end, 3),
                "shot_numbers": [s["number"] for s in inside],
                "shots": [{"number": s["number"],
                           "at": round(max(0.0, s["start"] - start), 3),
                           "text": s["text"]} for s in inside],
            },
        })

    # A shot outside every cut is written and never rendered. It happened
    # for real: the beat grid started at the first downbeat, 2.554s in, and
    # the opening shot simply vanished from the video with nothing said.
    lost = [s for s in shots if s["number"] not in placed]
    if lost and segments:
        first = float(cuts[0]["start"])
        last = float(cuts[-1]["end"])
        warnings.append(
            "%d shot(s) fall outside every cut and will NOT appear in the "
            "video: %s. The cuts cover %.2fs to %.2fs of the track, and "
            "those shots sit outside it — %s"
            % (len(lost),
               ", ".join("Shot %d at %.2fs" % (s["number"], s["start"])
                         for s in lost[:8]),
               first, last,
               "turn on 'cover whole track' on the Beat Map, or move the "
               "shot's timestamp inside that range"))

    prefix = style_prefix(description)
    if not prefix:
        warnings.append(
            "no style prefix found at the head of Shot 1 — set one on the "
            "slicer, or every segment may drift to a different look")
    context = {
        "style_prefix": prefix,
        "total_duration": float(total_duration),
        "shot_count": len(shots),
        "subject_definitions": sections.get("subject_definitions", ""),
        "summary": sections.get("summary", ""),
        "retention_analysis": sections.get("retention_analysis", ""),
        "overall_soundscape": sections.get("overall_soundscape", ""),
        "non_diegetic_music": sections.get("non_diegetic_music", ""),
    }
    _read_vocal_timeline(context, total_duration, warnings)
    return segments, context, warnings
