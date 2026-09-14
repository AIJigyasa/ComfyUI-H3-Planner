"""H3 Story Planner — the whole timeline from one brief, in one model call.

Why this exists alongside the Segment Prompter rather than replacing it.

The Segment Prompter writes each segment in its own model call, and it is told
to make each one *differ* from the last. For a music video, where consecutive
scenes are deliberately unrelated, that is exactly right. For an ad or a story
it is exactly wrong: nothing carries wardrobe, location, time of day, or where
the subject was standing, so every shot re-invents the world and the stitched
result reads as a slideshow.

This node inverts that. Continuity stops being something plumbed between calls
and becomes a property of a single generation:

  1. ONE call writes the bible and the beat sheet — style prefix, the frozen
     world block, the canonical cast wording, the scenes, and for every segment
     its duration, its action, and the physical state it hands to the next.
  2. The prose for each segment is expanded in windows, each window seeing the
     previous one's closing state, so a long piece never needs a single
     enormous reply.
  3. Assembly is deterministic. Durations snap to the frame ladder, the style
     prefix and cast wording are copied verbatim rather than requested, and
     every enforcement the Segment Prompter learned is reapplied here.

For a 30-second ad that is one call instead of the twenty-one the old path
needed (one treatment plus one per segment).

The Segment Prompter is untouched. Both write the same H3_TIMELINE, so the
render and stitch graphs do not care which one produced it.
"""

import math
import os
import re
import time

from . import engine, ladder, splitter, store
from .nodes_plan import CATEGORY, cited_tags
from .nodes_prompt import (FORMAT_GUIDANCE, FORMATS, AUDIO_ROLES, SECTIONS,
                           DIALOGUE_MODES,
                           audio_system_block, bind_tags,
                           _allowed_tags, _cast_block, _hash, asset_names,
                           canonical_subjects, enforce_audio_exact,
                           renumber_shots, strip_asset_names,
                           strip_unknown_tags)

# --------------------------------------------------------------------------
# the two model calls
# --------------------------------------------------------------------------

BEATS_SYSTEM = """
You are the director of a short video. You are writing the PLAN for the whole
piece in one pass: the look, the cast, the scenes, and every shot in order.

Everything you write here is binding on the segments that follow, so decide it
once and decide it properly.

STYLE PREFIX. One short phrase naming format, lens and grade — for example
"Cinematic 35mm anamorphic, warm daylight grade". It is copied verbatim to the
front of every segment, so it must be a fragment, not a sentence, and must
contain no story detail.

WORLD. The facts that must not drift: location, time of day, weather, wardrobe,
palette, lighting. Write it as plain declarative statements. This is the block
every segment is held to, so put anything that would embarrass you if it
changed halfway through the video in here.

  - YOU INVENT THE WORLD. Take the location from the brief when the brief names
    one. Otherwise choose one yourself that serves the piece, and commit to it.
  - NEVER take the world from a reference photograph. The backdrop a subject
    was photographed against is not the setting of this video, and a video set
    in a featureless studio because that is where the casting photo was taken
    is a failure, not a decision. If the brief does not name a location and you
    find yourself describing a plain studio backdrop, you have read it off the
    reference: throw it away and choose a real place.

SUBJECT DEFINITIONS. Define each cast tag once, in the wording that will be
reused for the whole video. One line per tag, beginning with the tag itself.
Describe only what is visible and permanent — build, face, hair, wardrobe. Do
not describe action here.

  - WHERE A REFERENCE IMAGE IS SUPPLIED, THAT IMAGE IS THE TRUTH ABOUT THE
    SUBJECT. The images are attached; look at them and describe the person or
    object that is actually there rather than inventing one from the brief. A
    supplied photograph of one person beside a written description of a
    different one is a contradiction, and the renderer resolves contradictions
    by blending them into someone who is neither.
  - READ THE SUBJECT OUT OF THE IMAGE AND NOTHING ELSE. A reference photograph
    is a casting card, not a location scout. Its backdrop, its lighting, its
    studio sweep, the surface an object is standing on, the way it was shot —
    none of that is part of the video and none of it goes in your answer. A
    white studio wall behind someone means they were photographed against a
    white wall; it does not mean the advert is set in a white room. Describe
    build, face, hair, skin, wardrobe and, for a product, its form, material,
    label and finish. Stop there.
  - If a cast entry carries a note, that note is the author's own description
    and outranks anything you would write. Keep its wording.
  - If you genuinely cannot see an image for a tag, say only what the brief
    and the note establish, and leave appearance alone rather than guessing.

SCENES. A scene is a run of shots sharing one location, one time of day and one
continuous stretch of action. Number them from 1. A new location, or a jump in
time, is a new scene.

USE THE TAGS. Every cast member listed below has a tag, and that tag is how the
renderer knows which supplied image to use. Refer to them by tag EVERYWHERE in
this plan — in action, in opens_from, in ends_with — never by a pronoun or a
description. "<Subject 1> lifts <Subject 2> to eye level" is correct; "she
lifts the bottle" is not, because nothing connects "she" to the photograph that
was supplied. A product is a cast member like any other: if it has a tag, name
it by that tag every time it is on screen.

SEGMENTS. Each segment is ONE rendered clip and may contain several shots
inside it — a clip is allowed to cut internally, and for a fast-cutting piece
it should. For every segment give:
  - seconds: the clip length, chosen ONLY from the allowed lengths listed below
  - scene: which scene it belongs to
  - action: one sentence, what actually happens on screen
  - opens_from: the physical state at the first frame — where the subject is,
    their posture, what is in their hands, where the camera sits. For the first
    segment of a scene, describe it fresh. Otherwise it must match the previous
    segment's ends_with exactly.
  - ends_with: the physical state at the last frame, in the same terms. The
    next segment opens on this, so be concrete. "She has stood up and is
    holding the cup in her right hand, camera now at her eye line" is useful;
    "the mood shifts" is not.
  - dialogue: the exact words spoken on camera in this segment, or "" when
    nobody speaks. See below.

VARY THE SHOT. A run of segments all framed the same way is the commonest way
a machine-written advert gives itself away: five mid-shots at eye level of one
person holding one object is not coverage, it is the same shot five times.
Across the piece you need real variety of SCALE, HEIGHT and MOVEMENT.

  - Open on something that establishes where we are — a wide, or a detail so
    close it is abstract. Not a mid-shot of the subject standing there.
  - Somewhere in the middle, at least one shot with NO PERSON in it: the
    product alone, a macro on its surface, material or lettering, light moving
    across it. This is the shot that makes a product look worth buying, and it
    is the one a plan written on autopilot always omits.
  - Change the camera height and the lens between segments. Eye level for every
    shot is a default, not a choice. Get low, get overhead, get close.
  - Give the camera something to do — a push, a rise, a lateral track, a rack
    focus — and do not repeat the same move twice in a row.
  - The closing shot is the hero. It should be the most composed frame in the
    piece, and it should not be another mid-shot of someone holding the thing.

THE AIRLOCK. When a segment continues directly from the one before it rather
than cutting away, it must not change and continue at the same instant. The
renderer is handed the previous clip's closing frames, and a prompt that opens
on a different arrangement of people reads as an ADDITION to them, not a
replacement: ask for a two-shot over a close-up and you get all three people.

So a continuing segment's action begins by HOLDING the state in opens_from for
a moment — roughly a fifth of the clip, and the exact length is given to you
per segment — and only then moves to whatever is new. Give that hold something
to do: a breath, a weight shift, an eyeline change. A held framing with nothing
happening renders as a literal freeze and reads as the video having stalled.
The camera holds still; the performer does not. Keep the hold free of dialogue.

DIALOGUE. If the brief asks for anyone to speak, you must write the actual
lines here — not a description of them. "She talks about the scent" is useless;
the line she says is what gets rendered.

  - Write what is said, verbatim, as it should be heard. No stage directions,
    no speaker name, no quotation marks: just the words.
  - Keep it speakable inside the segment's length. Roughly two and a half words
    per second is comfortable, so a 7-second shot holds about 17 words. Going
    over means the line is cut off mid-word when the clip ends.
  - Write a script, not a set of slogans: the lines run consecutively across
    the segments and must read as one continuous piece of writing, each picking
    up where the last left off.
  - Leave it "" for a shot that is picture only. Not every shot needs a line,
    and a hero shot at the end is usually stronger with a single short one or
    none at all.

The segment durations must add up to approximately the total duration given.
Use the whole range: a beat that needs four seconds should not be stretched to
ten, and a held moment should not be chopped.
""".strip()

PROSE_SYSTEM = """
You are writing the finished MiniMax H3 prompts for a run of consecutive
segments in one video. Follow the supplied Full-Reference Mode guide exactly,
with these additional constraints, which override anything assuming you are
writing a whole film.

EACH SEGMENT IS A COMPLETE, STANDALONE H3 PROMPT. It is rendered on its own,
with no knowledge of any other segment, so every section is written in full for
every segment. Return them in the same order they are given to you.

ONE SEGMENT PER OBJECT. THIS IS THE RULE THAT MATTERS MOST.

Each object you return describes ONE clip, a few seconds long — only that
segment's own action, only that segment's own line. You are given the other
segments so you know what comes before and after; they are CONTEXT, NOT
CONTENT. Never write the whole piece into one object.

  - Object 1 contains segment 1's action and nothing else. Object 2 contains
    segment 2's action and nothing else. And so on.
  - A segment's detailed_description must never contain another segment's
    spoken line. If a line is listed under segment 3, it appears in object 3
    and in no other.
  - Two objects that read alike are a failure. If your objects are
    interchangeable, you have written the same clip five times and five
    identical videos will be rendered.
  - Respect the SHOT BUDGET given for each segment. A five-second clip holds
    one or two shots, not the whole advert. Fitting eight shots into six
    seconds does not make a fast-paced film; it makes an unreadable one.

CONTINUITY IS THE POINT. These segments are consecutive shots of one continuous
piece. Each one opens on the state the previous one ended in — that state is
given to you explicitly. Honour the world block exactly: the location, the
light, the wardrobe and the time of day do not drift between segments. Do not
recap earlier segments and do not foreshadow later ones; write only what is on
screen now.

STYLE PREFIX. The supplied style prefix appears verbatim at the very start of
[Shot 1] of every segment, before anything else.

TIMING. Every segment starts at 0.000 and its timestamps are relative to itself.
[Shot 1] carries NO timestamp. Later shots are "[Shot N] At MM:SS.mmm", strictly
increasing, all strictly less than that segment's rendered duration. Number the
shots from 1 within each segment.

HOLD TAIL. The rendered duration is slightly longer than the planned duration.
All essential action completes by the planned duration; the remainder is a hold
on the final composition. Never begin a new action inside the hold.

INTERNAL CUTS. A segment may contain several shots. Cut inside a segment when
the pace calls for it — that is cheaper and more continuous than splitting into
more clips. Keep eyelines and screen direction consistent across those cuts.

REFERENCES. Cite only the tags listed in the supplied cast. Never invent a tag.
subject_definitions lists only the tags that segment actually cites. Do not
write <Video N> unless a video asset is listed: pacing and cut structure are
things you decide, not things a reference supplies.

ASSET NAMES. Refer to every asset by its tag alone — never a file name, a track
title or a cast key, anywhere, in any section.

THE AIRLOCK. A segment marked CONTINUES BELOW is handed the previous clip's
closing frames by the renderer, so it cannot change and continue at the same
instant — a prompt opening on a different arrangement of people is read as an
addition to what is already there, not a replacement.

Write [Shot 1] of such a segment as a hold on the state in "opens on", ending
at the timestamp given for that segment, carrying no dialogue. The camera does
not move. The performer does: a breath, a weight shift, an eyeline change,
because a truly still hold renders as a freeze and looks like the video
stalled. Only from [Shot 2] does anything new happen, and any spoken line
belongs there or later.

DIALOGUE. Where a segment below carries a spoken line, it MUST appear in
detailed_description in H3's dialogue form, or nothing will be said on screen:

    <Subject 1> (S1) turns to the lens and says, <d>[English] Your morning,
    bottled.</d>

  - The words go inside <d>...</d>, opened with the language in square
    brackets. Nothing else goes inside those tags.
  - Use the supplied line verbatim. Do not rewrite, shorten or embellish it.
  - Speaker IDs are (S1), (S2), … and are global: the same person keeps the
    same ID in every segment. Two people speaking together are (S1,S2).
  - Keep BOTH labels when a defined subject speaks: the reference tag and the
    speaker ID, as above.
  - Establish who is speaking outside the <d> tags, never inside them.
  - Never write a speaker ID in retention_analysis.
  - Say what the mouth is doing as the line lands — that is what makes the
    performance visible rather than merely audible.
""".strip()


VOICE_SAMPLE = "voice sample (clone the timbre)"

# The Story Planner's own list. The other two nodes keep the three roles they
# had: a voice sample changes what the whole audio enforcement is FOR, and
# quietly handing that to the Segment Prompter would rewrite music videos.
STORY_AUDIO_ROLES = tuple(AUDIO_ROLES) + (VOICE_SAMPLE,)


def speaker_id(index):
    return "(S%d)" % (index + 1)


def cast_voices(cast, chosen=None):
    """Which audio reference supplies which subject's voice.

    Returns ``[{"audio": "<Audio 1>", "subject": "<Subject 2>",
    "speaker": "(S1)"}]``.

    ``chosen`` is the director's own casting from the beat sheet, which is the
    only thing that can read "use <audio 1> for the male character" out of a
    brief. It is a model reply, so every pair is checked against the cast and
    anything unusable falls back to pairing in order.
    """
    members = (cast or {}).get("members") or []
    audios = [m["tag"] for m in members if m.get("kind") == "Audio"]
    subjects = [m["tag"] for m in members if m.get("kind") == "Subject"]
    if not audios:
        return []

    ids = {tag: speaker_id(i) for i, tag in enumerate(subjects)}
    picked, used = {}, set()
    for row in (chosen or []):
        if not isinstance(row, dict):
            continue
        audio = str(row.get("audio", "")).strip()
        subject = str(row.get("subject", "")).strip()
        if audio in audios and subject in subjects and audio not in picked:
            picked[audio] = subject
            used.add(subject)

    # Anything the director did not cast, or cast onto a tag that does not
    # exist, is paired with the first subject still without a voice.
    spare = [s for s in subjects if s not in used] or subjects
    for audio in audios:
        if audio in picked:
            continue
        picked[audio] = spare.pop(0) if spare else (subjects[0] if subjects else "")

    return [{"audio": a, "subject": picked[a],
             "speaker": ids.get(picked[a], speaker_id(0))}
            for a in audios]


def voice_line(voices):
    return ", ".join("%s is the voice of %s %s"
                     % (v["audio"], v["subject"], v["speaker"])
                     for v in voices if v.get("subject"))


def voice_system_block(voices):
    """What the model must understand about a voice reference.

    The opposite instruction to AUDIO_EXACT, and it has to be said that
    plainly. Under the other three roles the supplied audio IS the soundtrack
    and no new sound may be described. Here the audio is a timbre to imitate
    and the dialogue is generated, so a prompt that says "all audio comes from
    <Audio 1>, used exactly as supplied" forbids the very speech being asked
    for.
    """
    if not voices:
        return ""
    return "\n".join([
        "THE AUDIO ASSETS ARE VOICE SAMPLES, NOT A SOUNDTRACK.",
        "",
        "Each one supplies a vocal timbre for one character. Its contents are "
        "NOT reproduced: the words come from the script you write, spoken in "
        "that voice.",
        "",
        "CASTING, binding on every segment:",
    ] + ["  %s supplies the voice of %s, who speaks as %s"
         % (v["audio"], v["subject"], v["speaker"])
         for v in voices if v.get("subject")] + [
        "",
        "- Never write that the audio is copied, reused, played, or used "
        "exactly as supplied. It is a reference for timbre only.",
        "- Every spoken line carries both labels, in this order: "
        "<Subject N> (Sx) says, <d>[English] the words</d>.",
        "- A speaker keeps the same (Sx) for the whole video. Two characters "
        "never share one.",
        "- Describe the performance as picture: the mouth matching the words, "
        "the jaw, the breath, where the eyes go. That is what makes the "
        "speech visible rather than merely audible.",
        "- Do not invent music. If the brief does not ask for a score, there "
        "is none.",
    ])


def voice_soundscape(voices, speaks=True):
    """overall_soundscape under a voice sample.

    Two forms, because a segment with no line in it is not a quieter version
    of a segment with one. Written unconditionally, the speaking form reached
    the two silent segments of a five-part advert and H3 did as it was told:
    the characters mouthed words that were never in the script.
    """
    cast_line = voice_line(voices)
    if not speaks:
        return ("Nobody speaks in this segment and no mouth moves to words. "
                "The supplied audio provides vocal timbre for other parts of "
                "the piece and is not heard here (%s). Room tone appropriate "
                "to the location, and nothing else — no speech, no score, no "
                "added ambience." % cast_line)
    return ("The dialogue in this segment is spoken on camera by the subjects "
            "named, and their mouths match the words throughout. The supplied "
            "audio provides vocal timbre only and its own content is not "
            "reproduced: %s. Room tone appropriate to the location, and "
            "nothing else — no score, no added ambience." % cast_line)


def voice_music_line():
    return ("N/A - there is no scored music; the audio references supply "
            "vocal timbre only.")


def enforce_voice_reference(prompt, voices, speaks=True):
    """Make the two sound sections describe generated speech, not a replay.

    The counterpart to enforce_audio_exact, and it exists because that
    function's line ("All audio comes from <Audio 1> and is used exactly as
    supplied. No additional sound is present.") reached every segment of an
    advert whose whole point was two characters talking. Any line H3 would
    speak is "additional sound", so the prompt forbade the thing it asked for.
    """
    changed = []
    if not voices:
        return prompt, changed

    exact = voice_soundscape(voices, speaks)
    if prompt.get("overall_soundscape", "").strip() != exact:
        prompt["overall_soundscape"] = exact
        changed.append("overall_soundscape")

    music = voice_music_line()
    if prompt.get("non_diegetic_music", "").strip() != music:
        prompt["non_diegetic_music"] = music
        changed.append("non_diegetic_music")

    # One retention line per audio tag, not just the first. The H3 guide allows
    # fully_copy, partially_copy, reference and weak_reference for audio; a
    # voice sample is `reference`, which it defines as guiding the target
    # without copying unknown content. fully_copy means replay the file.
    retention = prompt.get("retention_analysis", "")
    for v in voices:
        canonical = (
            ("%s: reference - supplies the vocal timbre of %s %s; its content "
             "is not reproduced." % (v["audio"], v["subject"], v["speaker"]))
            if speaks else
            # weak_reference is the guide's marker for an asset that is
            # supplied but not active here, and it is the honest one for a
            # segment in which nobody says anything.
            ("%s: weak_reference - not heard in this part; nobody speaks."
             % v["audio"]))
        line = re.compile(r"[^.\n]*%s[^.\n]*\.?" % re.escape(v["audio"]))
        match = line.search(retention)
        if match:
            if match.group(0).strip() == canonical:
                continue
            retention = " ".join(filter(None, [
                retention[:match.start()].strip(),
                canonical,
                retention[match.end():].strip()]))
        else:
            retention = (retention.rstrip() + " " + canonical).strip()
        changed.append("audio retention -> reference")
    prompt["retention_analysis"] = retention
    return prompt, changed


# The director writes the world as a labelled block, because BEATS_SYSTEM asks
# for one. A real reply looks like this:
#
#   - Location: A sun-drenched corner cafe with exposed brick walls, vintage
#     coffee machines, and mismatched wooden tables
#   - Time of day: Late afternoon, golden hour light
#   - Lighting: Natural window light with subtle fill from overhead lamps
#
# Free prose happens too, so the first sentence is the fallback rather than an
# error: a world that will not parse must still reach the prompt.
WORLD_KEYS = ("location", "time of day", "weather", "wardrobe", "palette",
              "lighting")


def world_fields(world):
    """The world block as {label: value}, plus "" -> the whole thing."""
    text = (world or "").strip()
    if not text:
        return {}
    fields = {}
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        label = label.strip().lower()
        if label in WORLD_KEYS and value.strip():
            fields[label] = value.strip().rstrip(".")
    if not fields:
        first = re.split(r"(?<=[.!?])\s", text)[0].strip()
        fields[""] = first[:220]
    return fields


def world_sentence(fields, keys=("location", "time of day", "lighting")):
    """One sentence naming the place, stated identically in every segment.

    Identical wording is the entire mechanism. H3 renders each clip on its own,
    so two clips share a room only when they were told about it in the same
    words; "a cafe" and "a sun-drenched corner cafe with exposed brick" are two
    different rooms.
    """
    if not fields:
        return ""
    if "" in fields:
        return fields[""].rstrip(".") + "."
    place = fields.get("location")
    bits = []
    if place:
        bits.append("The whole piece takes place in one location: %s."
                    % place.rstrip("."))
    for label, key in (("Time of day", "time of day"), ("Light", "lighting")):
        value = fields.get(key)
        if value and key in keys:
            bits.append("%s: %s." % (label, value.rstrip(".")))
    if not bits:
        spare = [fields[k] for k in keys if fields.get(k)]
        return ("The whole piece takes place in %s." % "; ".join(spare)
                if spare else "")
    return " ".join(bits)


def world_clause(fields, limit=110):
    """The short form, for the head of a shot description."""
    if not fields:
        return ""
    head = fields.get("location") or fields.get("") or ""
    head = head.rstrip(".")
    if len(head) > limit:
        head = head[:limit].rsplit(" ", 1)[0]
    return head


def task_prefix(cast, audio_role, chained=False):
    """The square-bracketed task type the H3 guide requires on a summary.

    Its list is closed and the types are combined with " + ", never repeated.
    The mere presence of an asset does not create its type: reference images
    that only carry identity are reference generation, and a first frame
    handed over from the previous clip is keyframe completion.
    """
    members = (cast or {}).get("members") or []
    kinds = {m.get("kind") for m in members}
    types = []
    if chained:
        types.append("keyframe completion")
    if kinds & {"Subject", "Picture", "Video"}:
        types.append("reference generation")
    if "Audio" in kinds:
        types.append("audio reference" if audio_role == VOICE_SAMPLE
                     else "audio reuse")
    if not types:
        types.append("reference generation")
    return "[%s]" % " + ".join(types)


def build_summary(seg, index, total, fields, prefix_tag, format_name=""):
    """One segment's summary, written here rather than asked for.

    It was "N/A" in every segment of a five-part advert, because the prose call
    never asked for it, and it is the one section whose job is to say what the
    video IS. Written in code it also cannot drift: every segment gets the same
    sentence about the same room, which is what makes five clips read as one
    scene instead of five auditions for it.

    No reference labels: the guide forbids introducing one here, and this text
    is assembled without seeing which tags the description ended up citing.
    """
    bits = ["%s The target video is part %d of %d of one continuous %s, told "
            "in order and without a break in place or time."
            % (prefix_tag, index + 1, total,
               (format_name or "scene").replace("cinematic ad", "scene"))]
    place = world_sentence(fields)
    if place:
        bits.append(place)
    opens = (seg.get("opens_from") or "").strip().rstrip(".")
    ends = (seg.get("ends_with") or "").strip().rstrip(".")
    if opens:
        bits.append("This part opens on %s." % opens)
    if ends:
        bits.append("It ends with %s." % ends)
    if seg.get("link") == "continue":
        bits.append("Its first frame continues directly from the last frame "
                    "of the part before it.")
    return " ".join(bits)


def _content_words(text):
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def opening_landed(body, opens_from, need=0.45):
    """Did the first shot actually start where the last one finished?

    A token overlap rather than a phrase match, because the writer is supposed
    to re-word the handover, not paste it. Below the threshold the state is
    prepended instead: the instruction reached the writer and the writer was
    free to ignore it, which is the definition of something that has to be
    enforced in code.
    """
    wanted = _content_words(opens_from)
    if not wanted:
        return True
    first = splitter.SHOT_RE.search(body or "")
    head = body[first.end():] if first else (body or "")
    second = splitter.SHOT_RE.search(head)
    if second:
        head = head[:second.start()]
    return len(wanted & _content_words(head)) >= max(2, int(len(wanted) * need))


def insert_after_prefix(body, sentence, prefix=""):
    """Put a sentence at the head of [Shot 1], behind the style prefix.

    Behind it, never in front: the H3 guide puts the style clause at the very
    start of Shot 1, _force_prefix has just put it there, and refine checks it
    is still there. A sentence spliced in ahead of it moves the prefix and
    every one of those breaks at once.
    """
    marker = splitter.SHOT_RE.search(body or "")
    if not marker:
        return sentence + (body or "")
    head, rest = body[:marker.end()], body[marker.end():].lstrip()
    lead = (prefix or "").rstrip(" ,")
    if lead and rest.lower().startswith(lead.lower()):
        kept, rest = rest[:len(lead)], rest[len(lead):].lstrip(" ,")
        return "%s %s, %s%s" % (head.rstrip(), kept, sentence, rest)
    return "%s %s%s" % (head.rstrip(), sentence, rest)


def shot_spans(body):
    """[(start, end)] of every [Shot N] block in a description."""
    marks = list(splitter.SHOT_RE.finditer(body or ""))
    if not marks:
        return []
    edges = [m.start() for m in marks] + [len(body)]
    return [(edges[i], edges[i + 1]) for i in range(len(marks))]


def ending_landed(body, ends_with, need=0.45):
    """Does the LAST shot finish on the state the plan gave this segment?

    The mirror of opening_landed, and the one that was missing. A segment
    briefed to end with "she stands up from the table" was written with a
    second shot of her walking out of the door, which is the NEXT segment's
    job. The exit then rendered twice: once at the end of one clip and again
    at the start of the following one.
    """
    wanted = _content_words(ends_with)
    if not wanted:
        return True
    spans = shot_spans(body)
    tail = body[spans[-1][0]:spans[-1][1]] if spans else (body or "")
    return len(wanted & _content_words(tail)) >= max(2, int(len(wanted) * need))


def overran(body, ends_with, need=0.45):
    """The shot index where this segment's ending is described, if it is early.

    Returns the number of shots to keep, or 0 when nothing is wrong. Only
    reports an overrun it can prove: the planned ending is clearly described
    in an earlier shot AND shots follow it.
    """
    wanted = _content_words(ends_with)
    spans = shot_spans(body)
    if not wanted or len(spans) < 2 or ending_landed(body, ends_with, need):
        return 0
    threshold = max(2, int(len(wanted) * need))
    for i, (lo, hi) in enumerate(spans[:-1]):
        if len(wanted & _content_words(body[lo:hi])) >= threshold:
            return i + 1
    return 0


def trim_after_ending(body, ends_with):
    """Drop the shots that come after this segment's planned ending.

    Structural, not prose: whole [Shot N] blocks are removed, never words
    inside one. What they describe is the next segment's opening, so keeping
    them renders the same action twice and leaves the clip ending somewhere
    the next one does not begin. Returns (body, dropped).
    """
    keep = overran(body, ends_with)
    if not keep:
        return body, 0
    spans = shot_spans(body)
    return body[:spans[keep][0]].rstrip(), len(spans) - keep

def enforce_opening(body, opens_from, prefix=""):
    """Put the handover state at the head of [Shot 1] when it is missing."""
    state = (opens_from or "").strip().rstrip(".")
    if not state or opening_landed(body, state):
        return body
    return insert_after_prefix(body, "The shot opens on %s. " % state, prefix)


def anchor_world(body, clause, prefix=""):
    """Name the location in the description when the writer left it out.

    Segment two of a five-part cafe advert mentioned neither the cafe nor one
    detail of it, so H3 rendered a different room for it.
    """
    place = (clause or "").strip().rstrip(".")
    if not place:
        return body
    wanted = _content_words(place)
    if wanted and len(wanted & _content_words(body)) >= max(2, len(wanted) // 3):
        return body
    return insert_after_prefix(body, "The location is %s. " % place, prefix)


def beats_schema():
    return engine.json_schema({
        "style_prefix": {"type": "string"},
        "world": {"type": "string"},
        "subject_definitions": {"type": "string"},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seconds": {"type": "number"},
                    "scene": {"type": "integer"},
                    "scene_name": {"type": "string"},
                    "action": {"type": "string"},
                    "opens_from": {"type": "string"},
                    "ends_with": {"type": "string"},
                    "dialogue": {"type": "string"},
                    "speaker": {"type": "string"},
                },
                "required": ["seconds", "scene", "action", "ends_with"],
            },
        },
        "voices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "audio": {"type": "string"},
                    "subject": {"type": "string"},
                },
                "required": ["audio", "subject"],
            },
        },
    }, ["style_prefix", "world", "subject_definitions", "segments"])


def prose_schema():
    return engine.json_schema({
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {k: {"type": "string"} for k in SECTIONS},
                "required": list(SECTIONS),
            },
        },
    }, ["segments"])


# --------------------------------------------------------------------------
# deterministic duration planning
# --------------------------------------------------------------------------

def allowed_rungs(project, min_seconds, max_seconds):
    """The clip lengths H3 can actually render, inside the user's window.

    The frame ladder only accepts certain lengths, so there is no point letting
    a model pick 8.4 seconds: it would render 8.708 and the stitcher would throw
    the difference away. Handing it the real menu makes the plan free of waste.
    """
    table = ladder.rungs(project["fps"], max_seconds,
                         modulus=project["frame_modulus"],
                         remainder=project["frame_remainder"],
                         minimum=project["frame_minimum"])
    inside = [(f, s) for f, s in table if s >= min_seconds - 1e-6]
    if not inside:                       # window too narrow for any rung
        inside = table[-1:] or [(project["frame_minimum"],
                                 project["frame_minimum"] / float(project["fps"]))]
    return inside


def ladder_neighbours(project, low, high, spread=2):
    """The real clip lengths either side of a window, for an error message."""
    table = ladder.rungs(project["fps"], float(high) + 6.0,
                         modulus=project["frame_modulus"],
                         remainder=project["frame_remainder"],
                         minimum=project["frame_minimum"])
    seconds = [s for _, s in table]
    if not seconds:
        return []
    mid = min(range(len(seconds)), key=lambda i: abs(seconds[i] - float(low)))
    return seconds[max(0, mid - spread):mid + spread + 1]


def snap_to_rung(seconds, rungs_seconds):
    return min(rungs_seconds, key=lambda s: (abs(s - seconds), s))


def reconcile(wanted, total, rungs_seconds):
    """Snap every duration to a rung, then close the gap on the total.

    The model is asked for durations that sum to the runtime, and never quite
    manages it. Rather than trusting it or rescaling (which would land back off
    the ladder), each duration is snapped and then nudged one rung at a time,
    always taking the single move that most reduces the error. Deterministic,
    and it never leaves the ladder.
    """
    if not wanted:
        return []
    lengths = [snap_to_rung(float(s), rungs_seconds) for s in wanted]
    order = sorted(rungs_seconds)

    def error(values):
        return abs(sum(values) - total)

    tolerance = (order[1] - order[0]) / 2.0 if len(order) > 1 else 0.05
    for _ in range(len(lengths) * 8):
        if error(lengths) <= tolerance:
            break
        best, best_error = None, error(lengths)
        for i, value in enumerate(lengths):
            at = order.index(snap_to_rung(value, order))
            for step in (-1, 1):
                j = at + step
                if not 0 <= j < len(order):
                    continue
                trial = list(lengths)
                trial[i] = order[j]
                if error(trial) < best_error - 1e-9:
                    best, best_error = (i, order[j]), error(trial)
        if best is None:
            break
        lengths[best[0]] = best[1]
    return lengths


def feasible_counts(total, rungs_seconds):
    """How many clips a runtime can actually be built from.

    ``reconcile`` can only change the LENGTH of the segments it is given, so a
    model that returns twice as many beats as the runtime holds produces a
    video twice as long — and when the allowed window contains a single rung it
    cannot even nudge those. A 15-second brief came back as six 5.167s clips,
    31 seconds of advert.
    """
    if not rungs_seconds:
        return 1, 1
    shortest, longest = min(rungs_seconds), max(rungs_seconds)
    order = sorted(rungs_seconds)
    tolerance = ((order[1] - order[0]) / 2.0 if len(order) > 1
                 else shortest / 2.0)
    fewest = max(1, int(math.ceil((float(total) - tolerance) / longest)))
    most = max(fewest, int(math.floor((float(total) + tolerance) / shortest)))
    return fewest, most


def fit_count(rows, total, rungs_seconds):
    """Trim a beat list down to what the runtime can hold.

    The opening and closing beats are kept — they are the hook and the hero
    shot — and the surplus comes out of the middle, which is where a plan
    repeats itself anyway. Returns (rows, dropped_count).
    """
    _, most = feasible_counts(total, rungs_seconds)
    if len(rows) <= most:
        return rows, 0
    if most == 1:
        return rows[:1], len(rows) - 1
    keep = [0]
    middle = most - 2
    if middle > 0:
        # Spread the survivors evenly through the middle rather than taking
        # the first few: keeping beats 1 and 2 of six throws away the whole
        # second half of the story.
        span = len(rows) - 2
        keep += [1 + int(round((i + 0.5) * span / float(middle) - 0.5))
                 for i in range(middle)]
    keep.append(len(rows) - 1)
    ordered = sorted(dict.fromkeys(keep))[:most]
    return [rows[i] for i in ordered], len(rows) - len(ordered)


def segment_count(total, rungs_seconds):
    """How many clips a runtime wants, at the top of the allowed window."""
    longest = max(rungs_seconds)
    return max(1, int(math.ceil(total / longest - 1e-9)))


def cast_reference_images(cast, limit=4):
    """The cast's uploaded images, encoded for a vision call.

    Returns [] when nothing is readable or the prompt engine is absent, which
    simply means the planner falls back to describing nothing it cannot see.
    """
    from . import paths
    out = []
    for m in (cast or {}).get("members", []):
        if m.get("kind") not in ("Subject", "Picture") or not m.get("file"):
            continue
        blob = engine.image_b64(os.path.join(paths.cast_dir(), m["file"]))
        if blob:
            out.append(blob)
        if len(out) >= limit:
            break
    return out

# --------------------------------------------------------------------------

class H3PlannerStoryPlanner:
    """A brief in, a fully written timeline out, in one model call.

    Use this for anything that has to read as one continuous piece: an ad, a
    short film, a product story. Use the Segment Prompter instead when the
    segments are deliberately unrelated, as in most music videos.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "idea": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "the whole brief: what happens, who is in it, what it is for"}),
                "total_seconds": ("FLOAT", {
                    "default": 30.0, "min": 2.0, "max": 600.0, "step": 0.5}),
                "min_segment_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 15.0, "step": 0.1,
                    "tooltip": "shortest clip the planner may ask for. Longer clips cost less per second of video, because the fixed per-run overhead is amortised."}),
                "max_segment_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 15.0, "step": 0.1,
                    "tooltip": "longest clip your card can render. A ceiling, not a target — the planner varies length to suit each beat."}),
                "format": (FORMATS, {"default": "cinematic ad"}),
                "dialogue": (list(DIALOGUE_MODES), {
                    "default": "auto",
                    "tooltip": "whether anyone speaks on camera. 'spoken lines' makes the planner write the actual script and render it as H3 <d>[English] ...</d> dialogue with speaker IDs; 'auto' decides from the brief; 'none' keeps it picture only."}),
                "audio_role": (list(STORY_AUDIO_ROLES), {
                    "default": "performed on camera",
                    "tooltip": "what the connected audio IS. The first three treat it as the soundtrack, reused exactly: 'performed on camera' when a visible subject raps or sings it. 'voice sample' is the opposite and is the one for an ad: the audio is only a timbre, the dialogue is written here and generated in that voice. Ignored with no audio in the cast."}),
                "window": ("INT", {
                    "default": 4, "min": 1, "max": 12,
                    "tooltip": "segments written per prose call once the piece is too long for one reply. Each window sees the previous window's closing state."}),
                "single_call_max": ("INT", {
                    "default": 8, "min": 1, "max": 40,
                    "tooltip": "at or below this many segments the whole video is written in ONE prose call, which is what makes it consistent. Above it, windows are used."}),
                "chain_first_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "start each segment from the previous one's last frame, within a scene. Hard visual continuity at the joins, but artifacts compound across a long chain — leave off unless you need it."}),
                "provider": (engine.providers(), {"default": "Ollama (Local)"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": ("STRING", {"default": engine.default_ollama_model()}),
                "temperature": ("FLOAT", {"default": 0.35, "min": 0.0,
                                          "max": 1.2, "step": 0.05}),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "keep segments already written from the same brief"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "change to replan the whole video"}),
            },
            "optional": {
                "cast": ("H3_CAST",),
                "style_prefix_override": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "api_model": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 256,
                                              "max": 32768}),
                "num_ctx": ("INT", {"default": 16384, "min": 2048,
                                    "max": 131072}),
                "keep_alive": ("STRING", {"default": "10m"}),
                "request_timeout": ("INT", {"default": 900, "min": 30,
                                            "max": 3600}),
            },
        }

    RETURN_TYPES = ("H3_TIMELINE", "STRING", "STRING")
    RETURN_NAMES = ("timeline", "beat_sheet", "report")
    FUNCTION = "plan"
    CATEGORY = CATEGORY

    def plan(self, project, idea, total_seconds, min_segment_seconds,
             max_segment_seconds, format, dialogue, audio_role, window,
             single_call_max,
             chain_first_frames, provider, ollama_url, ollama_model,
             temperature, reuse_existing, seed, cast=None,
             style_prefix_override="", api_key="", api_model="",
             max_output_tokens=8192, num_ctx=16384, keep_alive="10m",
             request_timeout=900):

        if not idea.strip():
            raise ValueError("nothing to plan from — describe the video in `idea`.")
        if not engine.available():
            raise RuntimeError(engine.MISSING)

        started = time.time()
        low = min(float(min_segment_seconds), float(max_segment_seconds))
        high = min(float(max_segment_seconds), float(project["max_seconds"]))
        rungs = allowed_rungs(project, low, high)
        rung_seconds = [s for _, s in rungs]
        # 5.0s is not on the frame ladder at all — the neighbours are
        # 4.458s and 5.167s — so a window of exactly [5.0, 5.0] contains
        # nothing and allowed_rungs falls back to the closest rung. That
        # is the right thing to do and it must not be silent.
        window_note = ""
        if rung_seconds and rung_seconds[0] < low - 1e-6:
            window_note = (
                "no clip length exists between %.3fs and %.3fs, so %.3fs "
                "was used instead. H3 only renders certain lengths; the "
                "ones near your window are %s. Widen min/max_segment_"
                "seconds to pick deliberately."
                % (low, high, rung_seconds[0],
                   ", ".join("%.3fs" % x for x in ladder_neighbours(project, low, high))))

        backend_kwargs = dict(
            provider=provider, ollama_url=ollama_url,
            ollama_model=ollama_model, api_model=api_model,
            temperature=temperature, keep_alive=keep_alive,
            timeout=request_timeout,
            max_output_tokens=max_output_tokens, num_ctx=num_ctx)
        cfg = engine.backend_config(api_key=api_key,
                                    **backend_kwargs)


        allowed = _allowed_tags(cast)
        cast_block = _cast_block(cast)
        names = asset_names(cast)
        audio_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                          if m.get("kind") == "Audio"), "")
        subject_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                            if m.get("kind") == "Subject"), "")
        # Order-paired for now; the beat sheet recasts it from the brief.
        voices = cast_voices(cast)

        fingerprint = _hash(idea, total_seconds, low, high, format, seed,
                            dialogue, audio_role,
                            cast_block, style_prefix_override,
                            project["fps"], project["max_seconds"])

        # ---- 0. has anything actually changed? ---------------------------
        # Two model calls fired on every queue, even when the inputs were
        # identical and every prompt was about to be reused verbatim. On a
        # graph queued once per segment that is a call per render for no
        # output. If the timeline on disk was written from these exact
        # inputs, it IS the answer.
        done = self._already_planned(project, fingerprint, reuse_existing)
        if done is not None:
            ctx = done.get("context") or {}
            sheet = self._beat_sheet(done, ctx.get("style_prefix", ""),
                                     ctx.get("world", ""),
                                     ctx.get("subject_definitions", ""))
            note = ("unchanged since the last plan — %d segment(s) reused, "
                    "no model calls. Change the idea, the seed or a setting "
                    "to replan." % len(done["segments"]))
            print("[H3Planner] story planner: %s" % note)
            return (done, sheet, note)

        # ---- 1. the beat sheet, in one call ------------------------------
        wanted = segment_count(float(total_seconds), rung_seconds)
        # Show the planner the actual references. Described from the brief
        # alone it invents a person, and an invented description beside a real
        # photograph is the contradiction H3 blends.
        seen = cast_reference_images(cast)
        beats = self._call_beats(cfg, idea, float(total_seconds), wanted,
                                 rungs, format, cast_block, audio_tag,
                                 dialogue, seen, audio_role, voices)
        # "use <audio 1> for the male character" only exists in the brief, and
        # the director is the only pass that reads it. Untrusted like every
        # other field: cast_voices checks each pair against the real cast.
        voices = cast_voices(cast, beats.get("voices"))

        prefix = (style_prefix_override.strip()
                  or engine.clean(beats.get("style_prefix", "")))
        world = engine.clean(beats.get("world", ""))
        canon = engine.dedupe_subjects(
            engine.clean(beats.get("subject_definitions", "")))

        # Everything a single segment has to be told about the whole piece.
        # Assembled once so all of them get the SAME words for the same room;
        # identical wording is what makes two clips render one location.
        story = {"fields": world_fields(world), "cast": cast,
                 "format": format, "total": 0}

        # Who may speak. A `speaker` naming a subject the cast does not
        # have is a tag that binds no face, and it travelled all the way to
        # the card before anything looked at it.
        subjects = [m["tag"] for m in (cast or {}).get("members", [])
                    if m.get("kind") == "Subject"]
        miscast = []

        rows = list(beats.get("segments") or [])
        if not rows:
            raise RuntimeError(
                "the planner returned no segments. Try a longer `idea`, a "
                "larger num_ctx, or a stronger model.")

        # Enforce the runtime the user asked for. Without this the plan is
        # however long the model felt like making it.
        rows, dropped = fit_count(rows, float(total_seconds), rung_seconds)
        lengths = reconcile([r.get("seconds") for r in rows],
                            float(total_seconds), rung_seconds)

        # The runtime can be unreachable with the beats the model chose: nine
        # clips capped at 9.417s cannot fill a 90-second slot however they are
        # nudged. Say so rather than quietly delivering a short film — for an
        # ad with a hard slot length that is the difference between usable and
        # not.
        count_note = ""
        if dropped:
            count_note = (
                "the model planned %d beats but %.3fs only holds %d at "
                "these clip lengths, so %d were dropped from the middle. "
                "The opening and closing beats were kept."
                % (len(rows) + dropped, float(total_seconds), len(rows),
                   dropped))
        drift = sum(lengths) - float(total_seconds)
        tolerance = (rung_seconds[1] - rung_seconds[0]) / 2.0 \
            if len(rung_seconds) > 1 else 0.05
        drift_note = ""
        if abs(drift) > tolerance:
            reach = "%.3fs..%.3fs" % (len(rows) * min(rung_seconds),
                                      len(rows) * max(rung_seconds))
            drift_note = (
                "the plan is %.3fs %s than the %.3fs asked for. The model "
                "returned %d segment(s), which can only cover %s. %s"
                % (abs(drift), "longer" if drift > 0 else "shorter",
                   float(total_seconds), len(rows), reach,
                   "Raise max_segment_seconds, or re-run with a new seed."
                   if drift < 0 else "Lower min_segment_seconds, or re-run "
                   "with a new seed."))

        # ---- 2. lay out the timeline (arithmetic, no model) --------------
        segments, clock = [], 0.0
        for i, (row, seconds) in enumerate(zip(rows, lengths)):
            scene = int(row.get("scene") or 1)
            prev = segments[-1] if segments else None
            same_scene = prev is not None and prev["scene"] == scene
            segments.append({
                "id": "seg_%02d" % (i + 1),
                "beat": engine.clean(row.get("action", ""))[:120],
                "target_duration": round(seconds, 3),
                "scene": scene,
                "scene_name": engine.clean(row.get("scene_name", "")),
                "action": engine.clean(row.get("action", "")),
                "opens_from": engine.clean(row.get("opens_from", "")),
                "ends_with": engine.clean(row.get("ends_with", "")),
                "dialogue": ("" if dialogue == "none"
                             else engine.clean(row.get("dialogue", ""))),
                "speaker": self._speaker_of(row, subjects, miscast),
                # Chaining is opt-in and never crosses a scene: a cut to a new
                # location must not start from the last frame of the old one.
                "link": ("continue" if (chain_first_frames and same_scene)
                         else "cut"),
                "prompt": "",
                # the whole video and the reference track share one clock, and
                # it is the TRIMMED clock — the stitcher cuts every clip back
                # to target_duration, so this is where the segment really lands
                "audio_start": round(clock, 3),
            })
            clock += seconds

        # A shot must open on what the one before it actually ended on, whatever
        # the model wrote in its own opens_from.
        for prev, seg in zip(segments, segments[1:]):
            if seg["scene"] == prev["scene"] and prev["ends_with"]:
                seg["opens_from"] = prev["ends_with"]

        context = {
            "style_prefix": prefix,
            "world": world,
            "subject_definitions": canon,
            "total_duration": float(total_seconds),
            "shot_count": len(segments),
        }
        # So a single card can be refined later without re-queueing the
        # graph. The API key is deliberately never stored.
        store.remember_backend(context, backend_kwargs)
        timeline = store.normalize_timeline(
            {"name": project["name"], "context": context,
             "segments": segments}, project)
        story["total"] = len(timeline["segments"])
        for seg in timeline["segments"]:
            frames = ladder.snap_frames(
                seg["target_duration"], project["fps"],
                modulus=project["frame_modulus"],
                remainder=project["frame_remainder"],
                minimum=project["frame_minimum"], direction=project["snap"])
            seg["render_frames"] = frames
            seg["render_duration"] = frames / float(project["fps"])

        reused = self._reuse(project, timeline, fingerprint, reuse_existing)

                # ---- 3. the prose, in one call or a few windows ------------------
        todo = [s for s in timeline["segments"]
                if not s.get("prompt") and s.get("state") != store.LOCKED]
        # max(1, ...) because an empty `todo` made this zero, and range() with
        # a zero step raises. Every segment already being written is a normal
        # state — every card locked, or every prompt reused — not a crash.
        size = max(1, len(todo) if len(todo) <= int(single_call_max)
                   else int(window))
        # The beat sheet was a call too — reporting only the prose calls
        # understates what this cost, which is the number being compared
        # against the Segment Prompter's one-per-segment.
        calls, failed, notes = 1, [], []
        if not todo:
            notes.append(
                "nothing needed writing — every segment is locked or was "
                "reused. Unlock a card, or change the seed, to replan one.")

        for start in range(0, len(todo), size):
            batch = todo[start:start + size]
            before = self._previous(timeline, batch[0])
            try:
                written = self._call_prose(
                    cfg, timeline, batch, before, prefix, world, canon,
                    cast_block, audio_tag, audio_role, format, idea,
                    subject_tag, voices)
                calls += 1
            except Exception as ex:
                failed.append("%s..%s: %s" % (batch[0]["id"], batch[-1]["id"], ex))
                continue

            for seg, obj in zip(batch, written):
                prompt, invented = self._assemble(
                    obj, seg, prefix, allowed, names, audio_tag, canon,
                    audio_role, subject_tag, voices, story)
                seg["prompt"] = prompt
                seg["prompt_fingerprint"] = fingerprint
                seg["spec_hash"] = store.spec_hash(seg)
                if invented:
                    notes.append("%s: stripped %s"
                                 % (seg["id"], ", ".join(invented)))
                if not self._dialogue_landed(prompt, seg.get("dialogue")):
                    failed.append(
                        "%s was given a line to speak and the prompt came "
                        "back without it in <d>...</d>, so nothing will be "
                        "said on screen: %r" % (seg["id"], seg["dialogue"][:60]))
            if len(written) < len(batch):
                notes.append(
                    "%s..%s: the model returned %d of %d segments; the rest "
                    "are written on their own below"
                    % (batch[0]["id"], batch[-1]["id"], len(written), len(batch)))

        # Fill whatever the windowed call left out. A reply short of the batch
        # used to leave those segments with no prompt at all — the plan looked
        # complete on the beat sheet while only the first segment could
        # actually render. Asking for one segment at a time always produced a
        # reply, so the gap is recoverable rather than fatal.
        for seg in todo:
            if seg.get("prompt") or seg.get("state") == store.LOCKED:
                continue
            try:
                solo = self._call_prose(
                    cfg, timeline, [seg], self._previous(timeline, seg),
                    prefix, world, canon, cast_block, audio_tag, audio_role,
                    format, idea, subject_tag, voices)
                calls += 1
            except Exception as ex:
                failed.append("%s: not written (%s)" % (seg["id"], ex))
                continue
            if not solo:
                failed.append("%s: the provider returned nothing for it"
                              % seg["id"])
                continue
            prompt, invented = self._assemble(
                solo[0], seg, prefix, allowed, names, audio_tag, canon,
                audio_role, subject_tag, voices, story)
            seg["prompt"] = prompt
            seg["prompt_fingerprint"] = fingerprint
            seg["spec_hash"] = store.spec_hash(seg)
            if invented:
                notes.append("%s: stripped %s"
                             % (seg["id"], ", ".join(invented)))

        # A windowed call can return one film five times over. Rewriting the
        # offenders ALONE removes the temptation: with a single segment in the
        # request there is no whole-piece structure to copy.
        bleed = self._bleed(timeline["segments"], prefix)
        if bleed:
            print("[H3Planner] segment bleed detected in %s — rewriting one "
                  "at a time" % ", ".join(sorted(bleed)))
            for seg in timeline["segments"]:
                if seg["id"] not in bleed or seg.get("state") == store.LOCKED:
                    continue
                try:
                    solo = self._call_prose(
                        cfg, timeline, [seg], self._previous(timeline, seg),
                        prefix, world, canon, cast_block, audio_tag,
                        audio_role, format, idea, subject_tag, voices)
                    calls += 1
                except Exception as ex:
                    failed.append("%s: rewrite failed (%s)" % (seg["id"], ex))
                    continue
                if not solo:
                    continue
                prompt, invented = self._assemble(
                    solo[0], seg, prefix, allowed, names, audio_tag, canon,
                    audio_role, subject_tag, voices, story)
                seg["prompt"] = prompt
                seg["spec_hash"] = store.spec_hash(seg)
                if invented:
                    notes.append("%s: stripped %s"
                                 % (seg["id"], ", ".join(invented)))

            still = self._bleed(timeline["segments"], prefix)
            for seg_id in sorted(still):
                failed.append(
                    "%s still %s after being rewritten on its own. Every "
                    "affected clip would render the same footage — replan "
                    "with a different seed, or use a stronger model."
                    % (seg_id, " and ".join(still[seg_id])))
            fixed = sorted(set(bleed) - set(still))
            if fixed:
                notes.append("rewrote %s on their own after the windowed call "
                             "returned the whole piece in each"
                             % ", ".join(fixed))

        store.save(project["timeline_path"], timeline)

        sheet = self._beat_sheet(timeline, prefix, world, canon)
        if count_note:
            notes.append(count_note)
        if window_note:
            notes.append(window_note)
        unwritten = [x["id"] for x in timeline["segments"]
                     if not x.get("prompt")]
        if unwritten:
            failed.append(
                "%s have NO prompt and cannot render. Re-run with a "
                "different seed, or a stronger model."
                % ", ".join(unwritten))
        if drift_note:
            notes.append(drift_note)
        notes.extend(self._overlong(timeline["segments"]))
        notes.extend(self._script_shape(timeline["segments"],
                                        float(total_seconds)))
        notes.extend(story.get("trimmed") or [])
        mismatch = self._audio_role_mismatch(cast, audio_role, idea)
        if mismatch:
            notes.insert(0, mismatch)
        if audio_role == VOICE_SAMPLE:
            notes.extend(self._voices_used(timeline["segments"], voices))
        if miscast:
            notes.append(
                "the plan gave %d line(s) to a subject the cast does not "
                "have (%s); those lines fall to the first cast voice. Check "
                "who is meant to say them."
                % (len(miscast), ", ".join(sorted(set(miscast)))))
        stale = self._untagged(timeline["segments"], allowed)
        if stale:
            notes.append(
                "%s name the cast in prose rather than by tag, so the "
                "renderer cannot tell which supplied image they mean. "
                "Re-run with a new seed if the references are ignored."
                % ", ".join(stale))
        report = self._report(timeline, project, prefix, world, cast_block,
                              format, rungs, calls, reused, failed, notes,
                              float(total_seconds), time.time() - started,
                              audio_role, voices)
        return (timeline, sheet, report)

    @staticmethod
    def _already_planned(project, fingerprint, reuse_existing):
        """The saved timeline, when it was written from these exact inputs.

        Returns None whenever anything is missing or stale, so the only way to
        skip the model is for every segment to already carry a prompt stamped
        with this fingerprint.
        """
        if not reuse_existing:
            return None
        saved = store.load(project["timeline_path"])
        if not saved or not saved.get("segments"):
            return None
        for seg in saved["segments"]:
            if not seg.get("prompt"):
                return None
            if seg.get("state") == store.LOCKED:
                # Locked segments keep whatever they hold, so a hand-written
                # one carrying an older fingerprint is not a reason to replan
                # everything around it.
                continue
            if seg.get("prompt_fingerprint") != fingerprint:
                return None
        return saved

    # -- the calls --------------------------------------------------------

    def _call_beats(self, cfg, idea, total, wanted, rungs, format,
                    cast_block, audio_tag, dialogue="auto", images=None,
                    audio_role="background only", voices=()):
        menu = ", ".join("%.3fs" % s for _, s in rungs)
        system = BEATS_SYSTEM
        guidance = FORMAT_GUIDANCE.get(format, "").strip()
        if guidance:
            system += "\n\n" + guidance
        if audio_tag and audio_role == VOICE_SAMPLE:
            # The opposite instruction. Told the audio "will be used exactly
            # as given", the director plans a piece with no dialogue in it,
            # because the soundtrack is already decided.
            system += (
                "\n\nThe supplied audio assets are VOICE SAMPLES, one per "
                "character. Their contents are not used: you write the script "
                "and it is spoken in those voices. Cast them in `voices`, and "
                "name the speaking subject in each segment's `speaker`.")
        elif audio_tag:
            system += ("\n\nAn audio asset (%s) is supplied and will be used "
                       "exactly as given. Do not plan any sound of your own."
                       % audio_tag)

        user = "\n".join([
            "THE BRIEF:",
            idea.strip(),
            "",
            "TOTAL DURATION: %.3f seconds." % total,
            "",
            "ALLOWED CLIP LENGTHS — every `seconds` value must be one of these "
            "exactly. They are the only lengths the renderer accepts:",
            "  " + menu,
            "",
            self._dialogue_brief(dialogue, rungs, total,
                                 voices if audio_role == VOICE_SAMPLE else ()),
            "",
            "Aim for about %d segments. Fewer, longer clips are preferred: a "
            "clip may cut internally, and doing the fast cutting inside one "
            "clip looks more continuous than splitting it into more clips."
            % wanted,
            "",
            "CAST — the only tags that exist:",
            cast_block,
        ])
        if audio_role == VOICE_SAMPLE and voices:
            user += "\n\n" + "\n".join([
                "VOICE CASTING. Each audio asset is one character's voice. "
                "Return `voices` as one object per audio tag, {audio, "
                "subject}. Read the brief: it may say which voice belongs to "
                "whom. If it does not, choose and be consistent.",
                "  audio tags: " + ", ".join(v["audio"] for v in voices),
                "Then set every segment's `speaker` to the subject tag "
                "saying that segment's line, exactly as written in the cast.",
            ])
        obj, _ = engine.generate(
            cfg, system, user, beats_schema(),
            required_keys=("style_prefix", "world", "segments"),
            images=images or [])
        if not obj:
            raise RuntimeError(
                "the planner's first call returned nothing usable. Check the "
                "provider is reachable and the model supports structured JSON.")
        return obj

    def _call_prose(self, cfg, timeline, batch, before, prefix, world, canon,
                    cast_block, audio_tag, audio_role, format, idea,
                    subject_tag="", voices=()):
        system = engine.full_ref_system() + "\n\n" + PROSE_SYSTEM
        guidance = FORMAT_GUIDANCE.get(format, "").strip()
        if guidance:
            system += "\n\n" + guidance
        audio_block = (voice_system_block(voices)
                       if audio_role == VOICE_SAMPLE
                       else audio_system_block(audio_tag, audio_role))
        if audio_block:
            system += "\n\n" + audio_block

        total = sum(s["target_duration"] for s in timeline["segments"])
        lines = [
            "ONE VIDEO, %.2f seconds, %d segments in total. You are writing "
            "segments %d to %d of them, in order."
            % (total, len(timeline["segments"]),
               batch[0]["index"] + 1, batch[-1]["index"] + 1),
            "",
            "STYLE PREFIX (verbatim at the start of [Shot 1] of every segment):",
            "  " + (prefix or "(none — choose one and keep it identical)"),
            "",
            "THE WORLD — binding on every segment, must not drift:",
            world or "(not specified)",
            "",
            "SUBJECT DEFINITIONS — reuse this wording exactly:",
            canon or "(none)",
            "",
            "CAST — the only tags you may cite:",
            cast_block,
        ]
        if idea.strip():
            lines += ["", "THE BRIEF, for intent only — do not narrate it:",
                      idea.strip()[:1200]]
        if before is not None:
            lines += ["", "THE SEGMENT IMMEDIATELY BEFORE THIS RUN ended like "
                      "this. The first segment below opens on it:",
                      "  " + (before.get("ends_with")
                              or before.get("action") or "(not recorded)")]

        lines += ["", "THE SEGMENTS TO WRITE:"]
        for seg in batch:
            render = seg.get("render_duration") or seg["target_duration"]
            lines += [
                "",
                "--- SEGMENT %d (%s), scene %s%s ---"
                % (seg["index"] + 1, seg["id"], seg.get("scene"),
                   (" — " + seg["scene_name"]) if seg.get("scene_name") else ""),
                "  THIS OBJECT DESCRIBES ONLY THIS SEGMENT. Everything below "
                "belongs to it and to no other.",
                "  shot budget:       %d shot(s) — do not exceed it"
                % self._shot_budget(seg["target_duration"]),
                "  planned duration:  %.3fs — all essential action ends by here"
                % seg["target_duration"],
                "  rendered duration: %.3fs — no timestamp may reach this"
                % render,
                "  the last %.3fs is a hold on the final composition"
                % (render - seg["target_duration"]),
                ("  CONTINUES from the clip before: [Shot 1] holds the "
                 "state below and carries no dialogue; [Shot 2] begins "
                 "At %s and moves on."
                 % self._clock(self._hold_seconds(seg["target_duration"]))
                 if seg.get("link") == "continue" else "  starts on a cut."),
                "  opens on:  %s" % (seg.get("opens_from") or "(continue from above)"),
                "  action:    %s" % (seg.get("action") or seg.get("beat") or ""),
                ("  SAYS, verbatim, spoken by %s inside "
                 "<d>[English] ...</d>:  %s"
                 % (self._speaker_label(seg, voices, subject_tag),
                    seg["dialogue"])
                 if seg.get("dialogue") else "  says:      nothing, picture only"),
                ("  ENDS ON, and the clip stops there: %s. The segment "
                 "after this one opens on exactly that, so anything you "
                 "write past it is rendered twice."
                 % seg["ends_with"]) if seg.get("ends_with")
                else "  ends on:   (unspecified)",
            ]
        lines += [
            "",
            "Return exactly %d segment object(s), in this order. Object N "
            "describes SEGMENT N above and nothing else: its action, its "
            "shot budget, its line. Do not put the whole piece in any one "
            "of them." % len(batch),
        ]

        obj, _ = engine.generate(cfg, system, "\n".join(lines), prose_schema(),
                                 required_keys=("segments",))
        if not obj:
            raise RuntimeError("the provider returned nothing usable")
        written = obj.get("segments")
        if not isinstance(written, list) or not written:
            raise RuntimeError("no segments in the reply")
        return written[:len(batch)]

    @staticmethod
    def _speaker_of(row, subjects, miscast):
        """The `speaker` this beat row claims, if the cast really has it."""
        wanted = engine.clean(row.get("speaker", ""))
        if not wanted or not subjects:
            return wanted if not subjects else ""
        if wanted in subjects:
            return wanted
        miscast.append(wanted)
        return ""

    @staticmethod
    def _speaker_label(seg, voices, subject_tag=""):
        """The "<Subject 2> (S1)" label for this segment's line.

        It used to be the first subject and (S1), always, so in a two-hander
        the second character could not speak: both halves of the conversation
        came out of one mouth.
        """
        wanted = (seg.get("speaker") or "").strip()
        for v in (voices or []):
            if v.get("subject") and v["subject"] == wanted:
                return "%s %s" % (v["subject"], v["speaker"])
        if wanted:
            # Named a subject with no voice cast to it: still keep the two
            # labels together, since that is what binds the line to a face.
            return "%s %s" % (wanted, speaker_id(0))
        for v in (voices or []):
            if v.get("subject"):
                return "%s %s" % (v["subject"], v["speaker"])
        return "%s (S1)" % (subject_tag or "the speaker")

    # -- assembly ---------------------------------------------------------

    def _assemble(self, obj, seg, prefix, allowed, names, audio_tag, canon,
                  audio_role="background only", subject_tag="", voices=(),
                  story=None):
        """Everything the model is not trusted to get right, done in code.

        Deliberately the same sequence the Segment Prompter uses, so a prompt
        from either node is enforced identically — the only difference is how
        many calls produced it.
        """
        render = seg.get("render_duration") or seg["target_duration"]
        prompt = {k: engine.clean(obj.get(k)) or "N/A" for k in SECTIONS}
        prompt = {k: strip_asset_names(v, names) or "N/A"
                  for k, v in prompt.items()}
        # The summary is written here, not asked for. The prose call never
        # requested it, so it came back empty and every segment of a five-part
        # advert carried "N/A" in the one section whose job is to say what the
        # video is and that it is one continuous piece.
        #
        # Before the tag pass, not after: the handover it quotes is the
        # director's own prose and cites <Subject N>, so it has to be bound and
        # checked like everything else.
        if story:
            prompt["summary"] = build_summary(
                seg, seg.get("index", 0), story.get("total", 1),
                story.get("fields") or {},
                task_prefix(story.get("cast"), audio_role,
                            seg.get("link") == "continue"),
                story.get("format", ""))

        invented = []
        if allowed:
            prompt, _bound = bind_tags(prompt, allowed)
            prompt, invented = strip_unknown_tags(prompt, allowed)
        if audio_tag and audio_role == VOICE_SAMPLE:
            # A segment with no line must not be told its characters speak.
            # Both sources: the planned line, and whatever the writer actually
            # put inside <d>...</d>, since either alone can be empty.
            speaks = bool((seg.get("dialogue") or "").strip()
                          or re.search(r"<\s*d\s*>",
                                       prompt.get("detailed_description", ""),
                                       re.IGNORECASE))
            prompt, _ = enforce_voice_reference(prompt, voices, speaks)
        elif audio_tag:
            prompt, _ = enforce_audio_exact(prompt, audio_tag, audio_role,
                                            subject_tag)

        # The canonical wording wins outright. Asking every segment to describe
        # the cast identically does not work; copying one block does.
        shared = canonical_subjects(canon, cited_tags(prompt)) if canon else ""
        prompt["subject_definitions"] = engine.dedupe_subjects(
            shared or prompt["subject_definitions"])

        body = prompt["detailed_description"]
        if prefix:
            body = self._force_prefix(body, prefix)
        if story:
            # All three reached the writer and were free to be ignored.
            # Segment two of the cafe advert named neither the cafe nor the
            # state the previous clip ended on; segment four was briefed to
            # end on her standing up and went on to show her leaving, which
            # is segment five's job, so the exit rendered twice.
            body = anchor_world(body, world_clause(story.get("fields") or {}),
                                prefix)
            body = enforce_opening(body, seg.get("opens_from"), prefix)
            body, dropped = trim_after_ending(body, seg.get("ends_with"))
            if dropped:
                story.setdefault("trimmed", []).append(
                    "%s ran past its own ending and into the next segment: "
                    "%d shot(s) after \"%s\" were dropped, because the "
                    "segment after it opens there and would show the same "
                    "action again."
                    % (seg["id"], dropped,
                       (seg.get("ends_with") or "")[:60]))
        body = renumber_shots(body)
        body = engine.fix_shot_times(body, render)
        # The airlock hold is written here, after the creator's timing
        # pass, because that pass respaces shots evenly and would put the
        # first cut at the midpoint of the clip whatever we asked for.
        if seg.get("link") == "continue":
            body = self.enforce_hold(
                body, self._hold_seconds(seg["target_duration"]), render)
        prompt["detailed_description"] = body

        rendered = engine.normalize_labels(
            engine.render_full_ref(prompt, render), render)
        parsed = splitter.parse_sections(rendered)
        for key in SECTIONS:
            parsed.setdefault(key, "N/A")
        if invented:
            seg["tag_warnings"] = ["invented %s" % t for t in invented]
        return {k: parsed[k] for k in SECTIONS}, invented

    @staticmethod
    def _force_prefix(body, prefix):
        body = (body or "").strip()
        if not prefix:
            return body
        flat = lambda v: re.sub(r"[^a-z0-9]+", "", (v or "").lower())
        head_flat = flat(prefix)

        # Models like to restate the prefix as a bare line above [Shot 1], and
        # this then added it a second time inside the shot:
        #     Cinematic 35mm anamorphic, warm daylight grade
        #      [Shot 1] Cinematic 35mm anamorphic, warm daylight grade, she...
        marker = splitter.SHOT_RE.search(body)
        if marker and marker.start() > 0:
            lead = body[:marker.start()]
            if flat(lead) and head_flat.startswith(flat(lead)[:len(head_flat)]):
                body = body[marker.start():]
                marker = splitter.SHOT_RE.search(body)

        if not marker:
            return "[Shot 1] %s, %s" % (prefix.rstrip(" ,"), body.lstrip())
        head, rest = body[:marker.end()], body[marker.end():].lstrip()
        rest = re.sub(r"^[,\s]+", "", rest)
        if flat(rest).startswith(head_flat):
            return "%s %s" % (head.rstrip(), rest)
        return "%s %s, %s" % (head.rstrip(), prefix.rstrip(" ,"), rest)

    # -- bookkeeping ------------------------------------------------------

    # Natural speech is about 2.5 words a second. An advert that speaks for its
    # whole runtime is a radio ad with pictures attached: written commercials
    # carry roughly 1.5 words per second of runtime, which spends a little over
    # half the time speaking and leaves the rest to picture, sound design and
    # the end card. Wall-to-wall narration is the single most common failure in
    # a machine-written script.
    WORDS_PER_SECOND_SPOKEN = 2.5
    WORDS_PER_SECOND_TOTAL = 1.5

    @classmethod
    def _dialogue_brief(cls, mode, rungs, total_seconds=30.0, voices=()):
        """How much the beat sheet should be told about speech.

        Two budgets, and the total is the one that matters. Told only a
        per-segment limit, a model fills every segment to it and the result
        talks without pausing from the first frame to the last.
        """
        if mode == "none":
            return ("DIALOGUE: nobody speaks. Leave every segment's dialogue "
                    'field as "".')
        total_words = max(6, int(total_seconds * cls.WORDS_PER_SECOND_TOTAL))
        per_rung = ", ".join(
            "%.0fs fits about %d" % (s, int(s * cls.WORDS_PER_SECOND_SPOKEN))
            for _, s in rungs[::2])
        head = ("DIALOGUE: write the actual spoken lines."
                if mode == "spoken lines" else
                "DIALOGUE: if the brief calls for anyone to speak, write the "
                "actual spoken lines; otherwise leave them empty.")

        speakers = [v for v in (voices or []) if v.get("subject")]
        if len(speakers) > 1 and mode != "none":
            # A conversation is not a commercial voice-over, and the rules
            # below are the wrong ones for it: told that a segment in the
            # middle must be silent and the last one too, the director wrote
            # one line across five segments and gave it all to one character.
            return cls._conversation_brief(head, speakers, total_words,
                                           per_rung)
        return "\n".join([
            head,
            "  BUDGET: about %d words across the WHOLE piece — not per segment."
            % total_words,
            "    That is the budget a written %.0f-second commercial gets. It "
            "spends" % total_seconds,
            "    a little over half the runtime speaking and leaves the rest to",
            "    picture and sound. Coming in under it is fine; going over "
            "makes it",
            "    a radio ad with pictures attached.",
            "  SILENCE IS STRUCTURAL, not leftover space:",
            "    - At least one segment in the middle says nothing at all. That "
            "is the",
            "      beauty shot, where the product or the moment carries itself.",
            "    - The final segment is silent, or carries a tagline of six "
            "words or",
            "      fewer. A hero shot is weakened by narration over it.",
            "    - No segment is filled to its limit. A line lands better with "
            "a beat",
            "      of quiet before and after it than butted against the cut.",
            "  SHAPE: open on a hook, let the tension or need land, reveal the "
            "product,",
            "    give one concrete reason to care, then pay it off. One idea "
            "per line;",
            "    a line that carries three features carries none of them.",
            "  The lines still run consecutively as one continuous script.",
            "  Per segment, if you do write a line: %s." % per_rung,
        ])

    @staticmethod
    def _conversation_brief(head, speakers, total_words, per_rung):
        """The dialogue rules when two or more characters actually talk."""
        return "\n".join([
            head,
            "  THIS IS A CONVERSATION between %d characters, not narration "
            "over pictures." % len(speakers),
            "  WHO SPEAKS: set each segment's `speaker` to the subject tag "
            "saying its line.",
        ] + ["    %s speaks as %s" % (v["subject"], v["speaker"])
             for v in speakers] + [
            "  - They take turns. A segment where one of them talks is "
            "usually followed",
            "    by one where another answers. Do not give every line to the "
            "same person;",
            "    a character who never speaks is a character the audience "
            "does not meet.",
            "  - Every one of them speaks at least once, and the exchange "
            "reads as one",
            "    continuous scene: each line answers the one before it.",
            "  - Most segments carry a line. One or two silent beats are good "
            "for a look",
            "    or a reaction; a scene of people talking that is mostly "
            "silent is not.",
            "  - BUDGET: about %d words across the WHOLE scene. Real dialogue "
            "is short." % total_words,
            "    A reply is three to eight words far more often than it is a "
            "sentence.",
            "  - Nobody makes a speech. If a line runs past about twelve "
            "words, split it",
            "    and let the other character interrupt.",
            "  Per segment, if you do write a line: %s." % per_rung,
        ])

    @staticmethod
    def _overlong(segments, per_second=2.6):
        """Lines that cannot be said inside their own clip.

        The plan is free to write any length, and models routinely write twice
        what fits — 26 words into a 5.167s shot. H3 renders the picture for the
        clip's length and the sentence is simply cut off partway through, which
        is invisible until the stitch. Reported rather than trimmed: cutting
        someone's script by machine is worse than telling them it is long.
        """
        out = []
        for seg in segments:
            line = (seg.get("dialogue") or "").strip()
            if not line:
                continue
            words = len(line.split())
            budget = int(seg["target_duration"] * per_second)
            if words > budget:
                out.append("%s: %d words in %.3fs (about %d fit) — the line "
                           "will be cut off. Shorten it on the card, or give "
                           "the segment a longer rung."
                           % (seg["id"], words, seg["target_duration"], budget))
        return out

    @classmethod
    def _script_shape(cls, segments, total_seconds):
        """Is this a commercial script, or continuous narration?

        Checked in code because the budget is the instruction models ignore
        most reliably, and an over-written script is only obvious once you are
        watching someone talk for thirty seconds without drawing breath.
        """
        out = []
        spoken = [s for s in segments if (s.get("dialogue") or "").strip()]
        if not spoken:
            return out
        words = sum(len(s["dialogue"].split()) for s in spoken)
        budget = max(6, int(total_seconds * cls.WORDS_PER_SECOND_TOTAL))
        if words > budget * 1.25:
            out.append(
                "the script runs %d words across %.0fs; a written commercial "
                "of that length carries about %d. It will play as continuous "
                "narration with no room to breathe — cut roughly %d words, or "
                "clear a segment entirely."
                % (words, total_seconds, budget, words - budget))
        # Both notes below are advice for a commercial with a voice over
        # it. A scene of two people talking is allowed to talk throughout and
        # to end on a line, so they are skipped once more than one character
        # actually speaks.
        talking = len({(x.get("speaker") or "").strip()
                       for x in spoken if (x.get("speaker") or "").strip()})
        if talking > 1:
            return out
        if len(spoken) == len(segments) and len(segments) > 2:
            out.append(
                "every segment speaks. An advert needs at least one silent "
                "beat in the middle for the picture to carry itself.")
        last = segments[-1] if segments else None
        if last is not None and len((last.get("dialogue") or "").split()) > 6:
            out.append(
                "%s is the closing shot and carries %d words. A hero shot "
                "wants silence or a tagline of six words or fewer."
                % (last["id"], len(last["dialogue"].split())))
        return out

    # Words that mean the brief is asking for the audio to be a VOICE, not a
    # track to lay under the picture.
    VOICE_WORDS = ("voice", "speak", "says", "saying", "dialogue", "dialog",
                   "sound like", "sounds like", "timbre", "accent")

    @classmethod
    def _audio_role_mismatch(cls, cast, audio_role, idea):
        """The audio_role widget disagreeing with what was actually asked for.

        Adding the voice-sample role does nothing on its own: a saved workflow
        keeps the value it had. An advert briefed with "use <audio 1> for the
        male and <audio 2> for the female" ran with "performed on camera" and
        came back with both characters lip-syncing one reused track, and
        nothing in the report said why.
        """
        if audio_role == VOICE_SAMPLE:
            return ""
        tags = [m["tag"] for m in (cast or {}).get("members", [])
                if m.get("kind") == "Audio"]
        if not tags:
            return ""
        text = (idea or "").lower()
        named = re.search(r"<[ ]*audio[ ]*[0-9]+[ ]*>", text)
        # Their real brief never says "voice": it says "use <audio 1> for male
        # and <audio 2> for female characters". Assigning an audio tag TO
        # somebody is the same request in different words.
        # Word boundaries written as spaces on purpose: a "\b" in this file
        # became a literal backspace once, and the pattern then matched nothing
        # while reading, in every editor, exactly as intended.
        assigned = re.search(r"<[ ]*audio[ ]*[0-9]+[ ]*>[^.]{0,40}[ ]for[ ]", text)
        asked = named and (assigned
                           or any(w in text for w in cls.VOICE_WORDS))
        if len(tags) > 1:
            why = ("%d audio references are connected (%s). Only one can be "
                   "the soundtrack, so %s will be reused verbatim and the "
                   "rest will do nothing."
                   % (len(tags), ", ".join(tags), tags[0]))
        elif asked:
            why = ("the brief asks for %s to be used as a voice." % tags[0])
        else:
            return ""
        return ("AUDIO ROLE LOOKS WRONG. It is set to %r, which means the "
                "audio IS the soundtrack and is reused exactly as supplied; "
                "every prompt will say so and no dialogue can be generated "
                "over it. %s Set audio_role to %r on this node and queue "
                "again." % (audio_role, why, VOICE_SAMPLE))

    @staticmethod
    def _voices_used(segments, voices):
        """A cast voice that never speaks is a reference doing nothing.

        The whole point of connecting two samples is two characters. Silence
        about it is how an advert came back with one line across five
        segments, all of it from the same mouth.
        """
        speakers = [v for v in (voices or []) if v.get("subject")]
        if len(speakers) < 2:
            return []
        said = {(x.get("speaker") or "").strip() for x in segments}
        return ["%s was cast as the voice of %s, and %s never speaks. "
                "Re-plan, or write a line on one of the cards."
                % (v["audio"], v["subject"], v["subject"])
                for v in speakers if v["subject"] not in said]

    @staticmethod
    def _untagged(segments, cast_tags):
        """Beats that name a cast member in prose instead of by tag."""
        if not cast_tags:
            return []
        out = []
        for seg in segments:
            text = " ".join(filter(None, [seg.get("action"),
                                          seg.get("opens_from"),
                                          seg.get("ends_with")]))
            if text and "<" not in text:
                out.append(seg["id"])
        return out

    # The airlock hold, as a share of the clip rather than a flat two seconds.
    # Two seconds is the Motion Context pack's figure and it assumes a ten
    # second clip. Applied flat to a 5.167s segment it took HALF the runtime,
    # and a five-segment advert spent thirteen of its twenty-six seconds
    # standing still.
    HOLD_SHARE = 0.22
    HOLD_MIN = 0.8
    HOLD_MAX = 2.0

    @classmethod
    def _hold_seconds(cls, duration):
        return round(min(cls.HOLD_MAX,
                         max(cls.HOLD_MIN, float(duration) * cls.HOLD_SHARE)), 3)

    @staticmethod
    def _clock(seconds):
        seconds = max(0.0, float(seconds))
        return "%02d:%06.3f" % (int(seconds // 60), seconds % 60)

    @classmethod
    def enforce_hold(cls, body, hold, render):
        """Pin the first cut to the hold, deterministically.

        The creator's timing pass respaces every shot evenly whenever the
        model's own timestamps are not already valid, so with two shots the cut
        always landed at the exact midpoint whatever the prompt asked for. The
        hold has to be written in code afterwards or it is not really a
        setting.
        """
        marks = list(re.finditer(r"\[\s*Shot\s+(\d+)\s*\]"
                                 r"(?:\s*At\s+\d{1,2}:\d{2}(?:\.\d+)?\s*,?)?",
                                 body or "", re.IGNORECASE))
        if len(marks) < 2:
            return body
        ceiling = max(hold + 0.1, float(render) - 0.05)
        later = len(marks) - 1
        span = max(0.0, ceiling - hold)
        out, cursor = [], 0
        for i, m in enumerate(marks):
            out.append(body[cursor:m.start()])
            if i == 0:
                out.append("[Shot 1]")
            else:
                at = hold + (span * (i - 1) / later if later > 1 else 0.0)
                out.append("[Shot %s] At %s," % (m.group(1), cls._clock(at)))
            cursor = m.end()
        out.append(body[cursor:])
        return re.sub(r"\s{2,}", " ", "".join(out)).strip()

    @staticmethod
    def _shot_budget(seconds):
        """How many internal cuts a clip of this length can carry.

        A shot needs a couple of seconds to read. Without a stated budget the
        model treated every segment as room for the whole advert and packed
        five shots into five seconds.
        """
        seconds = float(seconds or 0)
        if seconds < 6.0:
            return 2
        if seconds < 9.0:
            return 3
        return 4

    @staticmethod
    def _comparable(body, prefix=""):
        """A description reduced to what actually distinguishes it.

        The style prefix is copied verbatim into every segment on purpose, and
        shot markers and timestamps are boilerplate. Leaving them in made two
        genuinely different segments look 90% alike and the duplicate check
        fired on healthy plans.
        """
        text = str(body or "")
        # The continuity sentences are injected identically on purpose, and
        # the opening state is by definition a repeat of the segment before.
        # Left in, they made healthy plans look like the bleed they are not.
        text = re.sub(r"The location is[^.]*[.]", " ", text,
                      flags=re.IGNORECASE)
        text = re.sub(r"The shot opens on[^.]*[.]", " ", text,
                      flags=re.IGNORECASE)
        if prefix:
            text = re.sub(re.escape(prefix.rstrip(" ,")), " ", text,
                          flags=re.IGNORECASE)
        text = re.sub(r"\[\s*Shot\s+\d+\s*\]", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bAt\s+\d{1,2}:\d{2}(?:\.\d+)?", " ", text,
                      flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9 ]+", " ", text.lower())

    @staticmethod
    def _bleed(segments, prefix=""):
        """Segments that swallowed the rest of the film.

        The failure this catches, seen in the wild: one prose call returned
        five objects that were byte-identical, each containing every shot and
        every spoken line of a thirty-second advert. Five clips were then
        rendered of the same thing. Nothing else in the pipeline noticed,
        because each prompt was individually well-formed.
        """
        import difflib

        raw, bodies = [], []
        for seg in segments:
            prompt = seg.get("prompt")
            body = (str(prompt.get("detailed_description", ""))
                    if isinstance(prompt, dict) else "")
            raw.append(body)
            bodies.append(H3PlannerStoryPlanner._comparable(body, prefix))

        bad = {}
        for i, seg in enumerate(segments):
            body = raw[i]
            if not body:
                continue
            # someone else's line, word for word, inside this segment
            for j, other in enumerate(segments):
                if i == j:
                    continue
                line = (other.get("dialogue") or "").strip()
                if len(line.split()) >= 4 and line.lower() in body.lower():
                    bad.setdefault(seg["id"], set()).add(
                        "carries %s's line" % other["id"])
            # a near-copy of another segment, compared on the distinguishing
            # text only, and only when there is enough of it to judge
            mine = bodies[i]
            if len(mine.split()) < 40:
                continue
            for j in range(i + 1, len(segments)):
                if len(bodies[j].split()) < 40:
                    continue
                if difflib.SequenceMatcher(None, mine, bodies[j]).ratio() > 0.9:
                    bad.setdefault(seg["id"], set()).add(
                        "reads the same as %s" % segments[j]["id"])
                    bad.setdefault(segments[j]["id"], set()).add(
                        "reads the same as %s" % seg["id"])
        return {k: sorted(v) for k, v in bad.items()}

    @staticmethod
    def _dialogue_landed(prompt, line):
        """Did the planned line actually reach the prompt as H3 dialogue?

        A line planned in phase one and dropped in phase two is silent on
        screen and invisible in the report, which is exactly how the last
        version shipped without any dialogue at all.
        """
        if not line:
            return True
        body = prompt.get("detailed_description", "")
        if "<d>" not in body:
            return False
        words = [w for w in re.findall(r"[A-Za-z']{4,}", line)][:6]
        if not words:
            return "<d>" in body
        hits = sum(1 for w in words if w.lower() in body.lower())
        return hits >= max(1, len(words) // 2)

    @staticmethod
    def _previous(timeline, seg):
        i = seg["index"]
        return timeline["segments"][i - 1] if i > 0 else None

    @staticmethod
    def _reuse(project, timeline, fingerprint, reuse_existing):
        """Carry over prompts already written from this exact brief."""
        saved = store.load(project["timeline_path"])
        if not saved:
            return 0
        known = {s["id"]: s for s in saved.get("segments", [])}
        kept = 0
        for seg in timeline["segments"]:
            prev = known.get(seg["id"])
            if not prev:
                continue
            if prev.get("state") == store.LOCKED:
                # A lock carries the prompt with it, whatever the fingerprint
                # says. Honouring the state but dropping the text left a locked
                # segment empty the moment the brief changed — and the write
                # loop then skips locked segments, so nothing ever refilled it.
                # A hand-written prompt is exactly what a lock exists to keep.
                seg["state"] = store.LOCKED
                if prev.get("prompt"):
                    seg["prompt"] = prev["prompt"]
                    seg["prompt_fingerprint"] = prev.get(
                        "prompt_fingerprint", "")
                    seg["spec_hash"] = store.spec_hash(seg)
                    kept += 1
                continue
            if (reuse_existing and prev.get("prompt")
                    and prev.get("prompt_fingerprint") == fingerprint):
                seg["prompt"] = prev["prompt"]
                seg["prompt_fingerprint"] = fingerprint
                seg["spec_hash"] = store.spec_hash(seg)
                kept += 1
        return kept

    @staticmethod
    def _beat_sheet(timeline, prefix, world, canon):
        lines = ["STYLE PREFIX", "  " + (prefix or "(none)"), "",
                 "WORLD", "  " + (world or "(none)").replace("\n", "\n  "), "",
                 "CAST", "  " + (canon or "(none)").replace("\n", "\n  "), "",
                 "BEATS"]
        scene = None
        for seg in timeline["segments"]:
            if seg.get("scene") != scene:
                scene = seg.get("scene")
                lines.append("")
                lines.append("  SCENE %s%s" % (
                    scene, (" — " + seg["scene_name"]) if seg.get("scene_name") else ""))
            lines += [
                "    %s  %6.3fs  %s" % (seg["id"], seg["target_duration"],
                                        seg.get("action") or seg.get("beat") or ""),
                "              opens: %s" % (seg.get("opens_from") or "-"),
                "              ends:  %s" % (seg.get("ends_with") or "-"),
            ]
            if seg.get("dialogue"):
                lines.append('              says:  "%s"' % seg["dialogue"])
        return "\n".join(lines)

    @staticmethod
    def _report(timeline, project, prefix, world, cast_block, format, rungs,
                calls, reused, failed, notes, total, elapsed,
                audio_role="background only", voices=()):
        segments = timeline["segments"]
        planned = sum(s["target_duration"] for s in segments)
        rendered = sum(s["render_duration"] for s in segments)
        scenes = sorted({s.get("scene") for s in segments if s.get("scene")})
        written = sum(1 for s in segments if s.get("prompt"))

        rows = []
        for seg in segments:
            rows.append("  %-8s sc%-3s %6.3fs -> %6.3fs (%3d f)  %-8s %s"
                        % (seg["id"], seg.get("scene") or "-",
                           seg["target_duration"], seg["render_duration"],
                           seg["render_frames"], seg["link"],
                           "written" if seg.get("prompt") else "NOT WRITTEN"))

        return "\n".join([
            "%d segment(s) across %d scene(s) — %d written, %d reused, "
            "%d model call(s) in %.0fs"
            % (len(segments), len(scenes), written, reused, calls, elapsed),
            "format        %s" % format,
            "style prefix  %s" % (prefix or "(none)"),
            "world         %s" % ((world or "(none)").splitlines() or ["(none)"])[0][:88],
            "cast          %s" % (cast_block.replace("\n", " | ")[:88]),
            "clip lengths  %s" % ", ".join("%.3fs" % s for _, s in rungs),
            "dialogue      %s" % (
                "%d of %d segment(s) speak, %d words"
                % (sum(1 for x in segments if x.get("dialogue")),
                   len(segments),
                   sum(len((x.get("dialogue") or "").split())
                       for x in segments))
                if any(x.get("dialogue") for x in segments)
                else "none - picture only"),
            # Who ended up speaking, because the casting is the whole feature
            # and it is invisible in the prompts unless you read all of them.
            "voices        %s" % (
                "; ".join("%s = %s %s (%d line(s))"
                          % (v["audio"], v["subject"], v["speaker"],
                             sum(1 for x in segments
                                 if (x.get("speaker") or "").strip()
                                 == v["subject"]))
                          for v in voices)
                if audio_role == VOICE_SAMPLE and voices
                else "n/a - the audio is the soundtrack, not a voice sample"),
            "planned       %.3fs against a %.3fs brief (%+.3fs)"
            % (planned, total, planned - total),
            "renders       %.3fs — the stitcher trims %.3fs back off"
            % (rendered, rendered - planned),
            "",
            "segments:",
        ] + rows
            + (["", "NOTES", "- " + "\n- ".join(notes)] if notes else [])
            + (["", "FAILURES", "! " + "\n! ".join(failed)] if failed else []))


NODE_CLASS_MAPPINGS = {"H3PlannerStoryPlanner": H3PlannerStoryPlanner}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PlannerStoryPlanner": "H3 Story Planner"}
