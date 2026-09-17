"""H3 Segment Prompter — one long prompt in, per-segment prompts out.

Format-neutral by design. Short films, ads, UGC, product films, explainers,
music videos: the job is the same either way — take the treatment, decide
where the cuts go, and write a complete standalone H3 prompt for each piece.
The `format` widget nudges the writing; nothing else in the node cares.

Why the prompts are rewritten rather than sliced: a slice of a long prompt is
not a valid H3 prompt. It has no style prefix, no subject_definitions of its
own, no soundscape, and its timestamps start wherever the cut fell.
"""

import difflib
import hashlib
import json
import math
import re
import time

from . import engine, ladder, splitter, store, vocals
from .nodes_plan import CATEGORY, cited_tags, flatten_prompt

SECTIONS = ("subject_definitions", "summary", "retention_analysis",
            "detailed_description", "overall_soundscape", "non_diegetic_music")

SEGMENT_SYSTEM = """
You are writing ONE segment of a longer MiniMax H3 video. Follow the supplied
Full-Reference Mode guide exactly, with these additional constraints, which
override anything that assumes you are writing a whole film:

THIS SEGMENT IS A COMPLETE, STANDALONE H3 PROMPT. It is rendered on its own,
with no knowledge of any other segment. Every section must be written in full.

STYLE PREFIX. The supplied style prefix must appear verbatim at the very start
of [Shot 1], before anything else. This is what keeps every segment of the
finished video looking like the same production. Never paraphrase it, never drop
it, never substitute a different style vocabulary.

TIMING. This segment starts at 0.000. Every timestamp is relative to this
segment, never to the whole video. [Shot 1] carries NO timestamp. Every later
shot is "[Shot N] At MM:SS.mmm" with strictly increasing times, all strictly
less than the segment's rendered duration. Number the shots from 1 — the source
shot numbers are for your reference only and must not appear.

HOLD TAIL. The rendered duration is slightly longer than the planned duration,
because H3 only accepts certain lengths. Every essential action must complete by
the planned duration; whatever remains is a hold on the final composition. Never
start a new action inside the hold.

REFERENCES. Cite only the tags listed in the supplied cast. Never invent a tag,
and never cite a number the cast does not contain. subject_definitions lists only
the tags THIS segment actually cites — not the whole cast. In particular, do not
write <Video N> unless a video asset is listed: pacing, cut structure and
editing rhythm are things you decide, not things a reference supplies.

ASSET NAMES. Refer to every asset by its tag alone. Never write a file name, a
track title or the cast key anywhere in the prompt — not in a parenthesis after
a tag, not in any section. "<Audio 1>" is correct; "<Audio 1> (my_track_02)" is
not.

SPEAKERS. Speaker IDs are supplied from the segments already written. A speaker
who already has an ID keeps it. Only assign a new ID to a voice not heard yet.

CONTINUITY. Do not summarise the story, do not recap earlier segments, and do not
foreshadow later ones. Write only what happens on screen in this segment. When a
spoken line runs across the cut into the next segment, use <scenetrans> at the
boundary and say the audio continues across the cut.
""".strip()

# What each kind of video needs from a single segment. Deliberately short —
# these steer emphasis, they do not rewrite the guide.
FORMAT_GUIDANCE = {
    "auto": "",
    "short film": """
This is a segment of a narrative short. Carry the scene's dramatic beat: what
changes for the character between the first frame and the last. Blocking,
eyelines and screen direction must stay consistent so the cut reads as one
scene. Performance detail beats camera flourish.""",
    "cinematic ad": """
This is a segment of a cinematic commercial. Every shot earns its place: strong
composition, deliberate camera movement, controlled light. Keep the product or
subject legible in frame. No dead air — a commercial second is expensive.""",
    "ugc": """
This is a segment of UGC-style content. Hand-held or phone-mounted framing,
natural available light, imperfect composition, direct address to the lens.
Avoid cinematic grading and crane moves — they break the format instantly.""",
    "product": """
This is a segment of a product film. The product is the subject: keep it sharp,
correctly proportioned and unobstructed. Describe surface, material and finish
precisely. Camera moves should reveal form rather than decorate the shot.""",
    "explainer": """
This is a segment of an explainer. Clarity over atmosphere. If on-screen text or
a graphic element appears, quote it verbatim in English double quotation marks.
Hold shots long enough to be read; avoid motion that competes with the point.""",
    "music video": """
This is a segment of a music video. Cut and camera movement should sit with the
track's rhythm. If a performer is audible, describe the mouth matching the words
rather than only moving to the beat.""",
}

FORMATS = list(FORMAT_GUIDANCE.keys())

# Appended whenever the cast carries an audio asset — driven by what is
# connected, not by what kind of video this is. A supplied track already
# contains everything that will be heard, so any sound the prompt *describes*
# is sound H3 is being asked to add on top of it.
#
# Whether a visible subject *produces* that audio is a separate question, and
# the user's call rather than something to infer: the same track can be a song
# the subject raps on camera, a narration nobody on screen speaks, or a bed
# that is simply heard.
AUDIO_ROLES = ("performed on camera", "voice-over (off camera)",
               "background only")

# Whether anyone speaks on camera is a third, separate question from where the
# audio comes from. H3 can generate a voice with no audio input at all, so this
# applies whether or not an audio asset is connected.
DIALOGUE_MODES = ("auto", "spoken lines", "none")

AUDIO_EXACT = """
AN AUDIO ASSET IS CONNECTED. It is used exactly as supplied.

- Do not invent ambience, traffic, train noise, crowd chatter, vendor calls,
  room tone, percussion, basslines or any other sound: the supplied audio
  already contains everything that is heard, and describing extra sound makes
  H3 generate it on top.
- Do not characterise the audio's genre, instrumentation, tempo or mood, and do
  not name it or write its file name.
- The retention marker for the audio tag is the strongest one available — the
  same signal is reused, not approximated.
""".strip()

# The crucial distinction. "Do not invent sound" and "do not say who is
# performing" are different instructions, and running them together produced
# prompts with the track attached and a subject sitting silently under it:
# describing a performance is picture direction, not a description of sound.
AUDIO_ROLE_GUIDANCE = {
    "performed on camera": """
THE AUDIO IS PERFORMED ON CAMERA. A visible subject produces the voice in the
supplied audio, and saying so is the single most important thing this prompt
does. Left out, H3 renders someone standing or sitting in silence while the
track plays over them.

- In detailed_description, state plainly that the subject is rapping, singing
  or speaking, and that their mouth matches the words. Do this in every shot
  where they are on screen and audible — not once at the top.
- Where the source material gives the actual words for a moment, keep them in
  the description as the line being performed, in quotation marks, attached to
  the subject performing them. Do not reduce the words to a camera cue.
- Describe the physical performance: jaw and lip movement, breath, the head and
  hands carrying the rhythm. That is picture direction, so it does not conflict
  with the rules above about not describing sound.
""",
    "voice-over (off camera)": """
THE AUDIO IS A VOICE-OVER. A voice is heard but nobody on screen produces it.
No subject's mouth matches it: do not write lip movement, singing or anyone
performing the words.
""",
    "background only": """
THE AUDIO IS BACKGROUND ONLY. Nobody on screen produces it and no mouth matches
it. Describe picture only.
""",
}

# Used in place of the role's guidance for a clip the vocal timeline says is
# instrumental. "Performed on camera" is true of the song and false of this
# clip: left to the role's guidance, the model writes the singer mouthing words
# over a passage with no words in it.
INSTRUMENTAL_GUIDANCE = """
THIS CLIP IS AN INSTRUMENTAL PASSAGE. The supplied audio has no voice in it
here. Nobody sings, raps, speaks or lip-syncs, and every mouth stays closed —
including the lead performer's. Show the performers playing, moving, reacting,
or the camera exploring the scene. Write no dialogue and no <d> tags.
"""


def audio_system_block(audio_tag, role, clip_vocals=None):
    """The audio rules for this cast and this role, or "" when no audio.

    ``clip_vocals`` is this clip's state from the treatment's vocal timeline;
    an instrumental clip gets instrumental guidance whatever the role says.
    """
    if not audio_tag:
        return ""
    if clip_vocals and clip_vocals["state"] == vocals.INSTRUMENTAL:
        guidance = INSTRUMENTAL_GUIDANCE.strip()
    else:
        guidance = AUDIO_ROLE_GUIDANCE.get(role, "").strip()
    return AUDIO_EXACT + ("\n\n" + guidance if guidance else "")

TAG_RE = re.compile(r"<\s*(Subject|Picture|Video|Audio)\s*(\d+)\s*>", re.IGNORECASE)
SHOT_MARKER_RE = re.compile(r"\[\s*Shot\s+\d+\s*\]", re.IGNORECASE)


def asset_names(cast):
    """File stems worth scrubbing out of a finished prompt.

    Only the uploaded file names — a user-typed key like "hero" is an ordinary
    word and must never be stripped out of prose.
    """
    import os
    names = set()
    for m in (cast or {}).get("members", []):
        stem = os.path.splitext(str(m.get("file") or ""))[0]
        if len(stem) >= 4:
            names.add(stem)
    return names


def strip_asset_names(text, names=()):
    """Remove file names the model copied out of the cast list.

    Two forms: a parenthesis straight after a tag holding something that looks
    like a file name, and any bare occurrence of a known upload's stem.
    """
    # A parenthesis straight after a tag holding something file-name shaped.
    # The lookahead spares speaker IDs: the guide's own dialogue form is
    # "<Subject 2> (S1) turns and says", and (S1) is a parenthesis after a tag
    # containing a digit, so without this it was deleted and every spoken line
    # lost the voice it belonged to.
    text = re.sub(
        r"(<\s*(?:Subject|Picture|Video|Audio)\s*\d+\s*>)\s*"
        r"\((?!\s*S\d+(?:\s*,\s*S\d+)*\s*\))\s*"
        r"([^)\s]*[_\d][^)]*)\)",
        r"\1", text, flags=re.IGNORECASE)
    for name in sorted(names, key=len, reverse=True):
        text = re.sub(r"\s*\(\s*%s[^)]*\)" % re.escape(name), "", text,
                      flags=re.IGNORECASE)
        text = re.sub(re.escape(name), "", text, flags=re.IGNORECASE)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r"\s+([,.;:])", r"\1", text).strip()


# Sound the model adds when it is describing a scene rather than reproducing a
# supplied track. Every one of these is sound H3 would then generate on top of
# the audio that was handed to it.
INVENTED_SOUND = (
    "background noise", "ambient", "ambience", "room tone", "traffic",
    "chatter", "vendor call", "train sound", "crowd noise", "footstep",
    "percussion", "bassline", "bass line", "instrumentation", "layered",
    "distant", "hum of", "murmur",
)

# A sentence carrying any of these is describing a person performing, which is
# picture direction and must survive the ambience strip above.
PERFORMANCE_WORDS = (
    "mouth", "lips", "lip-sync", "lip sync", "lipsync", "jaw", "teeth",
    "raps", "rapping", "sings", "singing", "sung", "speaks", "speaking",
    "mouthing", "mouths", "vocal", "verse", "chorus", "lyric", "breath",
    "delivers the line", "to camera",
)


def soundscape_line(audio_tag, role="background only", subject_tag="",
                    clip_vocals=None):
    """What overall_soundscape is replaced with, for this audio role.

    The old single line said only that no extra sound is present. That is true
    and it is also the whole problem: it overwrote the one place a prompt says
    a voice belongs to a person on screen, so H3 had nothing telling it to
    animate a performance.
    """
    head = ("All audio in this segment comes from %s and is used exactly as "
            "supplied" % audio_tag)
    who = subject_tag or "the subject on screen"
    if clip_vocals and clip_vocals["state"] == vocals.INSTRUMENTAL:
        return ("%s: this passage is instrumental, so nobody on screen sings or "
                "raps and every mouth stays closed. No additional sound is "
                "present." % head)
    if (clip_vocals and clip_vocals["state"] == vocals.MIXED
            and role == "performed on camera"):
        spans = " and from ".join(
            "%s to %s" % (vocals.timestamp(a), vocals.timestamp(b))
            for a, b in clip_vocals["spans"])
        return ("%s: the vocal is performed on camera by %s only from %s, "
                "whose mouth matches the words then; the rest is instrumental "
                "and every mouth stays closed. No additional sound is present."
                % (head, who, spans))
    if role == "performed on camera":
        return ("%s: the vocal is performed on camera by %s, whose mouth "
                "matches the words throughout. No additional sound is present."
                % (head, who))
    if role == "voice-over (off camera)":
        return ("%s, heard as voice-over; no one on screen produces it and no "
                "mouth matches it. No additional sound is present." % head)
    return "%s. No additional sound is present." % head


def enforce_audio_exact(prompt, audio_tag, role="background only",
                        subject_tag="", clip_vocals=None):
    """Make the two sound sections say the audio is reused, and nothing else.

    A supplied track already contains everything that will be heard. Left
    alone the model writes "ambient street sounds blend with rhythmic beats,
    featuring layered percussion" — a description of sound to synthesise, on
    top of the sound it was given.

    ``role`` says whether a visible subject produces that audio. It changes
    only what the soundscape line asserts; the ban on inventing sound is the
    same in every role.

    ``clip_vocals`` is where the song is sung inside this clip, from the
    treatment's vocal timeline. The role describes the whole song; this says
    whether this clip has any singing in it, and wins where they disagree.
    """
    changed = []
    exact = soundscape_line(audio_tag, role, subject_tag, clip_vocals)
    if prompt.get("overall_soundscape", "").strip() != exact:
        prompt["overall_soundscape"] = exact
        changed.append("overall_soundscape")

    music = "N/A — the audio is supplied by %s." % audio_tag
    if prompt.get("non_diegetic_music", "").strip() != music:
        prompt["non_diegetic_music"] = music
        changed.append("non_diegetic_music")

    # The retention line decides how literally H3 takes the track. Left alone
    # the model writes "reference - style and rhythm guide the performance
    # without copying exact lyrics", which is the opposite of using it exactly.
    retention = prompt.get("retention_analysis", "")
    canonical = "%s: fully_copy, all shots." % audio_tag
    line = re.compile(r"[^.\n]*%s[^.\n]*\.?" % re.escape(audio_tag))
    match = line.search(retention)
    if match:
        if match.group(0).strip() != canonical:
            # Keep the sentence boundary intact. Splicing straight onto the
            # previous full stop glues two entries into one, and anything that
            # later drops a line by sentence takes both.
            prompt["retention_analysis"] = " ".join(filter(None, [
                retention[:match.start()].strip(),
                canonical,
                retention[match.end():].strip()]))
            changed.append("audio retention -> fully_copy")
    else:
        prompt["retention_analysis"] = (
            (retention.strip() + " " + canonical)
            if retention and retention != "N/A" else canonical)
        changed.append("audio retention added")

    # And drop invented ambience from the description, but only from sentences
    # that are purely about sound — never from one carrying a shot or a subject,
    # and never from one describing the performance. A line like "his breath
    # carries the rhythm, the vocal distant in the mix" is picture direction
    # that happens to contain a banned word; deleting it is how the lip sync
    # went missing.
    body = prompt.get("detailed_description", "")
    kept, dropped = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        low = sentence.lower()
        if (any(term in low for term in INVENTED_SOUND)
                and not SHOT_MARKER_RE.search(sentence)
                and not TAG_RE.search(sentence)
                and not any(word in low for word in PERFORMANCE_WORDS)):
            dropped += 1
            continue
        kept.append(sentence)
    if dropped:
        prompt["detailed_description"] = " ".join(
            s.strip() for s in kept if s.strip())
        changed.append("%d invented-sound sentence(s)" % dropped)

    if clip_vocals:
        body, sung = vocals.apply(prompt.get("detailed_description", ""),
                                  audio_tag, clip_vocals)
        prompt["detailed_description"] = body
        if sung:
            changed.append("%d singing sentence(s) in an instrumental clip" % sung)
    return prompt, changed


def subject_lines(text):
    """subject_definitions split into one entry per tag it defines."""
    lines = []
    for chunk in re.split(r"(?:\r?\n|(?<=[.!?])\s+)", text or ""):
        chunk = chunk.strip()
        if chunk:
            lines.append(chunk)
    return lines


def canonical_subjects(canon, cited):
    """The shared definitions, cut down to the tags this segment cites.

    Each segment is written by its own model call, so each one re-words the
    cast: the hero is "a young man in a black hooded jacket" in one segment and
    "a man wearing dark outerwear" in the next. H3 then generates a slightly
    different person every time, and by segment six it is visibly not the same
    character. Copying ONE canonical block into every segment is the only way
    the wording can be identical, because asking for it does not work.

    ``cited`` is {kind: {numbers}} from the finished prompt.
    """
    wanted = set()
    for kind, numbers in (cited or {}).items():
        for n in numbers:
            wanted.add("<%s %d>" % (kind, n))
    if not wanted:
        return ""

    kept = []
    for line in subject_lines(canon):
        tags = {"<%s %d>" % (k.capitalize(), int(n))
                for k, n in TAG_RE.findall(line)}
        # A line belongs to this segment when the tag it DEFINES is cited —
        # that is the first tag on the line.
        first = next(iter(TAG_RE.findall(line)), None)
        if first is None:
            continue
        defines = "<%s %d>" % (first[0].capitalize(), int(first[1]))
        if defines in wanted or (tags & wanted and defines in wanted):
            kept.append(line)
    return "\n".join(kept)


SUBJECT_DEF_RE = re.compile(r"<?\s*Subject\s*(\d+)\s*>?", re.IGNORECASE)


def defined_subjects(text):
    """Subject numbers a subject_definitions block defines.

    Subjects are not assets. The Cast Board tags images as <Picture N>, and the
    analysis then finds the people and objects inside them, as many as there
    are: one stage photo can hold a whole band. Capping <Subject N> at the
    number of images stripped the third band member out of a two-picture cast.
    """
    return {int(n) for n in SUBJECT_DEF_RE.findall(text or "")}


def with_defined_subjects(prompt, allowed):
    """``allowed`` widened by every subject the prompt itself defines."""
    found = defined_subjects((prompt or {}).get("subject_definitions", ""))
    if not found:
        return allowed
    wider = {kind: set(numbers) for kind, numbers in (allowed or {}).items()}
    wider.setdefault("Subject", set()).update(found)
    return wider


def bind_tags(prompt, allowed):
    """Put the angle brackets back on tags the model wrote as plain words.

    H3 binds a reference image to the text through the bracket form. A model
    that writes "Subject 1 walks toward the camera" has produced ordinary
    prose: the reference is attached to the sampler, resized and wired, and
    nothing in the prompt ever asks for it. That is silent — the prompt reads
    perfectly, every section validates, and the rendered video simply contains
    a different person.

    Seen in the wild with subject_definitions and retention_analysis correctly
    bracketed while detailed_description, the section that actually matters for
    binding, used bare words throughout.

    Only tags the cast really has are bound, so this can never invent one.
    """
    allowed = with_defined_subjects(prompt, allowed)
    if not allowed:
        return prompt, []
    bound, out = [], {}
    for key, text in prompt.items():
        value = text or ""
        for kind, numbers in allowed.items():
            for number in sorted(numbers):
                # Not already inside a tag: no "<" before, no word char or ">"
                # after, so "<Subject 1>" and "Subject 12" are both left alone.
                pattern = re.compile(
                    r"(?<![<\w])%s\s*%d(?![\w>])" % (re.escape(kind), number),
                    re.IGNORECASE)
                value, hits = pattern.subn("<%s %d>" % (kind, number), value)
                if hits:
                    bound.append("%s x%d in %s" % ("<%s %d>" % (kind, number),
                                                   hits, key))
        out[key] = value
    return out, bound


def strip_unknown_tags(prompt, allowed):
    """Delete citations of tags the cast does not contain.

    The model reaches for <Video 1> to explain pacing even when no video is
    connected, and H3 then looks for a reference asset that was never wired.
    A sentence carrying a shot marker keeps its text and loses only the tag —
    dropping it whole would lose a shot.
    """
    allowed = with_defined_subjects(prompt, allowed)
    removed, cleaned = [], {}
    for key, text in prompt.items():
        if key == "retention_analysis":
            parts = re.split(r"(?:\r?\n|(?<=[.!?])\s+)", text)
        else:
            parts = re.split(r"(?<=[.!?])\s+", text)

        kept = []
        for part in parts:
            bad = [(k.capitalize(), int(n)) for k, n in TAG_RE.findall(part)
                   if int(n) not in allowed.get(k.capitalize(), set())]
            if not bad:
                kept.append(part)
                continue
            removed.extend("<%s %d>" % b for b in bad)
            if SHOT_MARKER_RE.search(part):
                for kind, number in bad:      # keep the shot, drop the tag
                    part = re.sub(r"<\s*%s\s*%d\s*>" % (kind, number), "",
                                  part, flags=re.IGNORECASE)
                kept.append(re.sub(r"[ \t]{2,}", " ", part))
        joined = " ".join(p.strip() for p in kept if p.strip()).strip()
        cleaned[key] = joined or "N/A"
    return cleaned, sorted(set(removed))


def renumber_shots(text):
    """Force [Shot N] to run 1, 2, 3 within this segment.

    The model keeps the treatment's original numbering — a segment cut from
    shots 4-6 comes back saying "[Shot 4]" — and H3 reads that as a video that
    starts on its fourth shot.
    """
    counter = [0]

    def bump(match):
        counter[0] += 1
        return match.group(0).replace(match.group(1), str(counter[0]), 1)

    return re.sub(r"\[\s*Shot\s+(\d+)\s*\]", bump, text, flags=re.IGNORECASE)


def _vocal_summary(audio_tag, sections, counts):
    """The report's vocals line."""
    if not audio_tag:
        return "(no audio)"
    if not sections:
        return ("no vocal timeline in the treatment — every clip follows "
                "audio_role alone, so a singer may be shown mouthing words over "
                "an instrumental. Connect the song to the H3 Full-Reference "
                "Prompt Creator and re-run it to get one.")
    sung = sum(s["end"] - s["start"] for s in sections
               if s["kind"] == vocals.VOCALS)
    total = sections[-1]["end"]
    return ("timeline from the treatment, %.1fs sung of %.1fs — %d sung, %d "
            "instrumental, %d mixed clip(s)"
            % (sung, total, counts.get(vocals.VOCALS, 0),
               counts.get(vocals.INSTRUMENTAL, 0), counts.get(vocals.MIXED, 0)))


def _hash(*parts):
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _cast_block(cast):
    if not cast or not cast.get("members"):
        return "(no references connected — write a text-only segment, cite no tags)"
    return "\n".join("%s = %s (%s)%s"
                     % (m["tag"], m["key"], m["role"],
                        ("; " + m["note"]) if m.get("note") else "")
                     for m in cast["members"])


def _allowed_tags(cast):
    """Which tags this cast can legitimately be cited by.

    A wired image slot backs both readings: <Picture N> is the literal frame,
    <Subject N> is the identity abstracted from it, and the guide expects a
    subject to name the picture it came from. So an image slot permits both.
    Video and Audio stay strict — those are the ones the model invents.
    """
    allowed = {}
    for m in (cast or {}).get("members", []):
        kind, _, number = m["tag"].strip("<>").partition(" ")
        kind, number = kind.capitalize(), int(number)
        allowed.setdefault(kind, set()).add(number)
        if kind in ("Subject", "Picture"):
            slot = int(m.get("slot") or number)
            allowed.setdefault("Subject", set()).add(slot)
            allowed.setdefault("Picture", set()).add(slot)
    return allowed


def _even_segments(total_seconds, segment_seconds, ceiling):
    """No treatment: lay out even beats to write into."""
    target = min(float(segment_seconds), ceiling)
    count = max(1, int(math.ceil(float(total_seconds) / target)))
    each = float(total_seconds) / count
    return [{
        "id": "seg_%02d" % (i + 1),
        "beat": "beat %d of %d" % (i + 1, count),
        "target_duration": round(each, 3),
        "link": "cut",
        "prompt": "",
        "audio_start": round(i * each, 3),
    } for i in range(count)]


# --------------------------------------------------------------------------
# idea mode: one plan for the whole video before any segment is written
# --------------------------------------------------------------------------
#
# Written segment by segment from an idea alone, an 86-second music video came
# back with a lead singer who was a guitarist in four segments, a character
# sheet used as the opening frame of seven, and singing over an intro the idea
# said was instrumental. Each call was blind to the others and to the pictures.
# The plan fixes who is who and what happens where, once; every segment is then
# written from it.

PLAN_SYSTEM = """
You are planning a video that will be generated in separate clips, one clip per
segment. You are NOT writing the clips. You decide, once for the whole video:

1. THE CAST. Look at the attached pictures. Every person or object that appears
   on screen gets exactly one <Subject N>, numbered from 1, and keeps that
   number for the whole video. For each one give:
   - "tag": "<Subject N>"
   - "who": their part in the video, e.g. "lead singer", "drummer", "the product"
   - "from": the <Picture N> they are taken from (use the picture numbers in the
     cast list exactly)
   - "looks": what they look like IN THAT PICTURE, precisely: face, hair, skin,
     build, clothing, accessories, instrument or object. Describe only what the
     picture shows. One sentence, no full stops inside it.
   Where the user's idea gives a role (lead singer, guitarist), match it to the
   person in the picture who fits it.

2. THE SEGMENTS. Exactly the number requested, in order. For each:
   - "index": its number, from 1
   - "action": two or three sentences of what happens on screen in this segment
     and how the camera covers it. Different from the segments around it.
   - "featured": the <Subject N> tags on screen in this segment
   - "vocals": "instrumental", "sung" or "mixed" for this segment's stretch of
     the audio, going ONLY by what the user's idea says about the audio.

3. "performance": the verb for the voice in the audio if someone performs it
   on camera: "sings", "raps" or "speaks". Take it from the idea (a metal or rock
   song is sung). Empty when nobody performs.
   "performer": the <Subject N> who performs that voice (the lead singer, the
   rapper), or empty. Only this subject's mouth ever moves to the words.

4. "vocal_sections": ONLY if the user's idea states where the audio has vocals
   or is instrumental (for example "the first 28 seconds are instrumental"),
   list those spans on the whole video's clock as
   {"start": seconds, "end": seconds, "kind": "instrumental" or "vocals"}.
   Otherwise return an empty list. Never guess this from the music itself.

Follow the PICTURE JOBS exactly: a picture marked IDENTITY only tells you what
someone or something looks like; it is never a location, a frame or a
composition. Obey every instruction in the user's idea.
Return JSON only.
""".strip()


def plan_schema():
    return engine.json_schema({
        "cast": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tag": {"type": "string"},
                    "who": {"type": "string"},
                    "from": {"type": "string"},
                    "looks": {"type": "string"},
                },
                "required": ["tag", "who", "from", "looks"],
            },
        },
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "action": {"type": "string"},
                    "featured": {"type": "array", "items": {"type": "string"}},
                    "vocals": {"type": "string"},
                },
                "required": ["index", "action", "featured", "vocals"],
            },
        },
        "performance": {"type": "string"},
        "performer": {"type": "string"},
        "vocal_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "kind": {"type": "string"},
                },
                "required": ["start", "end", "kind"],
            },
        },
    }, ["cast", "segments", "performance", "vocal_sections"])


# What a picture is FOR, from the role the user gave it on the Cast Board.
# Told only "<Picture 2> = character sheet (character)", the model opened seven
# segments on "the main stage from <Picture 2>".
IDENTITY_ROLES = ("character", "product", "wardrobe", "prop")
FRAME_ROLES = ("composition", "environment", "first_frame", "last_frame",
               "keyframe")


_KEEP_RE = re.compile(r"\[\[H3KEEP[^\]]*\]\]\.?\s*")


def render_keeping_labels(prompt, duration):
    """The prompt engine's render and label pass, with every tag's number kept.

    That pass renumbers <Picture N>, <Subject N>, <Audio N> and <Video N> by
    first appearance. subject_definitions comes first and named the character
    sheet, <Picture 2>, so the sheet became <Picture 1> and the stage became
    <Picture 2>: every prompt described the two images the wrong way round.
    A list of every tag in numeric order, placed ahead of everything, makes
    first appearance the numeric order, so nothing moves; it is removed after.
    """
    tops = {}
    for text in prompt.values():
        for kind, n in TAG_RE.findall(text or ""):
            kind = kind.capitalize()
            tops[kind] = max(tops.get(kind, 0), int(n))
    if not tops:
        return splitter.parse_sections(engine.normalize_labels(
            engine.render_full_ref(prompt, duration), duration))
    keep = "[[H3KEEP %s]]. " % " ".join(
        "<%s %d>" % (kind, n) for kind in sorted(tops)
        for n in range(1, tops[kind] + 1))
    # The definitions stay out of the engine's render: its dedupe renumbers
    # them 1, 2, 3 in order, so a segment defining <Subject 2> and
    # <Subject 4> came back defining <Subject 1> and <Subject 2> while its
    # description still cited 2 and 4.
    marked = dict(prompt)
    marked["subject_definitions"] = keep
    rendered = engine.normalize_labels(
        engine.render_full_ref(marked, duration), duration)
    parsed = splitter.parse_sections(_KEEP_RE.sub("", rendered))
    parsed["subject_definitions"] = (
        clean_definitions(prompt.get("subject_definitions")) or "N/A")
    return parsed


def dedupe_definitions(text):
    """One definition line per subject, numbers untouched.

    The prompt engine's own dedupe renumbers what it keeps from 1.
    """
    out, seen = [], set()
    for line in subject_lines(text):
        first = next(iter(TAG_RE.findall(line)), None)
        key = (int(first[1]) if first and first[0].lower() == "subject"
               else line.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)


def clean_definitions(text):
    """subject_definitions deduped, with sheet and view wording removed.

    The view wording pass works a line at a time, and a line it would strip of
    its own tag is kept as written: a definition that vanishes leaves its
    subject cited and undefined.
    """
    text = (text or "").strip()
    if not text or text == "N/A":
        return ""
    strip_views = getattr(engine.creator(), "_strip_view_language", None)
    lines = []
    for line in subject_lines(dedupe_definitions(text)):
        first = next(iter(TAG_RE.findall(line)), None)
        if strip_views and first:
            try:
                cleaned = strip_views(line)
            except Exception:
                cleaned = line
            again = next(iter(TAG_RE.findall(cleaned or "")), None)
            if again and again[1] == first[1] and again[0].lower() == first[0].lower():
                line = cleaned
        lines.append(line)
    return "\n".join(lines)


def picture_jobs(cast):
    """{picture number: "identity" | "frame" | "style"} from Cast Board roles."""
    jobs = {}
    for m in (cast or {}).get("members", []):
        if m.get("kind") not in ("Picture", "Subject"):
            continue
        number = int(m.get("slot") or m.get("number") or 0)
        if not number:
            continue
        role = m.get("role") or "character"
        jobs[number] = ("identity" if role in IDENTITY_ROLES
                        else "frame" if role in FRAME_ROLES else "style")
    return jobs


def picture_jobs_block(cast):
    """The PICTURE JOBS lines a planner or writer is given, or ""."""
    wording = {
        "identity": "IDENTITY reference: only what the people or objects in it "
                    "look like. Never a location, a frame or a composition, and "
                    "never the picture a shot opens on.",
        "frame": "FRAME reference: the place, set and composition the shots "
                 "are staged in.",
        "style": "STYLE reference: the look, grade and texture only.",
    }
    names = {int(m.get("slot") or m.get("number") or 0): m.get("key", "")
             for m in (cast or {}).get("members", [])
             if m.get("kind") in ("Picture", "Subject")}
    jobs = picture_jobs(cast)
    if not jobs:
        return ""
    return "\n".join("<Picture %d> (%s) = %s" % (n, names.get(n, ""),
                                                  wording[jobs[n]])
                     for n in sorted(jobs))


_ONE_TAG_RE = re.compile(r"<?\s*(Subject|Picture)\s*(\d+)\s*>?", re.IGNORECASE)


def _tag_number(value, kind):
    match = _ONE_TAG_RE.search(str(value or ""))
    if match and match.group(1).lower() == kind.lower():
        return int(match.group(2))
    return None


def _one_sentence(text, limit=400):
    """A model sentence made safe to sit on one definition line.

    subject_lines() splits definitions on full stops, and every part after the
    first carries no tag, so it was dropped: the looks after the first full
    stop never reached a segment.
    """
    text = re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".")
    text = re.sub(r"(?<=[^\d])[.!?]\s+", "; ", text)
    return text[:limit].rstrip(" ;,")


def validate_plan(obj, count, cast, total_seconds):
    """The plan, checked and repaired in code. Returns (plan, problems).

    ``plan`` is {"cast": [...], "beats": [count rows or None], "performance",
    "vocal_sections"}. A missing beat is None and is reported, never invented.
    """
    problems = []
    obj = obj if isinstance(obj, dict) else {}
    pictures = set(picture_jobs(cast))

    rows, seen = [], set()
    for entry in obj.get("cast") or []:
        if not isinstance(entry, dict):
            continue
        number = _tag_number(entry.get("tag"), "Subject")
        who = _one_sentence(entry.get("who"), 80)
        looks = _one_sentence(entry.get("looks"))
        if number is None or number in seen or not (who or looks):
            continue
        picture = _tag_number(entry.get("from"), "Picture")
        if picture is not None and picture not in pictures:
            problems.append("<Subject %d> was said to come from <Picture %d>, "
                            "which the cast does not have" % (number, picture))
            picture = None
        seen.add(number)
        rows.append({"number": number, "who": who, "looks": looks,
                     "picture": picture})
    rows.sort(key=lambda r: r["number"])

    beats = [None] * count
    raw = [b for b in (obj.get("segments") or []) if isinstance(b, dict)]
    if len(raw) != count:
        problems.append("the plan has %d segment(s), %d were asked for"
                        % (len(raw), count))
    by_index = {}
    for position, beat in enumerate(raw):
        try:
            index = int(beat.get("index") or position + 1) - 1
        except (TypeError, ValueError):
            index = position
        if not 0 <= index < count or index in by_index:
            index = position
        if 0 <= index < count and index not in by_index:
            by_index[index] = beat
    defined = {r["number"] for r in rows}
    for index, beat in by_index.items():
        action = re.sub(r"\s+", " ", str(beat.get("action") or "")).strip()
        if not action:
            continue
        featured = []
        for tag in beat.get("featured") or []:
            number = _tag_number(tag, "Subject")
            if number in defined and number not in featured:
                featured.append(number)
        state = str(beat.get("vocals") or "").strip().lower()
        beats[index] = {"action": action[:900], "featured": featured,
                        "vocals": state if state in (
                            "instrumental", "sung", "mixed") else ""}
    missing = [i + 1 for i, b in enumerate(beats) if b is None]
    if missing and len(raw) == count:
        problems.append("segment(s) %s came back without an action"
                        % ", ".join(map(str, missing)))

    performance = str(obj.get("performance") or "").strip().lower()
    performance = performance if performance in ("sings", "raps",
                                                 "speaks") else ""
    performer = _tag_number(obj.get("performer"), "Subject")
    if performer not in defined:
        # The idea's lead singer, when the model named the role but not the tag.
        performer = next((r["number"] for r in rows if re.search(
            r"\b(?:lead|singer|vocal\w*|rapper|frontman|frontwoman|mc)\b",
            r["who"], re.IGNORECASE)), None) if performance else None
    return ({"cast": rows, "beats": beats, "performance": performance,
             "performer": performer,
             "vocal_sections": obj.get("vocal_sections") or []}, problems)


def canon_from_plan(rows):
    """The plan's cast as subject_definitions, one line per subject."""
    lines = []
    for r in rows:
        head = "<Subject %d> is %s" % (r["number"], r["who"] or "a subject")
        if r["picture"]:
            head += ", from <Picture %d>" % r["picture"]
        lines.append(head + (": %s." % r["looks"] if r["looks"] else "."))
    return "\n".join(lines)


def reference_images(cast):
    """(images, shown, unseen) — the cast pictures a vision call can look at.

    The Segment Prompter used to send none, so every look it wrote was
    invented: "short dark hair, a black leather jacket" for a singer whose
    picture shows long hair and a scar. H3 then blends the invented description
    with the real picture.
    """
    from .nodes_story import cast_reference_images   # nodes_story imports us
    pictures = [m for m in (cast or {}).get("members", [])
                if m.get("kind") in ("Subject", "Picture")]
    images = cast_reference_images(cast)
    with_file = [m for m in pictures if m.get("file")]
    shown = [m["tag"] for m in with_file[:len(images)]]
    unseen = [m["tag"] for m in pictures if m["tag"] not in shown]
    return images, shown, unseen


# --------------------------------------------------------------------------
# every cited subject defined, every cited reference retained
# --------------------------------------------------------------------------

def defining_lines(text):
    """{subject number: the definitions line that defines it}."""
    out = {}
    for line in subject_lines(text):
        first = next(iter(TAG_RE.findall(line)), None)
        if first and first[0].lower() == "subject":
            out.setdefault(int(first[1]), line)
    return out


def cited_numbers(prompt, kind, skip=("subject_definitions",)):
    found = set()
    for key, text in (prompt or {}).items():
        if key in skip:
            continue
        for k, n in TAG_RE.findall(text or ""):
            if k.lower() == kind.lower():
                found.add(int(n))
    return found


def fill_definitions(prompt, canon):
    """Add the shared definition of every subject cited but not defined.

    Only segment 1's definitions used to reach every segment. The rest cited
    <Subject 2>, <Subject 3> and <Subject 4> with nothing defining them, and
    the tag pass then deleted each sentence naming them: six summaries of an
    11-segment video came out "N/A".
    """
    have = set(defining_lines(prompt.get("subject_definitions")))
    missing = cited_numbers(prompt, "Subject") - have
    lines = defining_lines(canon)
    added = sorted(n for n in missing if n in lines)
    if added:
        current = (prompt.get("subject_definitions") or "").strip()
        current = "" if current == "N/A" else current
        prompt["subject_definitions"] = "\n".join(
            filter(None, [current] + [lines[n] for n in added]))
    return prompt, added


_SUBJECT_TAG = r"<\s*Subject\s*%d\s*>"


def name_undefined_subjects(prompt, allowed):
    """Keep sentences that cite a subject nothing defines, minus the tag.

    The shared tag pass deletes the whole sentence, which is right for an
    invented <Video 1> and wrong for "The band <Subject 1>, <Subject 2> and
    <Subject 3> perform": that was the entire summary. Here the tag becomes a
    plain word, and a retention line about a subject with no asset is dropped.
    Returns (prompt, renamed tags).
    """
    allowed = with_defined_subjects(prompt, allowed)
    wanted = allowed.get("Subject", set())
    renamed, out = set(), {}
    for key, text in prompt.items():
        value = text or ""
        if key == "subject_definitions":
            out[key] = value
            continue
        numbers = {int(n) for k, n in TAG_RE.findall(value)
                   if k.lower() == "subject" and int(n) not in wanted}
        for number in sorted(numbers):
            renamed.add("<Subject %d>" % number)
            tag = _SUBJECT_TAG % number
            if key == "retention_analysis":
                parts = re.split(r"(?:\r?\n|(?<=[.!?])\s+)", value)
                value = " ".join(p for p in parts
                                 if not re.search(tag, p, re.IGNORECASE))
                continue
            value = re.sub(tag + r"\s*\(S\d+\)", "<Subject %d>" % number,
                           value, flags=re.IGNORECASE)
            value = re.sub(tag + r"'s\b", "someone's", value, flags=re.IGNORECASE)
            value = re.sub(tag, "someone", value, flags=re.IGNORECASE)
        out[key] = value
    return out, sorted(renamed)


def _shots_citing(body, tag_re):
    """"[Shot 1], [Shot 3]" for the shots of ``body`` that cite a tag."""
    shots = []
    for match in re.finditer(r"\[\s*Shot\s+(\d+)\s*\](.*?)(?=\[\s*Shot\s+\d+\s*\]|$)",
                             body or "", re.IGNORECASE | re.DOTALL):
        if re.search(tag_re, match.group(2), re.IGNORECASE):
            shots.append("[Shot %s]" % match.group(1))
    return shots


def complete_retention(prompt, jobs=None):
    """A retention line for every subject and picture the prompt cites.

    Returns (prompt, added tags). Lines the model wrote are left as they are.
    """
    jobs = jobs or {}
    retention = re.sub(r"([.;])(?=<)", r"\1 ",
                       prompt.get("retention_analysis", "") or "")
    retention = "" if retention.strip() == "N/A" else retention.strip()
    body = prompt.get("detailed_description", "")
    new, added = [], []
    for kind in ("Subject", "Picture"):
        numbers = cited_numbers(prompt, kind,
                                skip=("subject_definitions", "retention_analysis"))
        for number in sorted(numbers):
            tag_re = r"<\s*%s\s*%d\s*>" % (kind, number)
            if re.search(tag_re, retention, re.IGNORECASE):
                continue
            shots = _shots_citing(body, tag_re)
            # Only claim the shots the description really puts it in.
            where = (" (appears in %s)" % ", ".join(shots)) if shots else ""
            if kind == "Subject":
                line = ("<Subject %d>%s: fully_preserved - identity, face, "
                        "hair, build and clothing exactly as defined."
                        % (number, where))
            elif jobs.get(number) == "identity":
                line = ("<Picture %d>: reference - the appearance of the "
                        "subjects taken from it only; not a frame, location "
                        "or composition." % number)
            elif jobs.get(number) == "style":
                line = ("<Picture %d>: reference - look, grade and texture "
                        "only." % number)
            else:
                line = ("<Picture %d>%s: fully_preserved - the place, set "
                        "and composition the shots are staged in."
                        % (number, where.replace("appears in", "staging")))
            new.append(line)
            added.append("<%s %d>" % (kind, number))
    if new:
        prompt["retention_analysis"] = " ".join(filter(None, new + [retention]))
    return prompt, added


def fallback_summary(prompt, cast, audio_role, action=""):
    """A summary written in code, when two model attempts left it empty."""
    from .nodes_story import task_prefix             # nodes_story imports us
    source = action or prompt.get("detailed_description", "")
    text = re.sub(r"\[\s*Shot\s+\d+\s*\]\s*(?:At\s*\d{1,2}:\d{2}(?:\.\d+)?\s*,?)?",
                  " ", source, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    first = re.split(r"(?<=[.!?])\s+", text)[0] if text else ""
    first = first[:300].rstrip(" ,;")
    return ("%s The target video %s" % (
        task_prefix(cast, audio_role),
        ("shows this: " + first.rstrip(".") + ".") if first
        else "is one segment of a longer piece."))


# --------------------------------------------------------------------------
# idea mode: the song's structure, the shot count, and each picture's job
# --------------------------------------------------------------------------

# The idea has to talk about the audio's structure before anything it says is
# turned into a vocal timeline: a model asked "where are the vocals" of an idea
# that never says will answer anyway, and the singer then mimes over nothing.
_STRUCTURE_RE = re.compile(
    r"\b(instrumental|vocals?|lyrics?|sing\w*|sung|rap\w*|verse|chorus|hook|"
    r"intro|outro|lip[- ]?sync\w*|a ?cappella)\b", re.IGNORECASE)


def idea_vocal_timeline(plan, idea, total, audio_role, segments):
    """The plan's word on where the song is sung. Returns (sections, note).

    ``sections`` is in the treatment's vocal-timeline shape, so the existing
    per-clip enforcement (closed mouths, instrumental guidance) runs on it. It
    is [] when the idea says nothing about the audio's structure, or when
    nothing in it is instrumental: that is audio_role alone, as before.
    """
    if not plan or not _STRUCTURE_RE.search(idea or "") \
            or not re.search(r"\d", idea or ""):
        return [], ""
    total = float(total)
    spans = []
    for row in plan.get("vocal_sections") or []:
        if not isinstance(row, dict):
            continue
        try:
            start = max(0.0, min(total, float(row.get("start"))))
            end = max(0.0, min(total, float(row.get("end"))))
        except (TypeError, ValueError):
            continue
        kind = str(row.get("kind") or "").strip().lower()
        kind = (vocals.INSTRUMENTAL if kind.startswith("instrument")
                else vocals.VOCALS if kind in ("vocals", "vocal", "sung",
                                               "singing", "lyrics")
                else "")
        if kind and end - start > 0.05:
            spans.append((start, end, kind))
    origin = "the idea's own times"
    if not spans:
        # Fall back to the per-segment states the plan gave: coarser, but it
        # is still the idea's statement, placed on the real cuts.
        for seg, beat in zip(segments, plan.get("beats") or []):
            state = (beat or {}).get("vocals")
            if state not in ("instrumental", "sung"):
                continue
            start = float(seg.get("audio_start") or 0.0)
            spans.append((start, start + seg["target_duration"],
                          vocals.INSTRUMENTAL if state == "instrumental"
                          else vocals.VOCALS))
        origin = "the plan's segment states"
    if not any(k == vocals.INSTRUMENTAL for _, _, k in spans):
        return [], ""

    # Anything the idea did not place is the song's default for this role:
    # "performed on camera" means sung unless the idea says otherwise.
    gap_kind = (vocals.VOCALS if audio_role == "performed on camera"
                else vocals.INSTRUMENTAL)
    spans.sort()
    out, clock = [], 0.0
    for start, end, kind in spans:
        # Segment starts are rounded to the millisecond; a 1 ms gap between
        # two of them is not a gap in the song.
        start = clock if start - clock <= 0.05 else start
        end = min(end, total)
        if end <= start:
            continue
        if start - clock > 0.05:
            out.append([clock, start, gap_kind])
        out.append([start, end, kind])
        clock = end
    if total - clock > 0.05:
        out.append([clock, total, gap_kind])
    merged = []
    for start, end, kind in out:
        if merged and merged[-1][2] == kind and abs(merged[-1][1] - start) < 0.06:
            merged[-1][1] = end
        else:
            merged.append([start, end, kind])
    sections = [{"start": round(a, 3), "end": round(b, 3), "kind": k}
                for a, b, k in merged]
    note = "from your idea (%s): %s" % (origin, ", ".join(
        "%s-%s %s" % (vocals.timestamp(s["start"]), vocals.timestamp(s["end"]),
                      s["kind"]) for s in sections))
    return sections, note


MIN_SHOT_SECONDS = 1.5


def shot_budget(shots_per_segment, target_seconds):
    """How many shots an idea-mode segment may have.

    shots_per_segment was ignored without a treatment, and one 7.8-second clip
    came back with 25 shots, 0.32 seconds apart.
    """
    by_length = max(1, int(float(target_seconds) // MIN_SHOT_SECONDS))
    return max(1, min(int(shots_per_segment or 1), by_length))


_SHOT_HEAD = re.compile(
    r"\[\s*Shot\s+(\d+)\s*\]\s*(?:At\s*\d{1,2}:\d{2}(?:\.\d+)?\s*,?\s*)?",
    re.IGNORECASE)


def fold_shots(body, most):
    """Keep the first ``most`` shot markers; later shots continue the last one.

    Returns (body, folded count). The text of a folded shot is kept, so no
    action is lost, only the cut.
    """
    heads = list(_SHOT_HEAD.finditer(body or ""))
    if len(heads) <= most:
        return body, 0
    out, last = [], 0
    for n, head in enumerate(heads):
        out.append(body[last:head.start()])
        out.append(head.group(0) if n < most else "then ")
        last = head.end()
    out.append(body[last:])
    text = re.sub(r"\.\s+then ([a-z])", lambda m: ". Then " + m.group(1),
                  "".join(out))
    return re.sub(r"[ \t]{2,}", " ", text), len(heads) - most


def prune_shot_refs(text, count):
    """Drop "[Shot N]" references past the last shot a description has."""
    def keep(match):
        return match.group(0) if int(match.group(1)) <= count else ""
    text = re.sub(r"\[\s*Shot\s+(\d+)\s*\]", keep, text or "")
    text = re.sub(r"(,\s*)+(?=\))", "", text)
    text = re.sub(r"\(\s*(?:appears in\s*)?,?\s*\)", "", text)
    text = re.sub(r"\(\s*,\s*", "(", text)
    text = re.sub(r",\s*,", ",", text)
    return re.sub(r"[ \t]{2,}", " ", text)


_FRAME_WORDS = re.compile(
    r"\b(opens?|opening|begins?|starts?|anchor\w*|composition|key ?frames?|"
    r"first frame|last frame|frame|stage|set|setting|location|backdrop|"
    r"layout|positioned|venue|room|arena)\b", re.IGNORECASE)
_PIC = r"<\s*Picture\s*%d\s*>"


_PLACEHOLDER_LINE = re.compile(
    r"(?:\s*,?\s*(?:turns\s+and\s+)?(?:says|sings|speaks|raps|delivers)\s*,?)?"
    r"\s*<d>\s*(?:\[[A-Za-z]+\]\s*)?['\"\u2018\u201c]?\s*"
    r"(?:\[[^\]]*\]|\.{2,}|\u2026|lyrics?|words?)?\s*['\"\u2019\u201d]?\s*</d>\s*,?",
    re.IGNORECASE)


def drop_placeholder_dialogue(prompt):
    """Remove spoken lines that hold no words, like <d>'[lyrics]'</d>.

    An idea gives no lyrics, so the writer filled the line with a placeholder,
    and H3 would voice "lyrics". The singing itself is kept: it is described
    in the prose, and the words come from the supplied audio.
    Returns (prompt, removed count).
    """
    removed = 0
    for key in ("summary", "detailed_description"):
        text = prompt.get(key) or ""
        text, hits = _PLACEHOLDER_LINE.subn(" sings,", text)
        removed += hits
        text = re.sub(r"\bsings,\s*,", "sings,", text)
        prompt[key] = re.sub(r"[ \t]{2,}", " ", text)
    if removed:
        # A speaker id with no line left in its sentence names a voice that
        # never speaks, and the report then lists a speaker who has no line.
        for key in ("summary", "detailed_description"):
            parts = re.split(r"(?<=[.!?])\s+", prompt.get(key) or "")
            prompt[key] = " ".join(
                p if "<d>" in p else re.sub(r"\s*\(S\d+(?:,\s*S\d+)*\)", "", p)
                for p in parts)
    return prompt, removed


_SUMMARY_TYPE = re.compile(r"^\s*\[([^\]]*)\]\s*")


def fix_summary_type(summary, cast, audio_role):
    """The summary opens with the task type, and names no asset it lacks.

    Seen: "[video continuation]" with no video in the cast, and a summary that
    opened with "[Shot 1]".
    """
    from .nodes_story import task_prefix             # nodes_story imports us
    text = re.sub(r"\[\s*Shot\s+\d+\s*\]\s*(?:At\s*\d{1,2}:\d{2}(?:\.\d+)?\s*,?\s*)?",
                  "", summary or "", flags=re.IGNORECASE).strip()
    if not text or text == "N/A":
        return summary
    members = (cast or {}).get("members", [])
    has_video = any(m.get("kind") == "Video" for m in members)
    # "[keyframe completion]" came back for a cast with no frame picture.
    has_keyframe = any(m.get("role") in ("first_frame", "last_frame", "keyframe")
                       for m in members)
    head = _SUMMARY_TYPE.match(text)
    if head:
        kinds = head.group(1).lower()
        known = re.search(r"generation|reuse|reference|completion", kinds)
        if (known and (has_video or "video" not in kinds)
                and (has_keyframe or "completion" not in kinds)):
            return text
    body = text[head.end():] if head else text
    return "%s %s" % (task_prefix(cast, audio_role), body)


def only_performer_sings(prompt, performer):
    """Take singing away from anyone the plan did not name as the performer.

    Seen: "<Subject 5> (S1) plays the keyboard, mouth matching the words as he
    sings." in a video whose only singer is <Subject 1>. A sentence that names
    the performer is left alone; in one that does not, the singing clauses go,
    and a sentence that was nothing but singing becomes a closed-mouth line.
    Returns (prompt, changed count).
    """
    if not performer:
        return prompt, 0
    singer = re.compile(_SUBJECT_TAG % performer, re.IGNORECASE)
    changed = 0
    for key in ("summary", "detailed_description"):
        parts = re.split(r"(?<=[.!?])\s+", prompt.get(key) or "")
        for i, part in enumerate(parts):
            others = [int(n) for k, n in TAG_RE.findall(part)
                      if k.lower() == "subject" and int(n) != performer]
            if (singer.search(part) or not others
                    or not vocals.SINGING_RE.search(part)
                    and not re.search(r"mouth\w*\s+(?:match|moving|moves)\w*",
                                      part, re.IGNORECASE)):
                continue
            head = _SHOT_HEAD.match(part)
            lead = head.group(0) if head else ""
            clauses = re.split(r",\s*|;\s*|\s+as\s+|\s+while\s+",
                              part[len(lead):].rstrip(" ."))
            kept = [c for c in clauses
                    if not vocals.SINGING_RE.search(c)
                    and not re.search(r"\bmouth", c, re.IGNORECASE)]
            if kept and TAG_RE.search(kept[0]):
                text = ", ".join(kept)
            else:
                text = "<Subject %d> keeps playing, mouth closed" % others[0]
            parts[i] = lead + re.sub(r"\s*\(S\d+(?:,\s*S\d+)*\)", "", text) + "."
            changed += 1
        prompt[key] = " ".join(parts)
    return prompt, changed


def enforce_picture_jobs(prompt, jobs, performance="", idea=""):
    """Make the prompt use each picture only for the job it was given.

    Returns (prompt, changes). An IDENTITY picture (a character sheet) was the
    opening frame of seven segments of one music video. Here, a character
    picture used as a place becomes the one frame picture, and any other
    mention of it in the description is dropped: the subject definitions
    already carry who it shows. The verb the plan chose for the performer is
    kept too: a metal singer was "rapping" in three segments.
    """
    changes = []
    identity = sorted(n for n, j in jobs.items() if j == "identity")
    frames = sorted(n for n, j in jobs.items() if j == "frame")
    frame_tag = "<Picture %d>" % frames[0] if len(frames) == 1 else ""
    for key in ("summary", "detailed_description"):
        text = prompt.get(key) or ""
        for n in identity:
            pic = _PIC % n
            if not re.search(pic, text, re.IGNORECASE):
                continue
            # "<Subject 1> (from <Picture 2>)" is identity, said correctly.
            text = re.sub(
                r"(<\s*Subject\s*\d+\s*>)\s*,?\s*\(?\s*(?:as\s+(?:seen|shown)\s+"
                r"in|from|in|of)\s+" + pic + r"(?:\s*\))?", r"\1", text,
                flags=re.IGNORECASE)
            parts = re.split(r"(?<=[.!?])\s+", text)
            for i, part in enumerate(parts):
                if not re.search(pic, part, re.IGNORECASE):
                    continue
                if frame_tag and _FRAME_WORDS.search(part):
                    part = re.sub(pic, frame_tag, part, flags=re.IGNORECASE)
                else:
                    part = re.sub(
                        r"\s*,?\s*\(?\s*(?:(?:as\s+(?:seen|shown)\s+in|based\s+on|"
                        r"anchored\s+by|according\s+to|from|in|of|on|per)\s+)?"
                        + pic + r"(?:'s)?(?:\s*\))?", "", part,
                        flags=re.IGNORECASE)
                parts[i] = part
            text = re.sub(r"[ \t]{2,}", " ", " ".join(parts))
            text = re.sub(r"\s+([,.;:])", r"\1", text)
            changes.append("<Picture %d> in %s" % (n, key))
        prompt[key] = text or "N/A"

    # The model glues lines together ("identity.<Picture 2> ..."), which
    # hid the sheet's line from the pass below.
    retention = re.sub(r"([.;])(?=<)", r"\1 ",
                       prompt.get("retention_analysis") or "")
    lines = re.split(r"(?:\r?\n|(?<=[.!?;])\s+)", retention)
    kept = []
    for line in lines:
        first = next(iter(TAG_RE.findall(line)), None)
        if (first and first[0].lower() == "picture"
                and int(first[1]) in identity):
            n = int(first[1])
            line = ("<Picture %d>: reference - the appearance of the subjects "
                    "taken from it only; not a frame, location or "
                    "composition." % n)
            if line in kept:
                continue
        kept.append(line)
    prompt["retention_analysis"] = " ".join(p.strip() for p in kept if p.strip())

    body = prompt.get("detailed_description") or ""
    if frame_tag and not re.search(re.escape(frame_tag), body):
        head = _SHOT_HEAD.search(body)
        insert = "Staged in the setting of %s. " % frame_tag
        body = (body[:head.end()] + insert + body[head.end():] if head
                else insert + body)
        prompt["detailed_description"] = body
        changes.append("%s added as the setting" % frame_tag)

    if performance == "sings" and not re.search(r"\brap", idea or "",
                                                re.IGNORECASE):
        swaps = (("rapping", "singing"), ("rapped", "sang"), ("raps", "sings"),
                 ("rap", "sing"))
        for key in ("summary", "detailed_description", "overall_soundscape"):
            text = prompt.get(key) or ""
            for old, new in swaps:
                text, hits = re.subn(r"\b%s\b" % old, new, text,
                                     flags=re.IGNORECASE)
                if hits:
                    changes.append("'%s' -> '%s'" % (old, new))
            prompt[key] = text
    return prompt, changes


class H3PlannerSegmentPrompter:
    """Long prompt in, a fully prompted timeline out.

    Connect the `h3_prompt` output of H3 Full-Reference Video Prompt Creator,
    run at the WHOLE video's duration. This node cuts it on its own [Shot N]
    markers — never through the middle of a shot — then writes a complete
    six-section H3 prompt for each piece.

    Leave h3_prompt empty and fill `idea` instead to plan even beats from
    scratch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "h3_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "the h3_prompt output of your H3 prompt creator, run at the FULL video duration. Leave empty to plan from `idea` instead."}),
                "total_seconds": ("FLOAT", {
                    "default": 60.0, "min": 1.0, "max": 600.0, "step": 0.5,
                    "tooltip": "the whole video's length — the same number you gave the prompt creator"}),
                "segment_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5,
                    "tooltip": "rough length per segment; only used when there is no treatment to cut on"}),
                "shots_per_segment": ("INT", {
                    "default": 1, "min": 1, "max": 12,
                    "tooltip": "1 = one segment per shot in the treatment, and one prompt written for it — the most detail per clip. Raise it to pack several shots into one longer clip while they still fit under the segment cap. With only an idea (no treatment) it is the most shots each clip may have, and each shot is at least 1.5s: 1 = one continuous shot per clip."}),
                "format": (FORMATS, {
                    "default": "auto",
                    "tooltip": "what kind of video this is. Steers emphasis per segment; auto adds nothing."}),
                "audio_role": (list(AUDIO_ROLES), {
                    "default": "performed on camera",
                    "tooltip": "who makes the sound in the connected audio. Set 'performed on camera' when the subject raps, sings or speaks it and their mouth must match the words - without that H3 plays the track over someone sitting silently. Ignored when no audio asset is in the cast."}),
                "provider": (engine.providers(), {"default": "Ollama (Local)"}),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "ollama_model": ("STRING", {"default": engine.default_ollama_model()}),
                "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.2,
                                          "step": 0.05}),
                "reuse_existing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "skip segments already written from the same inputs"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF,
                                 "tooltip": "change to rewrite every segment"}),
            },
            "optional": {
                "cast": ("H3_CAST",),
                "idea": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "extra direction applied to every segment; the whole plan when h3_prompt is empty"}),
                "style_prefix_override": ("STRING", {
                    "default": "",
                    "tooltip": "blank = lift it off [Shot 1] of the treatment"}),
                "api_key": ("STRING", {
                    "default": "",
                    "tooltip": "prefer the environment variable — a key typed here is saved into the workflow"}),
                "api_model": ("STRING", {"default": ""}),
                "max_output_tokens": ("INT", {"default": 3072, "min": 256, "max": 32768}),
                "num_ctx": ("INT", {"default": 8192, "min": 2048, "max": 131072}),
                "keep_alive": ("STRING", {"default": "10m"}),
                "request_timeout": ("INT", {"default": 600, "min": 30, "max": 3600}),
            },
        }

    RETURN_TYPES = ("H3_TIMELINE", "STRING")
    RETURN_NAMES = ("timeline", "report")
    FUNCTION = "write"
    CATEGORY = CATEGORY

    def write(self, project, h3_prompt, total_seconds, segment_seconds,
              shots_per_segment, format, audio_role, provider, ollama_url,
              ollama_model,
              temperature, reuse_existing, seed, cast=None, idea="",
              style_prefix_override="", api_key="", api_model="",
              max_output_tokens=3072, num_ctx=8192, keep_alive="10m",
              request_timeout=600):

        # ---- 1. decide where the cuts go (arithmetic, no model) ----------
        ceiling_frames, ceiling_seconds = ladder.ceiling(
            project["fps"], project["max_seconds"],
            modulus=project["frame_modulus"],
            remainder=project["frame_remainder"],
            minimum=project["frame_minimum"])

        treatment = str(h3_prompt).strip()
        warnings = []
        if treatment:
            # segment_seconds is the longest a segment may be, never more than
            # the ladder allows. It used to be ignored here, and the only limit
            # was the ladder ceiling, which a stretched last shot ignored too:
            # an 86-second video came out as five segments and one of 59s.
            cap = min(float(segment_seconds), ceiling_seconds)
            segments, context, warnings = splitter.split_treatment(
                treatment, total_seconds, cap, 2.0,
                shots_per_segment, enforce_runtime=True)
            if not segments:
                raise ValueError(
                    "no [Shot N] markers in h3_prompt, so there is nothing to "
                    "cut on. Run your prompt creator at the full video duration "
                    "and connect its h3_prompt output — or clear h3_prompt and "
                    "fill `idea` to plan even beats instead.")
            source = "treatment"
        elif idea.strip():
            segments = _even_segments(total_seconds, segment_seconds,
                                      ceiling_seconds)
            context = {"style_prefix": "", "total_duration": float(total_seconds),
                       "shot_count": 0}
            source = "idea"
        else:
            raise ValueError(
                "nothing to plan from — connect h3_prompt from your prompt "
                "creator, or type an idea.")

        if style_prefix_override.strip():
            context["style_prefix"] = style_prefix_override.strip()
            warnings = [w for w in warnings if "style prefix" not in w]
        prefix = context.get("style_prefix", "")

        timeline = store.normalize_timeline(
            {"name": project["name"], "context": context, "segments": segments},
            project)
        # normalize_timeline copies the context. Everything below writes to
        # it (the plan's cast, the backend for refine), and writing to the
        # stale copy lost it: the saved timeline had no backend to refine with.
        context = timeline["context"]
        for seg in timeline["segments"]:
            frames = ladder.snap_frames(
                seg["target_duration"], project["fps"],
                modulus=project["frame_modulus"],
                remainder=project["frame_remainder"],
                minimum=project["frame_minimum"], direction=project["snap"])
            seg["render_frames"] = frames
            seg["render_duration"] = frames / float(project["fps"])

        self._adopt_persisted(project, timeline)

        # ---- 2. write a standalone prompt for each ------------------------
        if not engine.available():
            raise RuntimeError(engine.MISSING)

        backend_kwargs = dict(
            provider=provider, ollama_url=ollama_url,
            ollama_model=ollama_model, api_model=api_model,
            temperature=temperature, keep_alive=keep_alive,
            timeout=request_timeout,
            max_output_tokens=max_output_tokens, num_ctx=num_ctx)
        cfg = engine.backend_config(api_key=api_key,
                                    **backend_kwargs)
        # So a single card can be refined later without re-queueing the
        # graph. The API key is deliberately never stored.
        store.remember_backend(context, backend_kwargs)
        schema = engine.json_schema(
            {k: {"type": "string"} for k in SECTIONS}, list(SECTIONS))

        system = engine.full_ref_system() + "\n\n" + SEGMENT_SYSTEM
        guidance = FORMAT_GUIDANCE.get(format, "").strip()
        if guidance:
            system += "\n\n" + guidance

        allowed = _allowed_tags(cast)
        cast_block = _cast_block(cast)
        names = asset_names(cast)
        audio_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                          if m.get("kind") == "Audio"), "")
        subject_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                            if m.get("kind") == "Subject"), "")
        jobs = picture_jobs(cast)
        jobs_block = picture_jobs_block(cast)
        plan, plan_lines, vocal_note = None, [], ""
        images, shown, unseen = [], [], []
        if source == "idea":
            # The writer sees the pictures, and works from one plan for the
            # whole video: who is who, and what happens in each segment.
            images, shown, unseen = reference_images(cast)
            plan, plan_lines = self._plan(
                project, timeline, idea, cast, cast_block, jobs_block, images,
                cfg, format, audio_role, seed, reuse_existing,
                max_output_tokens, backend_kwargs, api_key)
            if plan and plan["cast"]:
                context["subject_definitions"] = canon_from_plan(plan["cast"])
            if plan and audio_tag:
                sections, vocal_note = idea_vocal_timeline(
                    plan, idea, sum(s["target_duration"]
                                    for s in timeline["segments"]),
                    audio_role, timeline["segments"])
                if sections:
                    context["vocal_timeline"] = sections
            if jobs_block:
                system += "\n\nPICTURE JOBS — what each cast picture is for:\n" \
                          + jobs_block
        if plan and plan.get("performer"):
            # The soundscape line names who performs the track. Without it,
            # it said "the subject on screen", and the guitarist sang.
            subject_tag = "<Subject %d>" % plan["performer"]
        timeline_sections = context.get("vocal_timeline") or []
        base_system = system

        speakers, rows, failed = {}, [], []
        written = skipped = 0
        bodies = []          # every description written so far, to catch repeats
        # ONE wording for the cast, reused verbatim in every segment. Seeded
        # from the treatment when it had definitions, otherwise from whichever
        # segment is written first.
        canon = engine.clean(context.get("subject_definitions") or "")
        canon_from = ("the plan" if plan and plan["cast"]
                      else "the treatment") if canon else ""
        repairs = []
        started = time.time()

        vocal_counts = {}
        for i, seg in enumerate(timeline["segments"]):
            clip_vocals = vocals.for_segment(context, seg) if audio_tag else None
            if clip_vocals:
                vocal_counts[clip_vocals["state"]] = (
                    vocal_counts.get(clip_vocals["state"], 0) + 1)
            # The system prompt is per clip: an instrumental clip must not be
            # told the vocal is performed on camera.
            system = base_system
            audio_block = audio_system_block(audio_tag, audio_role, clip_vocals)
            if audio_block:
                system += "\n\n" + audio_block
            # The vocal state is part of the fingerprint, or a prompt written
            # before the timeline existed is reused as "unchanged" and the
            # singer keeps singing through the intro.
            fingerprint = _hash(seg.get("source"), seg["target_duration"],
                                seg.get("render_duration"), prefix, cast_block,
                                idea, source, format, seed, audio_role,
                                clip_vocals,
                                *self._plan_fingerprint(source, plan, i))

            if seg.get("state") == store.LOCKED:
                rows.append("  %-8s locked, left alone" % seg["id"])
                skipped += 1
                continue
            if (reuse_existing and seg.get("prompt")
                    and seg.get("prompt_fingerprint") == fingerprint):
                rows.append("  %-8s %6.3fs  unchanged, reused"
                            % (seg["id"], seg["target_duration"]))
                skipped += 1
                continue

            beat = plan["beats"][i] if plan else None
            budget = (shot_budget(shots_per_segment, seg["target_duration"])
                      if source == "idea" else 0)
            opts = dict(canon=canon, jobs=jobs, idea_mode=source == "idea",
                        cast=cast,
                        performance=(plan or {}).get("performance", ""),
                        performer=(plan or {}).get("performer"),
                        idea=idea, max_shots=budget)
            self._last_changes = []
            prompt, obj, repeat, invented = None, None, "", []
            retry_reason = ""
            for attempt in (1, 2):
                user = self._user_message(
                    timeline, seg, i, prefix, cast_block, idea, source,
                    speakers, previous_body=bodies[-1] if bodies else "",
                    audio_tag=audio_tag, clip_vocals=clip_vocals,
                    plan=plan, max_shots=budget)
                if attempt == 2:
                    user += retry_reason
                try:
                    obj, note = engine.generate(
                        cfg, system, user, schema,
                        required_keys=("summary", "retention_analysis",
                                       "detailed_description"),
                        min_words=120, images=images)
                except Exception as ex:
                    failed.append("%s: %s" % (seg["id"], ex))
                    rows.append("  %-8s FAILED — %s" % (seg["id"], str(ex)[:70]))
                    obj = None
                    break
                if not obj:
                    failed.append("%s: the provider returned nothing usable"
                                  % seg["id"])
                    rows.append("  %-8s FAILED — empty reply" % seg["id"])
                    break

                prompt, invented = self._post_process(
                    obj, seg, prefix, allowed, names, audio_tag,
                    audio_role, subject_tag, clip_vocals, **opts)
                twin, score = self._closest(prompt["detailed_description"], bodies)
                if invented and attempt == 1:
                    # ask again before falling back to stripping
                    retry_reason = (
                        "\n\nYOUR PREVIOUS ATTEMPT CITED %s, WHICH NOTHING "
                        "DEFINES. Cite only the tags in the cast list, and "
                        "every <Subject N> you cite must be one listed under "
                        "the cast's subjects, with that same number."
                        % ", ".join(invented))
                    continue
                if twin is None:
                    repeat = ""
                    break
                repeat = "  (%d%% like segment %d)" % (round(score * 100), twin + 1)
                retry_reason = ("\n\nYOUR PREVIOUS ATTEMPT REPEATED AN EARLIER "
                                "SEGMENT ALMOST WORD FOR WORD. Write this segment "
                                "again from scratch: different action, different "
                                "framing, different camera move. Describe only "
                                "what the brief above says happens here.")
                if attempt == 2:
                    failed.append(
                        "%s still reads %d%% the same as segment %d — the model "
                        "is not differentiating; try a treatment with more "
                        "distinct shots, or raise temperature"
                        % (seg["id"], round(score * 100), twin + 1))

            if obj is None or prompt is None:
                continue

            # Every segment carries a summary, a retention line per reference
            # and a description. The model left the summary out, or the tag
            # pass emptied it, in six of eleven segments of one music video.
            empty = self._empty_sections(prompt)
            if empty:
                prompt, still = self._refill_sections(
                    cfg, system, user, empty, obj, seg, prefix, allowed, names,
                    audio_tag, audio_role, subject_tag, clip_vocals, images,
                    opts)
                if "summary" in still:
                    prompt["summary"] = fallback_summary(
                        prompt, cast, audio_role, (beat or {}).get("action", ""))
                    still.remove("summary")
                repairs.append("%s: %s empty, asked again%s" % (
                    seg["id"], ", ".join(empty),
                    "" if not still else " — STILL EMPTY: " + ", ".join(still)))
                if still:
                    failed.append("%s: %s still empty after a second request"
                                  % (seg["id"], ", ".join(still)))

            if not canon:
                canon = prompt["subject_definitions"]
                canon_from = seg["id"]
            else:
                own = defining_lines(prompt["subject_definitions"])
                shared = canonical_subjects(canon, cited_tags(prompt))
                # A subject the shared block does not know keeps this
                # segment's own definition, and that wording becomes the
                # shared one from here on. Replacing the block outright left
                # <Subject 2>, 3 and 4 cited in ten segments and defined in
                # none.
                known = set(defining_lines(canon))
                new = [n for n in sorted(cited_numbers(prompt, "Subject"))
                       if n not in known and n in own]
                if new:
                    canon = "\n".join([canon] + [own[n] for n in new])
                    shared = "\n".join(filter(None, [shared] +
                                              [own[n] for n in new]))
                    repairs.append("%s: added %s to the shared cast wording"
                                   % (seg["id"], ", ".join(
                                       "<Subject %d>" % n for n in new)))
                if shared:
                    prompt["subject_definitions"] = shared

            seg["prompt"] = prompt
            seg["prompt_fingerprint"] = fingerprint
            seg["spec_hash"] = store.spec_hash(seg)
            self._collect_speakers(prompt, speakers)
            bodies.append(prompt["detailed_description"])
            written += 1
            note = repeat
            if clip_vocals:
                note += "  [%s]" % vocals.describe(clip_vocals)
            if invented:
                note += "  (stripped %s — not in the cast)" % ", ".join(invented)
            if self._last_changes:
                repairs.append("%s: %s" % (seg["id"], "; ".join(
                    dict.fromkeys(self._last_changes))))
            rows.append("  %-8s %6.3fs  written, %d words%s"
                        % (seg["id"], seg["target_duration"],
                           len(flatten_prompt(prompt).split()), note))

        planned = sum(s["target_duration"] for s in timeline["segments"])
        rendered = sum(s["render_duration"] for s in timeline["segments"])
        report = "\n".join([
            "%d segment(s) from the %s — %d written, %d reused, %d failed in %.0fs"
            % (len(timeline["segments"]), source, written, skipped,
               len(failed), time.time() - started),
            "format        %s" % format,
            "audio         %s" % ("%s, %s" % (audio_tag, audio_role)
                                  if audio_tag else "(none connected)"),
            "vocals        %s" % (
                "%s — %d sung, %d instrumental, %d mixed clip(s)" % (
                    vocal_note, vocal_counts.get(vocals.VOCALS, 0),
                    vocal_counts.get(vocals.INSTRUMENTAL, 0),
                    vocal_counts.get(vocals.MIXED, 0))
                if vocal_note else
                _vocal_summary(audio_tag, timeline_sections, vocal_counts)),
            "soundscape    %s"
            % ("overall_soundscape and non_diegetic_music are REPLACED in "
               "every segment, and the %s line in retention_analysis set to "
               "fully_copy — a supplied track already contains everything "
               "that will be heard, so the treatment's own sound writing "
               "would tell H3 to synthesise more on top of it. Disconnect "
               "the audio asset to keep what the treatment wrote."
               % audio_tag
               if audio_tag else "the treatment's own, kept as written"),
            "style prefix  %s" % (prefix or "(none)"),
            "cast          %s" % (cast_block.replace("\n", " | ") if cast else "(none)"),
            "planned       %.3fs -> H3 renders %.3fs (stitcher trims %.3fs back)"
            % (planned, rendered, rendered - planned),
            "ceiling       %.3fs (%d frames) per segment"
            % (ceiling_seconds, ceiling_frames),
            "speakers      %s" % (", ".join(sorted(speakers)) or "(none)"),
            "cast wording  %s" % ("one shared block, from %s — every segment "
                                  "describes the cast identically" % canon_from
                                  if canon_from else "(none)"),
        ] + ([
            "pictures      %s" % (
                ("shown to the model: %s" % ", ".join(shown) if shown
                 else "NONE shown to the model — every look is written blind")
                + ("; not shown (wired in, no file): %s" % ", ".join(unseen)
                   if unseen else "")),
        ] if source == "idea" and (shown or unseen) else []) + plan_lines + [
            "",
            "segments:",
        ] + rows
            + (["", "REPAIRED", "  " + "\n  ".join(repairs)] if repairs else [])
            + (["", "WARNINGS", "! " + "\n! ".join(warnings)] if warnings else [])
            + (["", "FAILURES", "! " + "\n! ".join(failed)] if failed else []))

        return (timeline, report)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _adopt_persisted(project, timeline):
        """Carry prompts and locks over from the timeline already on disk.

        A timeline is rebuilt from scratch every run, so without this every
        segment would look unwritten and `reuse_existing` could never fire —
        one model call per segment on every queue.
        """
        saved = store.load(project["timeline_path"])
        if not saved:
            return
        known = {s["id"]: s for s in saved.get("segments", [])}
        for seg in timeline["segments"]:
            prev = known.get(seg["id"])
            if not prev:
                continue
            if prev.get("state") == store.LOCKED:
                seg["state"] = store.LOCKED
            if not seg.get("prompt") and prev.get("prompt"):
                seg["prompt"] = prev["prompt"]
                seg["prompt_fingerprint"] = prev.get("prompt_fingerprint", "")

    PLAN_VERSION = "idea-plan-1"

    def _plan(self, project, timeline, idea, cast, cast_block, jobs_block,
              images, cfg, format, audio_role, seed, reuse_existing,
              max_output_tokens, backend_kwargs, api_key):
        """One plan for the whole idea. Returns (plan or None, report lines).

        Saved in the timeline's context, and reused while the idea, cast and
        cut are unchanged and reuse_existing is on: a fresh plan on every
        queue would re-word every segment and discard every rendered clip.
        """
        segments = timeline["segments"]
        count = len(segments)
        total = sum(s["target_duration"] for s in segments)
        key = _hash(self.PLAN_VERSION, idea, cast_block, jobs_block,
                    [s["target_duration"] for s in segments], format,
                    audio_role, seed)
        context = timeline["context"]

        if reuse_existing:
            saved = (store.load(project["timeline_path"]) or {}).get("context")
            old = (saved or {}).get("plan")
            if (isinstance(old, dict) and old.get("key") == key
                    and isinstance(old.get("plan"), dict)
                    and len(old["plan"].get("beats") or []) == count):
                context["plan"] = old
                return old["plan"], self._plan_lines(
                    old["plan"], [], "reused from the saved timeline — the "
                    "idea, cast and cuts are unchanged")

        clock = []
        for n, s in enumerate(segments):
            start = float(s.get("audio_start") or 0.0)
            clock.append("  segment %d: %.2fs to %.2fs"
                         % (n + 1, start, start + s["target_duration"]))
        user = "\n".join([
            "THE VIDEO: %.2f seconds, %d segments, format: %s, audio: %s."
            % (total, count, format, audio_role),
            "",
            "THE SEGMENTS, on the whole video's clock:",
        ] + clock + [
            "",
            "CAST LIST — the only tags that exist:",
            cast_block,
            "",
            "PICTURE JOBS:",
            jobs_block or "(no pictures)",
            "",
            "THE USER'S IDEA — follow every instruction in it:",
            idea.strip(),
            "",
            "Return exactly %d segments, index 1 to %d." % (count, count),
        ])
        # The plan is the longest reply of the run: a cast and a beat per
        # segment. It gets at least 4096 tokens whatever the writer uses.
        plan_cfg = engine.backend_config(api_key=api_key, **dict(
            backend_kwargs,
            max_output_tokens=max(int(max_output_tokens), 4096)))
        needs_cast = any(m.get("kind") in ("Subject", "Picture")
                         for m in (cast or {}).get("members", []))

        best, best_problems, best_score = None, ["no plan came back"], -1
        request = user
        for attempt in (1, 2):
            try:
                obj, _note = engine.generate(plan_cfg, PLAN_SYSTEM, request,
                                             plan_schema(), images=images)
            except Exception as ex:
                best_problems = ["the plan request failed: %s" % str(ex)[:160]]
                break
            plan, problems = validate_plan(obj, count, cast, total)
            if needs_cast and not plan["cast"]:
                problems.append("the plan names no <Subject N> for the people "
                                "and objects in the pictures")
            score = (sum(1 for b in plan["beats"] if b) * 10
                     + len(plan["cast"]))
            if score > best_score:
                best, best_problems, best_score = plan, problems, score
            if not problems:
                break
            request = (user + "\n\nYOUR PREVIOUS PLAN WAS REJECTED: "
                       + "; ".join(problems) + ". Return exactly %d segments, "
                       "each with its index and an action, and one cast entry "
                       "for every person or object seen in the pictures."
                       % count)

        if best is None or best_score <= 0:
            return None, [
                "plan          FAILED — %s. Every segment was written on its "
                "own, without a shared cast or story; check the model and "
                "run again." % "; ".join(best_problems)]
        context["plan"] = {"key": key, "plan": best}
        return best, self._plan_lines(best, best_problems, "made for this run")

    @staticmethod
    def _plan_lines(plan, problems, how):
        beats = plan.get("beats") or []
        got = sum(1 for b in beats if b)
        lines = ["plan          one plan for the whole video, %s: %d of %d "
                 "segment(s) planned" % (how, got, len(beats))]
        for r in plan.get("cast") or []:
            lines.append("                <Subject %d> %s%s"
                         % (r["number"], r["who"],
                            " (from <Picture %d>)" % r["picture"]
                            if r["picture"] else ""))
        if plan.get("performance"):
            lines.append("                the performer %s%s" % (
                plan["performance"],
                " (<Subject %d>)" % plan["performer"]
                if plan.get("performer") else ""))
        missing = [str(i + 1) for i, b in enumerate(beats) if not b]
        if missing:
            lines.append("  ! segment(s) %s have no plan and were written from "
                         "the idea alone" % ", ".join(missing))
        for p in problems:
            lines.append("  ! plan: %s" % p)
        return lines

    def _plan_fingerprint(self, source, plan, index):
        """What the plan adds to a segment's fingerprint.

        Nothing for a treatment, so those fingerprints are unchanged. For an
        idea, the plan version is always in it: prompts written blind before
        the plan existed are rewritten once rather than reused.
        """
        if source != "idea":
            return ()
        beat = plan["beats"][index] if plan else None
        return (self.PLAN_VERSION, beat,
                (plan or {}).get("cast"), (plan or {}).get("performance"),
                (plan or {}).get("performer"))

    @staticmethod
    def _empty_sections(prompt):
        return [k for k in ("summary", "retention_analysis",
                            "detailed_description")
                if (prompt.get(k) or "").strip() in ("", "N/A")]

    def _refill_sections(self, cfg, system, user, empty, obj, seg, prefix,
                         allowed, names, audio_tag, audio_role, subject_tag,
                         clip_vocals, images, opts):
        """Ask once more for just the empty sections. Returns (prompt, still)."""
        ask = (user + "\n\nYOUR PREVIOUS ANSWER LEFT %s EMPTY. Return ONLY %s "
               "for this segment. What you wrote for detailed_description "
               "was:\n%s" % (" and ".join(empty), " and ".join(empty),
                             engine.clean(obj.get("detailed_description"))[:1500]))
        try:
            extra, _note = engine.generate(
                cfg, system, ask,
                engine.json_schema({k: {"type": "string"} for k in empty},
                                   list(empty)),
                required_keys=tuple(empty), images=images)
        except Exception:
            extra = None
        merged = dict(obj)
        for key in empty:
            if extra and engine.clean(extra.get(key)):
                merged[key] = extra[key]
        prompt, _ = self._post_process(
            merged, seg, prefix, allowed, names, audio_tag, audio_role,
            subject_tag, clip_vocals, **opts)
        return prompt, self._empty_sections(prompt)

    def _user_message(self, timeline, seg, index, prefix, cast_block, idea,
                      source, speakers, previous_body="", audio_tag="",
                      clip_vocals=None, plan=None, max_shots=0):
        segments = timeline["segments"]
        context = timeline.get("context", {})
        target = seg["target_duration"]
        render = seg.get("render_duration") or target

        lines = [
            "SEGMENT %d OF %d in a %.2f-second video."
            % (index + 1, len(segments),
               sum(s["target_duration"] for s in segments)),
            "",
            "STYLE PREFIX (verbatim at the start of [Shot 1]): %s"
            % (prefix or "(none supplied — choose one and keep it consistent)"),
            "",
            "DURATION",
            "  rendered duration: %.3f seconds — no timestamp may reach this" % render,
            "  planned duration:  %.3f seconds — all essential action ends by here" % target,
            "  the last %.3f seconds is a hold on the final composition" % (render - target),
            "",
            "CAST — the only tags you may cite:",
            cast_block,
        ]
        if context.get("subject_definitions") and source == "idea" and plan:
            lines += ["", "THE CAST FOR THE WHOLE VIDEO — these numbers are "
                      "fixed: never swap them. For every <Subject N> you cite, "
                      "copy its line below into subject_definitions word for "
                      "word:", context["subject_definitions"]]
        elif context.get("subject_definitions"):
            lines += ["", "HOW THE TREATMENT DESCRIBED THESE SUBJECTS (reuse this "
                      "wording so every segment agrees):",
                      context["subject_definitions"]]
        if speakers:
            lines += ["", "SPEAKER IDS ALREADY IN USE (keep them):"] + \
                ["  %s" % k for k in sorted(speakers)]

        carry_on = (seg.get("source") or {}).get("continuation")
        if source == "treatment" and carry_on:
            # The treatment stopped before the video did. Told nothing, the
            # model re-describes the last shot for every remaining segment.
            lines += ["", "WHAT HAPPENS IN THIS SEGMENT — the treatment has no "
                      "shots for this part of the video (%.3fs to %.3fs, part "
                      "%d of %d after its last shot). Continue the piece from "
                      "where the previous segment ends: the same world, cast, "
                      "style and energy, with new moments and new camera "
                      "angles. Do not repeat an earlier shot."
                      % (seg["source"]["start"], seg["source"]["end"],
                         carry_on["part"], carry_on["of"]),
                      "", "THE TREATMENT'S LAST SHOT, for continuity only: "
                      + vocals.TIMED_CUE_RE.sub(
                          "", carry_on.get("after_text", "")).strip()[:300]]
        elif source == "treatment" and seg.get("source"):
            lines += ["", "WHAT HAPPENS IN THIS SEGMENT — from the treatment, "
                      "timestamps already rebased to this segment's zero. "
                      "Number the shots from 1:"]
            for shot in seg["source"]["shots"]:
                # The treatment's vocal cues are on the whole video's clock;
                # this clip's own are stated below.
                text = vocals.TIMED_CUE_RE.sub("", shot["text"]).strip()
                if shot.get("of", 1) > 1:
                    # One long shot split across segments: each covers its
                    # own part, or the same action renders several times.
                    text += ("  (this segment is part %d of %d of that shot: "
                             "cover only that part of its action)"
                             % (shot["part"], shot["of"]))
                lines.append("  at %.3fs: %s" % (shot["at"], text))
            if clip_vocals:
                lines += ["", "VOCALS IN THIS CLIP — from the song's timeline, "
                          "and it overrides anything above:"]
                if clip_vocals["state"] == vocals.INSTRUMENTAL:
                    lines.append("  none. The audio is instrumental for the whole "
                                 "clip: nobody sings, raps or lip-syncs.")
                elif clip_vocals["state"] == vocals.VOCALS:
                    lines.append("  sung throughout.")
                else:
                    lines.append("  sung only from %s; instrumental otherwise, "
                                 "with every mouth closed." % " and from ".join(
                                     "%.3fs to %.3fs" % s
                                     for s in clip_vocals["spans"]))
            prev = segments[index - 1] if index > 0 else None
            nxt = segments[index + 1] if index + 1 < len(segments) else None
            if prev and prev.get("source", {}).get("shots"):
                lines += ["", "ENDS THE PREVIOUS SEGMENT (continuity only, do not "
                          "re-describe): " + vocals.TIMED_CUE_RE.sub(
                              "", prev["source"]["shots"][-1]["text"]).strip()[:300]]
            if nxt and nxt.get("source", {}).get("shots"):
                lines += ["", "BEGINS THE NEXT SEGMENT (so you can hand over "
                          "cleanly): " + vocals.TIMED_CUE_RE.sub(
                              "", nxt["source"]["shots"][0]["text"]).strip()[:300]]
        else:
            lines += self._idea_brief(segments, seg, index, plan)
            if max_shots:
                lines += ["", "SHOTS: write at most %d shot(s), [Shot 1] to "
                          "[Shot %d], each at least %.1f seconds long. Put the "
                          "energy in the camera and the performance, not in "
                          "more cuts." % (max_shots, max_shots,
                                          MIN_SHOT_SECONDS)]
            if clip_vocals:
                lines += ["", "VOCALS IN THIS CLIP — from the idea's timing, "
                          "and it overrides anything above:"]
                if clip_vocals["state"] == vocals.INSTRUMENTAL:
                    lines.append("  none. The audio is instrumental for the whole "
                                 "clip: nobody sings, raps or lip-syncs, and every "
                                 "mouth stays closed.")
                elif clip_vocals["state"] == vocals.VOCALS:
                    lines.append("  sung throughout.")
                else:
                    lines.append("  sung only from %s; instrumental otherwise, "
                                 "with every mouth closed." % " and from ".join(
                                     "%.3fs to %.3fs" % sp
                                     for sp in clip_vocals["spans"]))

        if idea.strip():
            lines += ["", "OVERALL IDEA AND DIRECTION:", idea.strip()]
        if previous_body:
            lines += ["", "THE PREVIOUS SEGMENT'S DESCRIPTION — this one must "
                      "describe DIFFERENT action, framing and camera work. Do "
                      "not restate it:", previous_body[:500]]
        if seg.get("link") == "continue":
            lines += ["", "This segment continues directly from the previous one "
                      "with no cut: open on the state it ended in."]
        if seg.get("notes"):
            lines += ["", "DIRECTION FOR THIS SEGMENT: " + seg["notes"]]
        return "\n".join(lines)

    @staticmethod
    def _idea_brief(segments, seg, index, plan):
        """What an idea-mode segment is told about its place and its action."""
        start = float(seg.get("audio_start") or 0.0)
        total = sum(s["target_duration"] for s in segments)
        lines = ["", "WHERE THIS SEGMENT SITS: %.2fs to %.2fs of the whole "
                 "%.2fs video, and of the audio when one is connected. Any "
                 "instruction in the idea about a time applies here only if "
                 "that time falls in this span."
                 % (start, start + seg["target_duration"], total)]
        beats = (plan or {}).get("beats") or []
        beat = beats[index] if index < len(beats) else None
        if not beat:
            return lines + [
                "", "WHAT HAPPENS IN THIS SEGMENT:",
                "  %s — invent the action for this beat, consistent with the "
                "idea below and with the segments around it."
                % (seg.get("beat") or "unspecified")]
        who = {r["number"]: r["who"] for r in plan.get("cast") or []}
        lines += ["", "WHAT HAPPENS IN THIS SEGMENT — from the plan for the "
                  "whole video. Follow it:", "  " + beat["action"]]
        if beat["featured"]:
            lines.append("  on screen: " + ", ".join(
                "<Subject %d> (%s)" % (n, who.get(n, "")) for n in beat["featured"]))
        if plan.get("performance"):
            singer = plan.get("performer")
            lines.append(
                "  %s %s — use that verb, not another.%s"
                % ("<Subject %d> (%s)" % (singer, who.get(singer, ""))
                   if singer else "the performer", plan["performance"],
                   " Nobody else's mouth moves to the words." if singer else ""))
        before = beats[index - 1] if index > 0 else None
        after = beats[index + 1] if index + 1 < len(beats) else None
        if before:
            lines += ["", "THE SEGMENT BEFORE (continuity only, do not "
                      "re-describe): " + before["action"][:300]]
        if after:
            lines += ["", "THE SEGMENT AFTER (so you can hand over cleanly): "
                      + after["action"][:300]]
        return lines

    def _post_process(self, obj, seg, prefix, allowed, names=(), audio_tag="",
                      audio_role="background only", subject_tag="",
                      clip_vocals=None, canon="", jobs=None, idea_mode=False,
                      performance="", idea="", max_shots=0, cast=None,
                      performer=None):
        """Enforce in code what the model reliably gets wrong.

        ``canon`` is the shared cast wording: a subject cited but not defined
        takes its definition from there. ``jobs`` is what each picture is for.

        Returns (prompt, invented_tags).
        """
        render = seg.get("render_duration") or seg["target_duration"]
        prompt = {k: engine.clean(obj.get(k)) or "N/A" for k in SECTIONS}
        prompt = {k: strip_asset_names(v, names) or "N/A"
                  for k, v in prompt.items()}
        invented = []
        if allowed:
            # Bind before validating: a bare "Subject 1" is a citation the
            # model forgot to bracket, not an unknown tag.
            wide = (with_defined_subjects({"subject_definitions": canon},
                                          allowed) if canon else allowed)
            prompt, _bound = bind_tags(prompt, wide)
            prompt, _filled = fill_definitions(prompt, canon)
            # A subject nothing defines loses its tag, not its sentence.
            prompt, renamed = name_undefined_subjects(prompt, allowed)
            prompt, invented = strip_unknown_tags(prompt, allowed)
            invented = sorted(set(invented) | set(renamed))
        changes = []
        if idea_mode and jobs:
            prompt, changes = enforce_picture_jobs(prompt, jobs, performance,
                                                   idea)
        if idea_mode:
            prompt, dropped = drop_placeholder_dialogue(prompt)
            if dropped:
                changes.append("%d spoken line(s) with no words removed"
                               % dropped)
            prompt, others = only_performer_sings(prompt, performer)
            if others:
                changes.append("singing taken from %d sentence(s) about "
                               "someone other than <Subject %d>"
                               % (others, performer))
            fixed = fix_summary_type(prompt.get("summary"), cast, audio_role)
            if fixed != prompt.get("summary"):
                prompt["summary"] = fixed
                changes.append("summary task type set")
        if audio_tag:
            prompt, _ = enforce_audio_exact(prompt, audio_tag, audio_role,
                                            subject_tag, clip_vocals)
        # Not engine.dedupe_subjects: it renumbers what it keeps from 1.
        prompt["subject_definitions"] = (
            dedupe_definitions(prompt["subject_definitions"]) or "N/A")

        body = prompt["detailed_description"]
        if prefix:
            body = self._force_prefix(body, prefix)
        # Renumber before the timing pass: fix_shot_times respaces by position,
        # and a segment cut from shots 4-6 comes back numbered 4, 5, 6.
        body = renumber_shots(body)
        if max_shots:
            body, folded = fold_shots(body, max_shots)
            if folded:
                changes.append("%d extra shot(s) folded into shot %d"
                               % (folded, max_shots))
        prompt["detailed_description"] = engine.fix_shot_times(body, render)
        if max_shots:
            shots = len(_SHOT_HEAD.findall(prompt["detailed_description"]))
            prompt["retention_analysis"] = prune_shot_refs(
                prompt["retention_analysis"], max(1, shots))
        prompt, _retained = complete_retention(prompt, jobs)
        self._last_changes = changes

        parsed = render_keeping_labels(prompt, render)
        for key in SECTIONS:
            parsed.setdefault(key, "N/A")

        if invented:
            seg["tag_warnings"] = ["invented %s" % t for t in invented]
        return {k: parsed[k] for k in SECTIONS}, invented

    @staticmethod
    def _force_prefix(body, prefix):
        marker = splitter.SHOT_RE.search(body)
        if not marker:
            return "[Shot 1] %s, %s" % (prefix, body.lstrip())
        head, rest = body[:marker.end()], body[marker.end():].lstrip()
        if rest.lower().startswith(prefix.lower()):
            return body
        return "%s %s, %s" % (head, prefix, rest)

    @staticmethod
    def _tag_warnings(prompt, allowed):
        out = []
        for kind, numbers in cited_tags(prompt).items():
            for n in sorted(numbers - allowed.get(kind, set())):
                out.append("cites <%s %d>, which the cast does not have" % (kind, n))
        return out

    @staticmethod
    def _closest(body, bodies, threshold=0.88):
        """Nearest earlier description, when it is close enough to be a repeat.

        Segments cut from adjacent shots invite the model to write the same
        paragraph twice; that is invisible until the whole video is stitched
        and every clip looks alike.
        """
        best, best_score = None, 0.0
        for i, other in enumerate(bodies):
            score = difflib.SequenceMatcher(None, body, other).ratio()
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score < threshold:
            return None, best_score
        return best, best_score

    @staticmethod
    def _collect_speakers(prompt, speakers):
        for match in re.finditer(r"\((S\d+(?:,S\d+)*)\)", flatten_prompt(prompt)):
            for sid in match.group(1).split(","):
                speakers.setdefault(sid.strip(), True)


NODE_CLASS_MAPPINGS = {"H3PlannerSegmentPrompter": H3PlannerSegmentPrompter}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PlannerSegmentPrompter": "H3 Segment Prompter"}
