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

from . import engine, ladder, splitter, store
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


def audio_system_block(audio_tag, role):
    """The audio rules for this cast and this role, or "" when no audio."""
    if not audio_tag:
        return ""
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


def soundscape_line(audio_tag, role="background only", subject_tag=""):
    """What overall_soundscape is replaced with, for this audio role.

    The old single line said only that no extra sound is present. That is true
    and it is also the whole problem: it overwrote the one place a prompt says
    a voice belongs to a person on screen, so H3 had nothing telling it to
    animate a performance.
    """
    head = ("All audio in this segment comes from %s and is used exactly as "
            "supplied" % audio_tag)
    who = subject_tag or "the subject on screen"
    if role == "performed on camera":
        return ("%s: the vocal is performed on camera by %s, whose mouth "
                "matches the words throughout. No additional sound is present."
                % (head, who))
    if role == "voice-over (off camera)":
        return ("%s, heard as voice-over; no one on screen produces it and no "
                "mouth matches it. No additional sound is present." % head)
    return "%s. No additional sound is present." % head


def enforce_audio_exact(prompt, audio_tag, role="background only",
                        subject_tag=""):
    """Make the two sound sections say the audio is reused, and nothing else.

    A supplied track already contains everything that will be heard. Left
    alone the model writes "ambient street sounds blend with rhythmic beats,
    featuring layered percussion" — a description of sound to synthesise, on
    top of the sound it was given.

    ``role`` says whether a visible subject produces that audio. It changes
    only what the soundscape line asserts; the ban on inventing sound is the
    same in every role.
    """
    changed = []
    exact = soundscape_line(audio_tag, role, subject_tag)
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
                    "tooltip": "1 = one segment per shot in the treatment, and one prompt written for it — the most detail per clip. Raise it to pack several shots into one longer clip while they still fit under the segment cap."}),
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
            segments, context, warnings = splitter.split_treatment(
                treatment, total_seconds, ceiling_seconds, 2.0,
                shots_per_segment)
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
        audio_block = audio_system_block(audio_tag, audio_role)
        if audio_block:
            system += "\n\n" + audio_block

        speakers, rows, failed = {}, [], []
        written = skipped = 0
        bodies = []          # every description written so far, to catch repeats
        # ONE wording for the cast, reused verbatim in every segment. Seeded
        # from the treatment when it had definitions, otherwise from whichever
        # segment is written first.
        canon = engine.clean(context.get("subject_definitions") or "")
        canon_from = "the treatment" if canon else ""
        started = time.time()

        for i, seg in enumerate(timeline["segments"]):
            fingerprint = _hash(seg.get("source"), seg["target_duration"],
                                seg.get("render_duration"), prefix, cast_block,
                                idea, source, format, seed)

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

            prompt, obj, repeat, invented = None, None, "", []
            for attempt in (1, 2):
                user = self._user_message(
                    timeline, seg, i, prefix, cast_block, idea, source,
                    speakers, previous_body=bodies[-1] if bodies else "")
                if attempt == 2:
                    user += ("\n\nYOUR PREVIOUS ATTEMPT REPEATED AN EARLIER "
                             "SEGMENT ALMOST WORD FOR WORD. Write this segment "
                             "again from scratch: different action, different "
                             "framing, different camera move. Describe only what "
                             "the shot list above says happens here.")
                try:
                    obj, note = engine.generate(
                        cfg, system, user, schema,
                        required_keys=("summary", "retention_analysis",
                                       "detailed_description"),
                        min_words=120)
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
                    audio_role, subject_tag)
                twin, score = self._closest(prompt["detailed_description"], bodies)
                if invented and attempt == 1:
                    continue          # ask again before falling back to stripping
                if twin is None:
                    repeat = ""
                    break
                repeat = "  (%d%% like segment %d)" % (round(score * 100), twin + 1)
                if attempt == 2:
                    failed.append(
                        "%s still reads %d%% the same as segment %d — the model "
                        "is not differentiating; try a treatment with more "
                        "distinct shots, or raise temperature"
                        % (seg["id"], round(score * 100), twin + 1))

            if obj is None or prompt is None:
                continue

            if not canon:
                canon = prompt["subject_definitions"]
                canon_from = seg["id"]
            else:
                shared = canonical_subjects(canon, cited_tags(prompt))
                if shared:
                    prompt["subject_definitions"] = shared

            seg["prompt"] = prompt
            seg["prompt_fingerprint"] = fingerprint
            seg["spec_hash"] = store.spec_hash(seg)
            self._collect_speakers(prompt, speakers)
            bodies.append(prompt["detailed_description"])
            written += 1
            note = repeat
            if invented:
                note += "  (stripped %s — not in the cast)" % ", ".join(invented)
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
            "",
            "segments:",
        ] + rows
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

    def _user_message(self, timeline, seg, index, prefix, cast_block, idea,
                      source, speakers, previous_body=""):
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
        if context.get("subject_definitions"):
            lines += ["", "HOW THE TREATMENT DESCRIBED THESE SUBJECTS (reuse this "
                      "wording so every segment agrees):",
                      context["subject_definitions"]]
        if speakers:
            lines += ["", "SPEAKER IDS ALREADY IN USE (keep them):"] + \
                ["  %s" % k for k in sorted(speakers)]

        if source == "treatment" and seg.get("source"):
            lines += ["", "WHAT HAPPENS IN THIS SEGMENT — from the treatment, "
                      "timestamps already rebased to this segment's zero. "
                      "Number the shots from 1:"]
            for shot in seg["source"]["shots"]:
                lines.append("  at %.3fs: %s" % (shot["at"], shot["text"]))
            prev = segments[index - 1] if index > 0 else None
            nxt = segments[index + 1] if index + 1 < len(segments) else None
            if prev and prev.get("source", {}).get("shots"):
                lines += ["", "ENDS THE PREVIOUS SEGMENT (continuity only, do not "
                          "re-describe): " + prev["source"]["shots"][-1]["text"][:300]]
            if nxt and nxt.get("source", {}).get("shots"):
                lines += ["", "BEGINS THE NEXT SEGMENT (so you can hand over "
                          "cleanly): " + nxt["source"]["shots"][0]["text"][:300]]
        else:
            lines += ["", "WHAT HAPPENS IN THIS SEGMENT:",
                      "  %s — invent the action for this beat, consistent with "
                      "the idea below and with the segments around it."
                      % (seg.get("beat") or "unspecified")]

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

    def _post_process(self, obj, seg, prefix, allowed, names=(), audio_tag="",
                      audio_role="background only", subject_tag=""):
        """Enforce in code what the model reliably gets wrong.

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
            prompt, _bound = bind_tags(prompt, allowed)
            prompt, invented = strip_unknown_tags(prompt, allowed)
        if audio_tag:
            prompt, _ = enforce_audio_exact(prompt, audio_tag, audio_role,
                                            subject_tag)
        prompt["subject_definitions"] = engine.dedupe_subjects(
            prompt["subject_definitions"])

        body = prompt["detailed_description"]
        if prefix:
            body = self._force_prefix(body, prefix)
        # Renumber before the timing pass: fix_shot_times respaces by position,
        # and a segment cut from shots 4-6 comes back numbered 4, 5, 6.
        body = renumber_shots(body)
        prompt["detailed_description"] = engine.fix_shot_times(body, render)

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
