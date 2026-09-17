"""Which clips are sung and which are instrumental, from the treatment.

The H3 Full-Reference Prompt Creator transcribes the song and writes a line
into the treatment's overall_soundscape:

    Vocal timeline of <Audio 1>: 00:00.000-00:25.300 instrumental;
    00:25.300-00:27.100 vocals; ...

The treatment's clock and the track's clock are the same clock, so a
segment's window in the video is also its window in the song, and this module
says what that window contains.

It exists because "performed on camera" used to be applied to every segment
alike. A music video with an instrumental intro then showed the singer
mouthing words over it: the audio role said the voice belongs to someone on
screen, and nothing said when there was no voice.

Deliberately free of the prompt-engine bridge, like the Slicer that uses it.
"""

import re

VOCALS = "vocals"
INSTRUMENTAL = "instrumental"
MIXED = "mixed"

_LINE_RE = re.compile(
    r"Vocal timeline of\s*(<\s*Audio\s*\d+\s*>)\s*:\s*(.+?)(?:\.\s*$|\.\s+[A-Z<]|$)",
    re.IGNORECASE | re.DOTALL)
_ENTRY_RE = re.compile(
    r"(\d{1,2}):(\d{2}(?:\.\d{1,3})?)\s*-\s*(\d{1,2}):(\d{2}(?:\.\d{1,3})?)"
    r"\s+(vocals|instrumental)", re.IGNORECASE)

# A vocal spilling this far over a cut is not a reason to sing in the next clip.
EDGE_TOLERANCE = 0.3
# At or above this share of a clip sung, the clip is simply "sung".
FULL_COVERAGE = 0.85

# A sentence showing someone voice the track. Identical to _SINGING_RE in the
# prompt creator, which tests/test_vocals.py checks; widened after "delivers
# 'Just a build free world'" got past the first version.
SINGING_RE = re.compile(
    r"<d>"
    r"|\b(?:sing|sings|singing|sang|sung|raps|rapping|rapped|lip[- ]?sync\w*|lipsync\w*|"
    r"mouths? (?:the )?(?:words|lyrics|along)|mouth\w* match\w*|belt(?:s|ing)? out|croon\w*|"
    r"vocali[sz]\w*|scream(?:s|ing|ed)?|shout(?:s|ing|ed)?|yell(?:s|ing|ed)?|growl(?:s|ing|ed)?|"
    r"chant(?:s|ing)?)\b"
    r"|\bdeliver(?:s|ing|ed)?\s+(?:(?:the|a|his|her|their)\s+)?(?:lines?|lyrics?|verse|chorus|hook|bars?)\b"
    r"|\bdeliver(?:s|ing|ed)?\s+['\"‘“]"
    r"|\bperform(?:s|ing|ed)?\s+(?:(?:the|a)\s+)?(?:lines?|lyrics?|verse|chorus|song|hook)\b",
    re.IGNORECASE)

# The prompt creator's per-shot cues. Those with a time in them, or saying a
# shot is instrumental, were written on the whole video's clock for a whole
# shot; a segment restates them on its own clock. Its plain lyric cue ("carries
# the vocal here, <d>...</d>") is kept on a sung clip — the words are right.
TIMED_CUE_RE = re.compile(
    r"\s*<Audio\s*\d+>\s+(?:is instrumental\b|carries the vocal only from\b)"
    r".*?(?:</d>)?[^.<]*\.", re.IGNORECASE | re.DOTALL)
ANY_CUE_RE = re.compile(
    r"\s*<Audio\s*\d+>\s+(?:is instrumental|carries the vocal)\b"
    r".*?(?:</d>)?[^.<]*\.", re.IGNORECASE | re.DOTALL)
_SHOT_HEAD_RE = re.compile(
    r"^\s*\[\s*Shot\s+\d+\s*\]\s*(?:At\s*\d{1,2}:\d{2}(?:\.\d+)?\s*,?)?",
    re.IGNORECASE)


def parse(text):
    """(audio label, sections) from a treatment's soundscape; ("", []) if none."""
    match = _LINE_RE.search(text or "")
    if not match:
        return "", []
    sections = []
    for m in _ENTRY_RE.finditer(match.group(2)):
        sections.append({
            "start": int(m.group(1)) * 60 + float(m.group(2)),
            "end": int(m.group(3)) * 60 + float(m.group(4)),
            "kind": m.group(5).lower(),
        })
    label = re.sub(r"\s+", "", match.group(1))
    label = label.replace("<Audio", "<Audio ")
    return label, sections


def window(sections, start, end):
    """What a stretch of the track holds, with sung spans rebased to its zero."""
    start, end = float(start), float(end)
    length = max(1e-6, end - start)
    spans = []
    for s in sections or []:
        if s["kind"] != VOCALS:
            continue
        a, b = max(start, s["start"]), min(end, s["end"])
        if b > a:
            spans.append((round(a - start, 3), round(b - start, 3)))
    sung = sum(b - a for a, b in spans)
    if sung < EDGE_TOLERANCE:
        return {"state": INSTRUMENTAL, "spans": [], "sung_seconds": round(sung, 3)}
    if sung / length >= FULL_COVERAGE:
        return {"state": VOCALS, "spans": spans, "sung_seconds": round(sung, 3)}
    return {"state": MIXED, "spans": spans, "sung_seconds": round(sung, 3)}


def for_segment(context, seg):
    """This segment's vocal state, or None when the treatment has no timeline."""
    sections = (context or {}).get("vocal_timeline") or []
    if not sections:
        return None
    source = seg.get("source") or {}
    start = source.get("start", seg.get("audio_start", 0.0))
    end = source.get("end", float(start) + float(seg.get("target_duration") or 0.0))
    return window(sections, start, end)


def timestamp(seconds):
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    return "%02d:%02d.%03d" % (millis // 60000, (millis // 1000) % 60, millis % 1000)


def cue(audio_tag, vocals):
    """The one sentence a clip's description gets about its own vocal, or ""."""
    if not vocals or vocals["state"] == VOCALS:
        return ""
    if vocals["state"] == INSTRUMENTAL:
        return ("%s is instrumental throughout this clip: nobody sings or raps, "
                "and every mouth stays closed." % audio_tag)
    spans = " and from ".join("%s to %s" % (timestamp(a), timestamp(b))
                              for a, b in vocals["spans"])
    return ("%s carries the vocal only from %s of this clip; outside those "
            "times it is instrumental and every mouth stays closed."
            % (audio_tag, spans))


def describe(vocals):
    """A short report fragment for one segment."""
    if not vocals:
        return ""
    if vocals["state"] == MIXED:
        return "vocal %s" % ", ".join("%.1f-%.1fs" % s for s in vocals["spans"])
    return vocals["state"]


def _sentences(text):
    """Sentences, without breaking a <d>...</d> lyric apart."""
    parts, buf, depth = [], "", 0
    for token in re.split(r"(<d>|</d>|(?<=[.!?])\s+)", text or ""):
        if not token:
            continue
        if token == "<d>":
            depth += 1
        elif token == "</d>":
            depth = max(0, depth - 1)
        if depth == 0 and not token.strip():
            if buf.strip():
                parts.append(buf.strip())
            buf = ""
            continue
        buf += token
    if buf.strip():
        parts.append(buf.strip())
    return parts


def apply(body, audio_tag, vocals):
    """Make one clip's description agree with its vocal state.

    Returns (body, number of singing sentences removed). With no timeline the
    body comes back untouched: the old behaviour, exactly.

    A removed sentence keeps its shot marker and whatever came before the
    singing in it. Shot 1 reads "[Shot 1] gritty handheld, <Subject 1> screams
    into the mic"; dropping the whole sentence took the clip's look with it.
    """
    if not vocals:
        return body, 0
    instrumental = vocals["state"] == INSTRUMENTAL
    text = (ANY_CUE_RE if instrumental else TIMED_CUE_RE).sub("", body or "")
    removed = 0
    if instrumental:
        kept = []
        for sentence in _sentences(text):
            if SINGING_RE.search(sentence):
                removed += 1
                # A sentence carrying a shot marker keeps the marker: losing it
                # would merge two shots into one.
                head = _SHOT_HEAD_RE.match(sentence)
                lead = []
                for clause in sentence[head.end() if head else 0:].split(","):
                    if SINGING_RE.search(clause):
                        break
                    lead.append(clause.strip())
                keep = " ".join(filter(None, [
                    head.group(0).strip() if head else "",
                    (", ".join(c for c in lead if c) + ",") if any(lead) else ""]))
                if keep:
                    kept.append(keep)
                continue
            kept.append(sentence)
        text = " ".join(kept)
    line = cue(audio_tag, vocals)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return ((text + " " + line).strip() if line else text), removed
