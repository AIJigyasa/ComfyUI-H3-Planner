"""H3 Segment Slicer — a treatment into segment prompts, with no model at all.

The Segment Prompter asks a language model to *rewrite* each slice of a
treatment as a standalone H3 prompt. That rewriting is the only reason it needs
Ollama, and when a treatment already exists most of it is unnecessary: the
treatment carries the shared sections, and cutting it is arithmetic.

It is also a source of drift. Asked to rewrite, a model paraphrases, and a
paraphrase can contradict the very style prefix it was told to preserve —
"cinematic, literary music-video style" written under a prefix that says
"2d, illustration art style". This node cannot do that, because it never
rewrites anything: it slices, renumbers, rebases the clock and copies.

What it is not: a planner. With no treatment there is nothing to cut, and no
amount of string handling invents a shot list. Use the Story Planner for that.

Deliberately free of the prompt-engine bridge. Nothing here needs a provider, a
key or a network, so it works on a machine with no Ollama installed at all.
"""

import hashlib
import json
import re
import time

from . import ladder, splitter, store, vocals
from .nodes_plan import CATEGORY, cited_tags, flatten_prompt
from .nodes_prompt import (AUDIO_ROLES, SECTIONS, _allowed_tags, _vocal_summary,
                           asset_names,
                           bind_tags,
                           canonical_subjects, enforce_audio_exact,
                           strip_asset_names, strip_unknown_tags)

# A shot marker plus, optionally, the timestamp the treatment gave it. Reused
# from the splitter so both agree on what a marker looks like; matched at the
# head of a shot's own text, which is where the splitter cut it.
_LEAD_PUNCT = re.compile(r"^\s*[-‒–—―:,•]+\s*")
_SHOT_ANY = re.compile(r"\[\s*Shot\s+\d+\s*\]", re.IGNORECASE)

# Enough of a gap that two shots never share an instant after rounding.
_MIN_GAP = 0.05
# H3 must never see a timestamp at or past the rendered length.
_TAIL_GUARD = 0.05


def _starts_with(text, prefix):
    """Does this body already open with the style prefix?

    Compared on letters and digits alone, so a trailing comma, a doubled space
    or a difference in case does not read as a different prefix.
    """
    def flat(value):
        return re.sub(r"[^a-z0-9]+", "", (value or "").lower())
    head = flat(prefix)
    return bool(head) and flat(text).startswith(head)


def format_timestamp(seconds):
    """MM:SS.mmm, the only form H3's guide accepts."""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    return "%02d:%06.3f" % (minutes, seconds - minutes * 60)


def rebase_times(shots, render_duration):
    """Segment-relative, strictly increasing, all inside the render.

    The splitter already rebased each shot's ``at`` to the segment's zero, but
    a treatment can hand back equal or backwards times, and a segment whose
    last shot sits past the rendered length gives H3 a cue it can never reach.
    """
    ceiling = max(0.0, float(render_duration) - _TAIL_GUARD)
    times = []
    for i, shot in enumerate(shots):
        at = 0.0 if i == 0 else max(0.0, float(shot.get("at") or 0.0))
        if i and times and at <= times[-1]:
            at = times[-1] + _MIN_GAP
        times.append(at)

    # Pull the whole run back under the ceiling rather than clamping the tail,
    # which would pile every late shot onto the same instant.
    if times and ceiling > 0 and times[-1] > ceiling:
        span = times[-1]
        scale = ceiling / span if span else 1.0
        times = [t * scale for t in times]
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                times[i] = min(ceiling, times[i - 1] + _MIN_GAP)
    return times


def shot_body(shots, render_duration, prefix=""):
    """The segment's detailed_description: sliced, renumbered, re-clocked."""
    if not shots:
        return ""
    times = rebase_times(shots, render_duration)
    parts = []
    for i, shot in enumerate(shots):
        text = str(shot.get("text") or "")
        match = splitter.SHOT_RE.match(text)
        rest = text[match.end():] if match else text
        # The marker's own trailing separator ("— ", "- ", ": ") belongs to the
        # marker, not the sentence.
        rest = _LEAD_PUNCT.sub("", rest).strip()
        # Any marker left inside the body would renumber the segment wrongly.
        rest = _SHOT_ANY.sub("", rest).strip()
        if i == 0:
            head = "[Shot 1]"
            # The prefix is usually lifted OFF this very shot, so prepending it
            # blindly wrote it twice: "[Shot 1] cinematic 35mm, cinematic 35mm,
            # a man steps onto the roof."
            if prefix and not _starts_with(rest, prefix):
                rest = "%s, %s" % (prefix.rstrip(" ,"), rest)
        else:
            head = "[Shot %d] At %s" % (i + 1, format_timestamp(times[i]))
        parts.append(("%s %s" % (head, rest)).strip())
    return " ".join(p for p in parts if p)


def dedupe_lines(text):
    """One definition per tag, first wording wins."""
    seen, kept = set(), []
    for line in re.split(r"(?:\r?\n|(?<=[.!?])\s+)", text or ""):
        line = line.strip()
        if not line:
            continue
        key = re.sub(r"\s+", " ", line.lower())
        if key in seen:
            continue
        seen.add(key)
        kept.append(line)
    return "\n".join(kept)


def _hash(*parts):
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _parse_cuts(raw):
    """The Beat Map's cuts output, or [] when nothing is wired."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        raise ValueError(
            "the `cuts` input is not the JSON the H3 Beat Map produces. Wire "
            "its `cuts` output, or leave it empty to cut on the treatment.")
    cuts = data.get("cuts") if isinstance(data, dict) else data
    if not isinstance(cuts, list) or not cuts:
        raise ValueError("the `cuts` input carries no cuts")
    for cut in cuts:
        if not isinstance(cut, dict) or "start" not in cut or "end" not in cut:
            raise ValueError("a cut is missing its start or end")
    return cuts


class H3PlannerSegmentSlicer:
    """Treatment in, prompted timeline out. No model, no key, no network.

    Connect the same ``h3_prompt`` you would give the Segment Prompter. Every
    segment gets the treatment's own shared sections and its own shots, with
    the clock rebased to that segment's zero.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "h3_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "the h3_prompt output of your prompt creator, run at the FULL video duration. This node cuts it on its [Shot N] markers; it cannot invent one."}),
                "total_seconds": ("FLOAT", {
                    "default": 60.0, "min": 1.0, "max": 600.0, "step": 0.5,
                    "tooltip": "the whole video's length — the same number you gave the prompt creator"}),
                "shots_per_segment": ("INT", {
                    "default": 2, "min": 1, "max": 12,
                    "tooltip": "how many of the treatment's shots to pack into one clip, as far as the segment cap allows"}),
                "min_segment_seconds": ("FLOAT", {
                    "default": 2.0, "min": 0.2, "max": 15.0, "step": 0.1,
                    "tooltip": "a tail shorter than this is folded into the segment before it"}),
                "audio_role": (list(AUDIO_ROLES), {
                    "default": "performed on camera",
                    "tooltip": "who makes the sound in the connected audio. Ignored when the cast has no audio asset."}),
                "carry_sections": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "copy the treatment's summary, retention, soundscape and music into every segment. Off writes a minimal stand-in instead."}),
            },
            "optional": {
                "cast": ("H3_CAST",),
                "cuts": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "the `cuts` output of H3 Beat Map. Wired, the cuts land on the music instead of on the treatment's own shot markers, and shots_per_segment is ignored."}),
                "style_prefix_override": ("STRING", {
                    "default": "",
                    "tooltip": "blank = lift it off [Shot 1] of the treatment"}),
            },
        }

    RETURN_TYPES = ("H3_TIMELINE", "STRING")
    RETURN_NAMES = ("timeline", "report")
    FUNCTION = "slice_treatment"
    CATEGORY = CATEGORY

    def slice_treatment(self, project, h3_prompt, total_seconds,
                        shots_per_segment, min_segment_seconds, audio_role,
                        carry_sections, cast=None, style_prefix_override="",
                        cuts=""):
        started = time.time()
        treatment = str(h3_prompt or "").strip()
        if not treatment:
            raise ValueError(
                "nothing to slice — connect the h3_prompt output of your "
                "prompt creator. This node has no model and cannot invent a "
                "shot list; use the H3 Story Planner to plan from an idea.")

        ceiling_frames, ceiling_seconds = ladder.ceiling(
            project["fps"], project["max_seconds"],
            modulus=project["frame_modulus"],
            remainder=project["frame_remainder"],
            minimum=project["frame_minimum"])

        supplied = _parse_cuts(cuts)
        if supplied:
            # The music decided where the cuts fall; the treatment only says
            # what happens inside them.
            segments, context, warnings = splitter.split_on_cuts(
                treatment, supplied, float(total_seconds))
        else:
            segments, context, warnings = splitter.split_treatment(
                treatment, float(total_seconds), ceiling_seconds,
                float(min_segment_seconds), int(shots_per_segment))
        if not segments:
            raise ValueError(
                "no [Shot N] markers in h3_prompt, so there is nothing to cut "
                "on. Run your prompt creator at the full video duration and "
                "connect its h3_prompt output.")

        if style_prefix_override.strip():
            context["style_prefix"] = style_prefix_override.strip()
            warnings = [w for w in warnings if "style prefix" not in w]
        prefix = context.get("style_prefix", "")

        timeline = store.normalize_timeline(
            {"name": project["name"], "context": context,
             "segments": segments}, project)
        for seg in timeline["segments"]:
            frames = ladder.snap_frames(
                seg["target_duration"], project["fps"],
                modulus=project["frame_modulus"],
                remainder=project["frame_remainder"],
                minimum=project["frame_minimum"], direction=project["snap"])
            seg["render_frames"] = frames
            seg["render_duration"] = frames / float(project["fps"])

        self._adopt_locked(project, timeline)

        allowed = _allowed_tags(cast)
        names = asset_names(cast)
        audio_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                          if m.get("kind") == "Audio"), "")
        subject_tag = next((m["tag"] for m in (cast or {}).get("members", [])
                            if m.get("kind") == "Subject"), "")
        canon = str(context.get("subject_definitions") or "").strip()

        fingerprint = _hash(treatment, total_seconds, shots_per_segment,
                            min_segment_seconds, prefix, audio_role,
                            carry_sections, sorted(names),
                            project["fps"], project["max_seconds"])
        timeline_sections = context.get("vocal_timeline") or []
        vocal_counts = {}

        rows, notes, written, skipped = [], [], 0, 0
        for seg in timeline["segments"]:
            if seg.get("state") == store.LOCKED:
                rows.append("  %-8s locked, left alone" % seg["id"])
                skipped += 1
                continue
            clip_vocals = vocals.for_segment(context, seg) if audio_tag else None
            if clip_vocals:
                vocal_counts[clip_vocals["state"]] = (
                    vocal_counts.get(clip_vocals["state"], 0) + 1)
            prompt, stripped = self._build(
                seg, context, prefix, canon, allowed, names, audio_tag,
                audio_role, carry_sections, clip_vocals)
            seg["prompt"] = prompt
            seg["prompt_fingerprint"] = fingerprint
            seg["spec_hash"] = store.spec_hash(seg)
            written += 1
            if stripped:
                notes.append("%s: stripped %s"
                             % (seg["id"], ", ".join(stripped)))
            shots = (seg.get("source") or {}).get("shots") or []
            rows.append(
                "  %-8s %6.3fs -> %6.3fs (%3d f)  %d shot(s)  %d words%s"
                % (seg["id"], seg["target_duration"], seg["render_duration"],
                   seg["render_frames"], len(shots),
                   len(flatten_prompt(prompt).split()),
                   "  [%s]" % vocals.describe(clip_vocals) if clip_vocals else ""))

        empty = [s["id"] for s in timeline["segments"]
                 if not str((s.get("prompt") or {}).get(
                     "detailed_description", "")).strip()
                 or (s.get("prompt") or {}).get(
                     "detailed_description") == "N/A"]
        if empty:
            warnings.append(
                "no shot text reached %s — the treatment's markers were found "
                "but carried no body" % ", ".join(empty))

        store.save(project["timeline_path"], timeline)

        planned = sum(s["target_duration"] for s in timeline["segments"])
        rendered = sum(s["render_duration"] for s in timeline["segments"])
        report = "\n".join([
            "%d segment(s) sliced from the treatment in %.0f ms — no model, "
            "no key, no network" % (len(timeline["segments"]),
                                    (time.time() - started) * 1000),
            "written       %d, %d locked and left alone" % (written, skipped),
            "style prefix  %s" % (prefix or "(none found — set an override)"),
            "audio         %s" % ("%s, %s" % (audio_tag, audio_role)
                                  if audio_tag else "(none connected)"),
            "vocals        %s" % _vocal_summary(audio_tag, timeline_sections,
                                                vocal_counts),
            "soundscape    %s"
            % ("overall_soundscape and non_diegetic_music are REPLACED in "
               "every segment, and the %s line in retention_analysis set to "
               "fully_copy — a supplied track already contains everything "
               "that will be heard, so the treatment's own sound writing "
               "would tell H3 to synthesise more on top of it. Disconnect "
               "the audio asset to keep what the treatment wrote."
               % audio_tag
               if audio_tag else "the treatment's own, kept as written"),
            "cuts          %s"
            % ("%d supplied by the beat map" % len(supplied) if supplied
               else "from the treatment shot markers"),
            "sections      %s" % ("copied from the treatment" if carry_sections
                                  else "minimal stand-ins"),
            "planned       %.3fs -> H3 renders %.3fs (stitcher trims %.3fs back)"
            % (planned, rendered, rendered - planned),
            "ceiling       %.3fs (%d frames) per segment"
            % (ceiling_seconds, ceiling_frames),
            "",
            "segments:",
        ] + rows
            + (["", "NOTES", "- " + "\n- ".join(notes)] if notes else [])
            + (["", "WARNINGS", "! " + "\n! ".join(warnings)] if warnings else []))

        return (timeline, report)

    # -- building one segment --------------------------------------------

    def _build(self, seg, context, prefix, canon, allowed, names, audio_tag,
               audio_role, carry_sections, clip_vocals=None):
        """Every section of one segment, assembled from the treatment."""
        render = seg.get("render_duration") or seg["target_duration"]
        shots = (seg.get("source") or {}).get("shots") or []
        body = shot_body(shots, render, prefix)

        if carry_sections:
            prompt = {
                "subject_definitions": canon,
                "summary": str(context.get("summary") or "").strip(),
                "retention_analysis": str(
                    context.get("retention_analysis") or "").strip(),
                "detailed_description": body,
                "overall_soundscape": str(
                    context.get("overall_soundscape") or "").strip(),
                "non_diegetic_music": str(
                    context.get("non_diegetic_music") or "").strip(),
            }
        else:
            prompt = {k: "" for k in SECTIONS}
            prompt["subject_definitions"] = canon
            prompt["detailed_description"] = body
            prompt["summary"] = "Segment %d of %s." % (
                seg["index"] + 1, context.get("shot_count") or "the video")

        # Never let an upload's file name or a cast key reach the prompt.
        prompt = {k: strip_asset_names(v or "", names) for k, v in prompt.items()}

        # Cut the shared definitions down to the tags this segment actually
        # cites, falling back to the whole block rather than shipping nothing.
        if canon:
            filtered = canonical_subjects(canon, cited_tags(prompt))
            prompt["subject_definitions"] = filtered or canon
        prompt["subject_definitions"] = dedupe_lines(
            prompt["subject_definitions"])

        stripped = []
        if allowed:
            prompt, _bound = bind_tags(prompt, allowed)
            prompt, stripped = strip_unknown_tags(prompt, allowed)
        if audio_tag:
            prompt, _ = enforce_audio_exact(prompt, audio_tag, audio_role,
                                            subject_tag="",
                                            clip_vocals=clip_vocals)

        for key in SECTIONS:
            value = str(prompt.get(key) or "").strip()
            prompt[key] = value or "N/A"
        return {k: prompt[k] for k in SECTIONS}, stripped

    @staticmethod
    def _adopt_locked(project, timeline):
        """Carry locks, and the prompts they protect, over from disk."""
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
                if prev.get("prompt"):
                    seg["prompt"] = prev["prompt"]
                    seg["prompt_fingerprint"] = prev.get(
                        "prompt_fingerprint", "")


NODE_CLASS_MAPPINGS = {"H3PlannerSegmentSlicer": H3PlannerSegmentSlicer}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PlannerSegmentSlicer": "H3 Segment Slicer"}
