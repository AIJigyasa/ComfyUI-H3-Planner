"""Revise ONE segment's shot description from a note, in place.

The whole point is that nothing else moves. A plan is usually 80% right and one
shot is wrong, and replanning to fix that shot rewrites the other nine — new
wording for the cast, a different soundscape, a different script. So this takes
back only ``detailed_description`` and copies every other section verbatim.

The reference tags are guaranteed rather than requested: the set of tags this
segment may cite is read out of the prompt it already has, so a revision can
neither introduce a tag the cast lacks nor quietly drop the character.

Runs inside the ComfyUI server process, from the timeline's own routes, so a
card can be fixed without re-queueing the graph.

Works on any H3 timeline whatever wrote it — Story Planner, Segment Prompter or
Slicer — because it reads the finished prompt rather than the planner's state.
"""

import os
import re

from . import engine, splitter, store
from .nodes_plan import cited_tags, flatten_prompt
from .nodes_prompt import (SECTIONS, bind_tags, enforce_audio_exact,
                           renumber_shots, strip_unknown_tags)

# Kept out of the timeline deliberately: a key written to disk travels with
# every workflow JSON and every screenshot. The provider is remembered, the
# secret is read from the environment when it is needed.
BACKEND_FIELDS = store.BACKEND_FIELDS

REFINE_SYSTEM = """
You are revising ONE shot description inside an existing video prompt. A
director has given you a note about what is wrong with it.

Rewrite the description so it satisfies the note, and change NOTHING ELSE. This
is a revision, not a fresh draft: if a sentence is not affected by the note,
leave it alone.

WHAT MUST SURVIVE UNCHANGED

- Every reference tag exactly as it appears — <Subject 1>, <Picture 2>,
  <Audio 1>. Do not introduce a tag that is not already in the description and
  do not drop one that is. They bind the supplied photographs; a tag you remove
  is a character who stops appearing.
- Every spoken line, word for word, inside its <d>[Language] ...</d> tags,
  attached to the same speaker ID.
- The style prefix at the very start of [Shot 1].
- The segment's length. [Shot 1] carries no timestamp; later shots are
  "[Shot N] At MM:SS.mmm", strictly increasing and all below the rendered
  duration given to you.

Return only the revised detailed_description.
""".strip()



DEFAULT_BACKEND = {
    "provider": "Ollama (Local)",
    "ollama_url": "http://127.0.0.1:11434",
    "temperature": 0.3,
}


def backend_config(context):
    """Rebuild the provider config, taking the key from the environment.

    Falls back to local Ollama when the timeline records nothing. It often
    records nothing for a good reason: the Slicer and the Beat Map write a
    whole timeline without a model anywhere, so there is no provider to
    remember — and refusing to refine those made the feature useless for the
    entire music-video path, which is the path it was asked for. Refining needs
    *a* model, not the one that wrote the prompt.
    """
    saved = dict((context or {}).get("backend") or {})
    borrowed = not saved
    if borrowed:
        saved = dict(DEFAULT_BACKEND)
        saved["ollama_model"] = engine.default_ollama_model()
    saved.setdefault("provider", "Ollama (Local)")
    saved["api_key"] = ""       # engine falls back to the provider's env var
    try:
        return engine.backend_config(**saved), borrowed
    except TypeError:
        # An older engine that does not accept one of the saved keys.
        keep = {k: v for k, v in saved.items()
                if k in BACKEND_FIELDS or k == "api_key"}
        return engine.backend_config(**keep), borrowed


def _tag_set(prompt):
    """Every tag this prompt cites, as {kind: {numbers}}."""
    return {kind: set(numbers) for kind, numbers in cited_tags(prompt).items()}


def _spoken(text):
    """The dialogue lines in a description, for checking they survive."""
    return [m.strip() for m in
            re.findall(r"<d>\s*(?:\[[^\]]*\]\s*)?(.*?)</d>", text or "",
                       re.IGNORECASE | re.DOTALL) if m.strip()]


def _force_prefix(body, prefix):
    """The style prefix at the head of [Shot 1], exactly once."""
    body = (body or "").strip()
    if not prefix:
        return body
    flat = lambda v: re.sub(r"[^a-z0-9]+", "", (v or "").lower())
    head_flat = flat(prefix)
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


def _assemble(new_body, original, seg, context, allowed):
    """Enforce on the revision everything the original was held to."""
    prompt = dict(original)
    prompt["detailed_description"] = new_body

    if allowed:
        prompt, _ = bind_tags(prompt, allowed)
        prompt, _ = strip_unknown_tags(prompt, allowed)

    audio_tag = next(("<Audio %d>" % n for n in sorted(allowed.get("Audio", ()))),
                     "")
    if audio_tag:
        # Wanted for its invented-ambience strip on the description only; the
        # two sound sections are the author's and are restored below.
        prompt, _ = enforce_audio_exact(prompt, audio_tag)

    body = prompt["detailed_description"]
    body = _force_prefix(body, str((context or {}).get("style_prefix") or ""))
    body = renumber_shots(body)
    render = seg.get("render_duration") or seg["target_duration"]
    try:
        body = engine.fix_shot_times(body, render)
    except Exception:
        pass        # no engine bridge: the model's own timings stand

    # Everything except the description is the original, untouched.
    out = {k: original.get(k, "N/A") for k in SECTIONS}
    out["detailed_description"] = body.strip() or original.get(
        "detailed_description", "")
    return out


def refine_segment(timeline, seg, note, attempts=2):
    """Rewrite one segment's description to satisfy ``note``.

    Returns a human-readable status. Raises RuntimeError when nothing usable
    came back, leaving the segment exactly as it was.
    """
    note = (note or "").strip()
    if not note:
        raise RuntimeError("nothing to do — describe what should change.")
    original = seg.get("prompt")
    if not isinstance(original, dict) or not original.get("detailed_description"):
        raise RuntimeError(
            "%s has no written prompt yet. Plan it first, then refine."
            % seg["id"])
    if seg.get("state") == store.LOCKED:
        raise RuntimeError(
            "%s is locked. Unlock it before refining." % seg["id"])
    if not engine.available():
        raise RuntimeError(engine.MISSING)

    context = timeline.get("context") or {}
    cfg, borrowed = backend_config(context)
    schema = engine.json_schema({"detailed_description": {"type": "string"}},
                                ["detailed_description"])

    allowed = _tag_set(original)
    want_lines = _spoken(original["detailed_description"])
    render = seg.get("render_duration") or seg["target_duration"]

    tag_list = ", ".join(
        "<%s %d>" % (kind, n) for kind in sorted(allowed)
        for n in sorted(allowed[kind])) or "(none)"
    lines = [
        "THE NOTE FROM THE DIRECTOR:",
        note,
        "",
        "SEGMENT LENGTH",
        "  rendered duration: %.3fs — no timestamp may reach this" % render,
        "  planned duration:  %.3fs" % seg["target_duration"],
        "",
        "TAGS THAT MUST STILL APPEAR, AND NO OTHERS:",
        "  " + tag_list,
    ]
    if want_lines:
        lines += ["", "SPOKEN LINES THAT MUST SURVIVE WORD FOR WORD:"] + \
            ["  " + line for line in want_lines]
    if context.get("style_prefix"):
        lines += ["", "STYLE PREFIX, verbatim at the head of [Shot 1]:",
                  "  " + context["style_prefix"]]
    if context.get("world"):
        lines += ["", "THE WORLD, unchanged:", str(context["world"])[:600]]
    lines += ["", "THE CURRENT DESCRIPTION:",
              original["detailed_description"]]

    problems = []
    for attempt in range(1, int(attempts) + 1):
        user = "\n".join(lines)
        if attempt > 1:
            user += ("\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED: %s. Fix that and "
                     "keep everything else identical." % "; ".join(problems))
        try:
            obj, _ = engine.generate(cfg, REFINE_SYSTEM, user, schema,
                                     required_keys=("detailed_description",))
        except Exception as ex:
            if borrowed:
                raise RuntimeError(
                    "refining needs a language model, and this timeline "
                    "was written without one (the Slicer and Beat Map use "
                    "no model at all), so local Ollama at %s was tried and "
                    "failed: %s. Start Ollama, or plan this project once "
                    "with the Story Planner or Segment Prompter so the "
                    "provider is recorded."
                    % (DEFAULT_BACKEND["ollama_url"], ex))
            raise RuntimeError("the provider failed: %s" % ex)
        body = engine.clean((obj or {}).get("detailed_description"))
        if not body:
            problems = ["it returned nothing"]
            continue

        candidate = _assemble(body, original, seg, context, allowed)
        problems = []

        lost = []
        got = _tag_set(candidate)
        for kind, numbers in allowed.items():
            for n in numbers:
                if n not in got.get(kind, set()):
                    lost.append("<%s %d>" % (kind, n))
        if lost:
            problems.append("it dropped %s" % ", ".join(sorted(lost)))

        if want_lines:
            have = " ".join(_spoken(candidate["detailed_description"])).lower()
            missing = [l for l in want_lines
                       if l.lower()[:40] not in have]
            if missing:
                problems.append("it lost the spoken line %r"
                                % missing[0][:50])

        if problems:
            continue

        seg["prompt"] = candidate
        seg["spec_hash"] = store.spec_hash(seg)
        seg["refine_note"] = note
        words = len(flatten_prompt(candidate).split())
        return ("%s revised (%d words)%s. Its clip no longer matches the "
                "prompt, so it is back to pending and will render again."
                % (seg["id"], words,
                   " using local Ollama, since this timeline was written without a model" if borrowed else ""))

    raise RuntimeError(
        "%s was left unchanged: %s. Try a more specific note, or edit the "
        "prompt directly on the card."
        % (seg["id"], "; ".join(problems) or "nothing usable came back"))
