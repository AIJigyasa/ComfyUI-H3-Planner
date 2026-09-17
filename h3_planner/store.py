"""Timeline model, persistence, and the claim/complete state machine.

The rule that makes long renders survivable: a segment is *outstanding for a
pass* when it is not locked, is not currently running, and has no stored clip
for that pass. Claiming is therefore idempotent — queue forty runs against a
twelve-segment timeline and the extra twenty-eight find nothing outstanding and
no-op, rather than re-rendering anything.

Nothing here imports torch or ComfyUI, so it is testable standalone.
"""

import hashlib
import itertools
import json
import os
import tempfile
import threading
import time

VERSION = 1

PENDING = "pending"
RUNNING = "running"
FAILED = "failed"
LOCKED = "locked"

# Fields that describe what to render. Changing any of them invalidates the
# stored result for that segment; changing anything else (notes, state, clips)
# does not.
# first_frame_path is deliberately absent: it is derived by Vault Write from
# the previous segment's last frame, not authored, so writing it must never
# invalidate a stored clip.
SPEC_FIELDS = (
    "prompt", "target_duration", "seed", "refine_seed", "link",
    "audio_start", "cast_used", "negative_note",
)

_locks = {}
_locks_guard = threading.Lock()
_run_counter = itertools.count()


def run_token():
    """A value that can never compare equal to the previous one.

    IS_CHANGED conventionally returns float("nan") to force a node to re-run,
    which relies on NaN inequality surviving however the executor builds and
    compares its cache keys. On this ComfyUI build it does not: the dispatcher
    was served from cache and never executed again, so every queued run
    re-stored the SAME segment and auto-advance queued another — a loop that
    rendered nothing and never ended. A monotonic string cannot be mistaken
    for unchanged.
    """
    return "%.6f-%d" % (time.time(), next(_run_counter))


def _lock_for(path):
    with _locks_guard:
        lock = _locks.get(path)
        if lock is None:
            lock = _locks[path] = threading.Lock()
        return lock


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def spec_hash(segment):
    payload = {k: segment.get(k) for k in SPEC_FIELDS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def prompt_hash(prompt):
    """A stable fingerprint of a prompt, whether text or six sections."""
    blob = json.dumps(prompt if prompt is not None else "", sort_keys=True,
                      ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def derive_seed(base_seed, index, salt=0):
    """Deterministic per-segment seed, stable across re-plans."""
    raw = "%d:%d:%d" % (int(base_seed), int(index), int(salt))
    digest = hashlib.sha1(raw.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return value


def normalize_segment(raw, index, project):
    """Fill an authored segment out into the full schema.

    Authoring is deliberately forgiving: ``{"prompt": "...", "duration": 6}``
    is enough.
    """
    if isinstance(raw, str):
        raw = {"prompt": raw}
    if not isinstance(raw, dict):
        raise ValueError("segment %d is not an object or string" % index)

    seg_id = str(raw.get("id") or "seg_%02d" % (index + 1))
    duration = raw.get("target_duration", raw.get("duration", raw.get("seconds")))
    if duration is None:
        duration = float(project.get("default_duration", 6.0))
    duration = float(duration)
    if duration <= 0:
        raise ValueError("segment %s has non-positive duration" % seg_id)

    base_seed = int(project.get("base_seed", 0))
    seed = raw.get("seed")
    refine = raw.get("refine_seed")

    prompt = raw.get("prompt", "")
    if isinstance(prompt, dict):  # six named sections
        prompt = dict(prompt)

    seg = {
        "id": seg_id,
        "index": index,
        "beat": str(raw.get("beat", "")),
        "prompt": prompt,
        "target_duration": duration,
        "link": str(raw.get("link", "cut")),
        "cast_used": list(raw.get("cast_used", []) or []),
        "audio_start": (None if raw.get("audio_start") is None
                        else float(raw["audio_start"])),
        "first_frame_path": raw.get("first_frame_path"),
        "negative_note": str(raw.get("negative_note", "")),
        "seed": int(seed) if seed is not None else derive_seed(base_seed, index, 0),
        "refine_seed": (int(refine) if refine is not None
                        else derive_seed(base_seed, index, 1)),
        "notes": str(raw.get("notes", "")),
        "prompt_fingerprint": raw.get("prompt_fingerprint", ""),
        # Story continuity, written by the Story Planner and ignored by every
        # other producer. `scene` groups segments that share a location, a
        # look and a wardrobe; `opens_from` / `ends_with` are the physical
        # handover between consecutive shots — where the subject is, what is
        # in their hand, where the camera sits. Outside SPEC_FIELDS: they
        # shape the prompt that gets written, and that prompt is what decides
        # whether a stored clip is still valid.
        "scene": (None if raw.get("scene") is None else int(raw["scene"])),
        "scene_name": str(raw.get("scene_name", "")),
        "opens_from": str(raw.get("opens_from", "")),
        "ends_with": str(raw.get("ends_with", "")),
        # The words actually spoken on camera in this segment. Kept on the
        # segment rather than only inside the prompt so the card strip and the
        # beat sheet can show the script, and so a re-write reuses the same
        # line instead of inventing a new one.
        "dialogue": str(raw.get("dialogue", "")),
        # Which cast subject says it. Two characters talking need two voices,
        # and the prose call used to hardcode the first subject and (S1), so
        # only one of them could ever speak.
        "speaker": str(raw.get("speaker", "")),
        # The last plain-English note used to refine this shot, kept so
        # the card can show what was asked for.
        "refine_note": str(raw.get("refine_note", "")),
        # The prompt as the planner last delivered it. Comparing the card
        # against it is the only way to tell a refine or a hand edit from a
        # stale copy, so an edit can survive the planner re-running upstream.
        "planned_prompt_hash": str(raw.get("planned_prompt_hash", "")),
        # What the splitter cut this segment from: the source shots with their
        # timestamps rebased to zero. Read by the prompter, ignored by everyone
        # else, and outside SPEC_FIELDS because the prompt it produces is what
        # actually decides whether a stored clip is still valid.
        "source": raw.get("source"),
        # runtime, never authored
        "state": PENDING,
        "take": 0,
        "clips": {},
        "render_frames": None,
        "render_duration": None,
        "claimed_at": None,
        "last_error": "",
    }
    if raw.get("locked"):
        seg["state"] = LOCKED
    seg["spec_hash"] = spec_hash(seg)
    return seg


def normalize_timeline(authored, project):
    """Turn authored JSON (dict, or bare list of segments) into a timeline."""
    if isinstance(authored, list):
        authored = {"segments": authored}
    if not isinstance(authored, dict):
        raise ValueError("timeline must be a JSON object or array")

    segments_raw = authored.get("segments")
    if segments_raw is None:
        raise ValueError('timeline JSON has no "segments"')
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError('"segments" must be a non-empty array')

    segments = []
    seen = set()
    for i, raw in enumerate(segments_raw):
        seg = normalize_segment(raw, i, project)
        if seg["id"] in seen:
            raise ValueError("duplicate segment id %r" % seg["id"])
        seen.add(seg["id"])
        segments.append(seg)

    return {
        "version": VERSION,
        "project": project.get("name", "default"),
        "fps": float(project.get("fps", 24.0)),
        "name": str(authored.get("name", project.get("name", "default"))),
        # Shared across every segment: style prefix, the treatment's own
        # subject definitions, soundscape and music. The prompter copies the
        # style prefix into each segment verbatim — that is what stops segment
        # six drifting to a different look.
        "context": dict(authored.get("context") or {}),
        "segments": segments,
        "created": _now(),
        "updated": _now(),
    }


def merge(existing, fresh):
    """Carry finished work across a re-author.

    A segment keeps its clips, take count and state when its spec_hash is
    unchanged. Edit one prompt and only that segment goes back to pending.
    """
    if not existing or not existing.get("segments"):
        return fresh

    old = {s["id"]: s for s in existing["segments"]}
    for seg in fresh["segments"]:
        prev = old.get(seg["id"])
        if not prev:
            continue
        if prev.get("spec_hash") != seg["spec_hash"]:
            # Respec'd: stays pending with no clips. Dropping a clip is the
            # right call — it no longer matches the prompt — but doing it in
            # silence looked exactly like the renderer re-rendering the same
            # segment forever. Record it so the caller can say so out loud.
            if prev.get("clips"):
                fresh.setdefault("_discarded", []).append({
                    "id": seg["id"],
                    "passes": sorted(prev["clips"]),
                })
            continue
        # render_frames / render_duration are deliberately NOT carried over:
        # the caller recomputes them from the current fps and ladder just
        # before merging, and copying the old values back put segments on disk
        # claiming a 6.583s render against an 8.000s plan.
        for key in ("state", "take", "clips", "claimed_at", "last_error",
                    "chosen", "prompt_fingerprint"):
            if key in prev:
                seg[key] = prev[key]
        if seg.get("state") == RUNNING:
            # a claim that never completed; let the stale sweep decide
            seg["state"] = RUNNING
    if existing.get("context") and not fresh.get("context"):
        fresh["context"] = existing["context"]
    fresh["created"] = existing.get("created", fresh["created"])
    fresh["updated"] = _now()
    return fresh


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


# How to reach the model that wrote a timeline, so one card can be refined
# later without re-queueing the graph. The API KEY IS DELIBERATELY ABSENT: a
# key written here would travel with every workflow JSON and every screenshot.
# It is read from the provider's environment variable when refining.
BACKEND_FIELDS = ("provider", "ollama_url", "ollama_model", "api_model",
                  "temperature", "keep_alive", "timeout", "max_output_tokens",
                  "num_ctx")


def remember_backend(context, values):
    context["backend"] = {k: values.get(k) for k in BACKEND_FIELDS
                          if values.get(k) is not None}
    return context

def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            return data
    except FileNotFoundError:
        return None
    except Exception as ex:
        print("[H3Planner] timeline unreadable (%s): %s" % (path, ex))
    return None


def _replace_with_retry(tmp, path, attempts=6):
    """os.replace, but tolerant of Windows holding the target open.

    Renaming over a file needs DELETE access to it, and CPython opens files
    without FILE_SHARE_DELETE. So while the web routes read timeline.json for
    the card strip — every 2.5s during a render — os.replace fails with
    WinError 5 and the dispatcher died mid-claim, before the sampler ever ran.
    The lock in this module only covers our own threads; it cannot cover the
    HTTP handler, or a virus scanner opening a file we just wrote.

    A few fast retries clear the usual microsecond overlap. Past that we write
    in place, which needs only WRITE access — something those same readers do
    permit — so it succeeds where the rename cannot. Non-atomic, but a timeline
    that saves unatomically beats a render that crashes on the way to the GPU,
    and the whole write is one buffered call under our own lock.
    """
    last = None
    for attempt in range(attempts):
        try:
            os.replace(tmp, path)
            return
        except OSError as ex:
            if not isinstance(ex, PermissionError) \
                    and getattr(ex, "winerror", None) not in (5, 32):
                raise
            last = ex
            time.sleep(0.01 * (2 ** attempt))  # 10ms .. 320ms, ~0.6s total
    try:
        with open(tmp, "r", encoding="utf-8") as src:
            blob = src.read()
        with open(path, "w", encoding="utf-8") as dst:
            dst.write(blob)
        os.remove(tmp)
        return
    except Exception:
        raise last


def save(path, timeline):
    timeline["updated"] = _now()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=1, ensure_ascii=False)
        _replace_with_retry(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


# --------------------------------------------------------------------------
# state machine
# --------------------------------------------------------------------------

def find(timeline, seg_id):
    for seg in timeline["segments"]:
        if seg["id"] == seg_id:
            return seg
    return None


def is_outstanding(seg, pass_name):
    if seg.get("state") == LOCKED:
        return False
    if seg.get("state") == RUNNING:
        return False
    return not seg.get("clips", {}).get(pass_name)


def sweep_stale(timeline, stale_seconds):
    """Return segments stuck in RUNNING past the timeout to PENDING."""
    if stale_seconds <= 0:
        return []
    now = time.time()
    revived = []
    for seg in timeline["segments"]:
        if seg.get("state") != RUNNING:
            continue
        claimed = seg.get("claimed_at") or 0
        if now - float(claimed) > stale_seconds:
            seg["state"] = PENDING
            seg["claimed_at"] = None
            seg["last_error"] = "run did not complete; reclaimed after timeout"
            revived.append(seg["id"])
    return revived


def select(timeline, pass_name, mode="next_pending", index=0,
           range_start=1, range_end=0):
    """Pick the segment this run should render, or None."""
    segments = timeline["segments"]

    if mode == "manual_index":
        i = int(index) - 1
        if i < 0 or i >= len(segments):
            return None, "manual_index %d is outside 1..%d" % (index, len(segments))
        seg = segments[i]
        if seg.get("state") == LOCKED:
            return None, "segment %s is locked" % seg["id"]
        return seg, "manual"

    if mode == "failed_only":
        for seg in segments:
            if seg.get("state") == FAILED:
                return seg, "retry failed"
        return None, "no failed segments"

    if mode == "range":
        a = max(1, int(range_start)) - 1
        b = len(segments) if int(range_end) <= 0 else min(len(segments), int(range_end))
        for seg in segments[a:b]:
            if is_outstanding(seg, pass_name):
                return seg, "range"
        return None, "range %d..%d complete for pass %s" % (a + 1, b, pass_name)

    for seg in segments:  # next_pending
        if is_outstanding(seg, pass_name):
            return seg, "next pending"

    # Be precise about why nothing was claimable. "All done" and "everything is
    # wedged on running" are the same silence from outside, and only one of
    # them means the render finished.
    c = counts(timeline, pass_name)
    if c["running"]:
        return None, ("%d segment(s) still marked running from an earlier run"
                      % c["running"])
    if c["failed"]:
        return None, "%d segment(s) failed and none are pending" % c["failed"]
    if c["locked"] and c["done"] + c["locked"] == c["total"]:
        return None, "the rest are locked"
    return None, "every segment has a %s clip" % pass_name


def claim(path, seg_id, pass_name, force=False, render_frames=None,
          render_duration=None):
    """Mark a segment running and persist, so the next queued run skips it.

    Re-reads from disk and writes back only this segment's runtime fields.
    Saving a caller's in-memory copy instead would silently roll back whatever
    a previous run completed while that copy was being held.
    """
    lock = _lock_for(path)
    with lock:
        timeline = load(path)
        if timeline is None:
            raise RuntimeError("no timeline at %s — run the H3 Timeline node "
                               "before dispatching" % path)
        seg = find(timeline, seg_id)
        if seg is None:
            raise RuntimeError("segment %s is not in the timeline" % seg_id)
        if force:
            seg.setdefault("clips", {}).pop(pass_name, None)
            seg["take"] = int(seg.get("take", 0)) + 1
        seg["state"] = RUNNING
        seg["claimed_at"] = time.time()
        seg["last_error"] = ""
        if render_frames is not None:
            seg["render_frames"] = int(render_frames)
        if render_duration is not None:
            seg["render_duration"] = float(render_duration)
        save(path, timeline)
        return timeline, seg


def complete(path, seg_id, pass_name, clip, frames=None, duration=None):
    """Record a finished render. Re-reads from disk so concurrent runs merge."""
    lock = _lock_for(path)
    with lock:
        timeline = load(path)
        if timeline is None:
            raise RuntimeError("timeline vanished at %s" % path)
        seg = find(timeline, seg_id)
        if seg is None:
            raise RuntimeError("segment %s is not in the timeline" % seg_id)
        # How long this segment actually took, measured rather than modelled.
        # The dispatcher projects the remaining time from these, so the
        # estimate reflects this machine and this resolution instead of an
        # assumption about either.
        claimed = seg.get("claimed_at")
        if claimed:
            elapsed = max(0.0, time.time() - float(claimed))
            if elapsed < 24 * 3600:       # ignore a claim left over from a
                seg["render_elapsed"] = round(elapsed, 1)   # previous session
        seg.setdefault("clips", {})[pass_name] = clip
        seg["state"] = pass_name
        # In one_go both branches complete in an order ComfyUI decides, so
        # pick the better of the two rather than whichever landed last.
        seg["chosen"] = ("final" if seg["clips"].get("final") else pass_name)
        seg["claimed_at"] = None
        seg["last_error"] = ""
        if frames is not None:
            seg["render_frames"] = int(frames)
        if duration is not None:
            seg["render_duration"] = float(duration)
        save(path, timeline)
        return timeline, seg


def fail(path, seg_id, message):
    lock = _lock_for(path)
    with lock:
        timeline = load(path)
        if timeline is None:
            return None
        seg = find(timeline, seg_id)
        if seg is None:
            return timeline
        seg["state"] = FAILED
        seg["claimed_at"] = None
        seg["last_error"] = str(message)[:500]
        save(path, timeline)
        return timeline


def eta(timeline, pass_name):
    """Seconds still to render, projected from what this machine actually did.

    Scaled by frame count, because a 10-second clip costs roughly twice a
    5-second one and a timeline mixes lengths. Returns (seconds, samples) —
    ``samples`` is how many finished renders the estimate rests on, so a caller
    can decline to show a number drawn from one lucky run.
    """
    rates = []
    for seg in timeline["segments"]:
        elapsed = seg.get("render_elapsed")
        frames = seg.get("render_frames")
        if elapsed and frames:
            rates.append(float(elapsed) / float(frames))
    if not rates:
        return None, 0
    rates.sort()
    per_frame = rates[len(rates) // 2]          # median resists one bad run
    remaining = 0.0
    for seg in timeline["segments"]:
        if is_outstanding(seg, pass_name):
            remaining += per_frame * float(seg.get("render_frames") or 0)
    return remaining, len(rates)


def format_duration(seconds):
    seconds = int(round(seconds))
    if seconds < 90:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm" % round(seconds / 60.0)
    return "%dh %02dm" % (seconds // 3600, (seconds % 3600) // 60)


def counts(timeline, pass_name):
    """Every segment bucketed, so a stall can name itself."""
    out = {"done": 0, "pending": 0, "running": 0, "failed": 0, "locked": 0}
    for seg in timeline["segments"]:
        if seg.get("clips", {}).get(pass_name):
            out["done"] += 1
        elif seg.get("state") == RUNNING:
            out["running"] += 1
        elif seg.get("state") == FAILED:
            out["failed"] += 1
        elif seg.get("state") == LOCKED:
            out["locked"] += 1
        else:
            out["pending"] += 1
    out["total"] = len(timeline["segments"])
    return out


def unstick(timeline):
    """Force every RUNNING segment back to PENDING."""
    freed = []
    for seg in timeline["segments"]:
        if seg.get("state") == RUNNING:
            seg["state"] = PENDING
            seg["claimed_at"] = None
            freed.append(seg["id"])
    return freed


def progress(timeline, pass_name):
    total = len(timeline["segments"])
    done = sum(1 for s in timeline["segments"]
               if s.get("clips", {}).get(pass_name))
    failed = sum(1 for s in timeline["segments"] if s.get("state") == FAILED)
    return done, total, failed


def summary(timeline, pass_name):
    done, total, failed = progress(timeline, pass_name)
    lines = ["%s — %d/%d %s clips%s"
             % (timeline.get("name", "timeline"), done, total, pass_name,
                (", %d failed" % failed) if failed else "")]
    for seg in timeline["segments"]:
        mark = "*" if seg.get("clips", {}).get(pass_name) else " "
        rd = seg.get("render_duration")
        lines.append(
            "%s %-8s %-10s target %6.3fs  render %s  seed %d  %s"
            % (mark, seg["id"], seg.get("state", "?"),
               seg["target_duration"],
               ("%6.3fs" % rd) if rd else "   —   ",
               seg["seed"], seg.get("beat", "")))
    return "\n".join(lines)
