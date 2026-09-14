"""TimelineVault — clips addressed by (segment, pass, take).

The Clip Vault in Aijigyasa_nodes stores a flat, chronological list and hands
back "the newest". A timeline needs the opposite: ask for segment 7's final
take and get exactly that, whatever was rendered since. Same on-disk shape —
mp4 plus optional wav, manifest alongside, everything surviving a restart —
with the addressing the timeline needs added.

Promotion never overwrites: a draft and a final for the same segment are
separate entries, so a promote is reversible.
"""

import json
import os
import tempfile
import time

from . import media, paths, store as _store

VERSION = 1
MANIFEST = "vault.json"


def _manifest_path(project):
    return os.path.join(paths.vault_dir(project), MANIFEST)


def load_manifest(project):
    try:
        with open(_manifest_path(project), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data.get("clips"), list):
            return data
    except FileNotFoundError:
        pass
    except Exception as ex:
        print("[H3Planner] vault manifest unreadable: %s" % ex)
    return {"version": VERSION, "clips": []}


def save_manifest(project, manifest):
    path = _manifest_path(project)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
        # Same Windows rename hazard as the timeline, and this one lands at the
        # very end of Vault Write: the clip is already encoded and on disk, so
        # failing here loses a finished render and leaves the segment running.
        _store._replace_with_retry(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def _stem(seg_id, pass_name, take):
    return "%s__%s__t%d" % (seg_id, pass_name, int(take))


def store(project, seg_id, pass_name, take, images, fps,
          audio=None, spec_hash="", seed=None):
    """Encode and register one clip. Returns the manifest entry."""
    vdir = paths.vault_dir(project)
    stem = _stem(seg_id, pass_name, take)
    mp4 = stem + ".mp4"
    wav = None

    if audio is not None:
        try:
            wav = stem + ".wav"
            media.save_wav(audio, os.path.join(vdir, wav))
        except Exception as ex:
            print("[H3Planner] audio not stored for %s: %s" % (stem, ex))
            wav = None

    width, height, frames = media.encode_clip(
        images, fps, os.path.join(vdir, mp4),
        wav_path=os.path.join(vdir, wav) if wav else None)

    thumb = stem + ".jpg"
    try:
        media.write_thumbnail(images, os.path.join(vdir, thumb))
    except Exception as ex:
        print("[H3Planner] thumbnail skipped for %s: %s" % (stem, ex))
        thumb = None

    entry = {
        "segment_id": seg_id,
        "pass": pass_name,
        "take": int(take),
        "file": mp4,
        "wav": wav,
        "thumb": thumb,
        "has_audio": bool(wav),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": float(fps),
        "duration": frames / float(fps),
        "seed": seed,
        "spec_hash": spec_hash,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": media.file_size(os.path.join(vdir, mp4)),
    }

    manifest = load_manifest(project)
    manifest["clips"] = [c for c in manifest["clips"]
                         if not (c.get("segment_id") == seg_id
                                 and c.get("pass") == pass_name
                                 and int(c.get("take", 0)) == int(take))]
    manifest["clips"].append(entry)
    save_manifest(project, manifest)
    return entry


def get(project, seg_id, pass_name, take=None):
    """Newest matching clip, or the exact take when given."""
    best = None
    for clip in load_manifest(project)["clips"]:
        if clip.get("segment_id") != seg_id or clip.get("pass") != pass_name:
            continue
        if take is not None and int(clip.get("take", 0)) != int(take):
            continue
        if best is None or int(clip.get("take", 0)) >= int(best.get("take", 0)):
            best = clip
    return best


def resolve(project, seg, prefer):
    """The clip that should represent this segment.

    ``prefer`` is "final", "draft", or "chosen" — chosen falls back through
    final then draft, so a half-promoted timeline still stitches.
    """
    order = {"final": ["final", "draft"],
             "draft": ["draft", "final"]}.get(
                 prefer, [seg.get("chosen") or "final", "final", "draft"])
    for pass_name in order:
        if not pass_name:
            continue
        clip = get(project, seg["id"], pass_name)
        if clip and os.path.exists(os.path.join(paths.vault_dir(project),
                                                clip["file"])):
            return clip
    return None


def abs_path(project, clip):
    return os.path.join(paths.vault_dir(project), clip["file"])


def thumb_path(project, clip):
    if not clip or not clip.get("thumb"):
        return None
    p = os.path.join(paths.vault_dir(project), clip["thumb"])
    return p if os.path.exists(p) else None


def prune(project, keep_takes=2):
    """Drop all but the newest ``keep_takes`` takes per (segment, pass)."""
    manifest = load_manifest(project)
    groups = {}
    for clip in manifest["clips"]:
        groups.setdefault((clip.get("segment_id"), clip.get("pass")),
                          []).append(clip)

    vdir = paths.vault_dir(project)
    kept, removed = [], 0
    for clips in groups.values():
        clips.sort(key=lambda c: int(c.get("take", 0)), reverse=True)
        kept.extend(clips[:keep_takes])
        for dead in clips[keep_takes:]:
            for key in ("file", "wav", "thumb"):
                if dead.get(key):
                    try:
                        os.remove(os.path.join(vdir, dead[key]))
                    except OSError:
                        pass
            removed += 1

    manifest["clips"] = kept
    save_manifest(project, manifest)
    return removed
