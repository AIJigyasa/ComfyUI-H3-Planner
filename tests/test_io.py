"""The two failures that stalled a live render, pinned so they cannot return.

Both were invisible in unit tests because both are about *concurrency and
buffering*, not logic — the code was correct and still hung. Run from the pack
root:

    python tests/test_io.py

Needs a real ffmpeg. Skips the encode half cleanly when there is none.
"""

import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from h3_planner import media, store, vault  # noqa: E402

FAILED = []


def check(label, got, want):
    if got == want:
        print("  ok   %s" % label)
    else:
        FAILED.append(label)
        print("  FAIL %s\n       got  %r\n       want %r" % (label, got, want))


def ok(label, condition, detail=""):
    check(label + (" (%s)" % detail if detail else ""), bool(condition), True)


tmp = tempfile.mkdtemp(prefix="h3io_")
try:
    # ----------------------------------------------------------------------
    print("\nsaving a timeline while something else holds it open")
    # Renaming over a file needs DELETE access, which CPython's open() does not
    # share. The card strip polls timeline.json every 2.5s during a render, so
    # os.replace hit WinError 5 and killed the dispatcher mid-claim.
    path = os.path.join(tmp, "proj", "timeline.json")
    timeline = {"version": 1, "project": "t", "segments": [
        {"id": "seg_01", "index": 0, "target_duration": 5.0, "prompt": "x",
         "seed": 1, "refine_seed": 2, "state": "pending", "clips": {}}]}
    store.save(path, timeline)

    holder = open(path, "r", encoding="utf-8")
    try:
        store.save(path, timeline)
        ok("save survives a reader holding the file open", True)
    except Exception as ex:
        ok("save survives a reader holding the file open", False, str(ex))
    finally:
        holder.close()
    ok("the timeline is still readable after that", store.load(path) is not None)

    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    f.read()
            except Exception:
                pass
            time.sleep(0.001)

    threads = [threading.Thread(target=poll, daemon=True) for _ in range(6)]
    for t in threads:
        t.start()
    errors = []
    started = time.time()
    for _ in range(40):
        try:
            store.claim(path, "seg_01", "draft", force=True,
                        render_frames=141, render_duration=5.875)
        except Exception as ex:
            errors.append(str(ex))
    elapsed = time.time() - started
    stop.set()
    check("40 claims under 6 concurrent readers, no failures", errors, [])
    ok("and they stayed quick", elapsed < 20.0, "%.2fs" % elapsed)
    live = store.load(path)
    ok("the timeline survived intact", live is not None)
    check("every claim landed", live["segments"][0]["take"], 40)

    # the vault manifest lands at the very end of Vault Write, so a rename
    # failure there loses an already-encoded clip
    vpath = os.path.join(tmp, "vproj", "vault.json")
    os.makedirs(os.path.dirname(vpath), exist_ok=True)
    store.save(vpath, {"version": 1, "segments": [], "clips": []})
    keeper = open(vpath, "r", encoding="utf-8")
    try:
        fd, scratch = tempfile.mkstemp(dir=os.path.dirname(vpath), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write('{"version": 1, "clips": []}')
        store._replace_with_retry(scratch, vpath)
        ok("the vault manifest writes through a held handle too", True)
    except Exception as ex:
        ok("the vault manifest writes through a held handle too", False, str(ex))
    finally:
        keeper.close()

    # ----------------------------------------------------------------------
    print("\nencoding a clip without deadlocking on ffmpeg's stderr")
    try:
        media.ffmpeg_bin()
        have_ffmpeg = True
    except Exception as ex:
        have_ffmpeg = False
        print("  skip  no ffmpeg on PATH (%s)" % ex)

    if have_ffmpeg:
        # The exact shape that hung: 141 frames at 416x736 is 129 MB pushed
        # into ffmpeg's stdin. With stderr on a pipe we could not drain it
        # while feeding, so ffmpeg blocked on a warning, stopped reading
        # stdin, and both sides waited forever — leaving a 48-byte mp4 whose
        # mdat box was empty and no moov atom at all.
        n, height, width, fps = 141, 736, 416, 24
        frames = np.random.default_rng(0).random(
            (n, height, width, 3), dtype=np.float32)

        wav = os.path.join(tmp, "a.wav")
        sr = 44100
        tone = (np.sin(np.arange(int(sr * n / float(fps))) * 0.05)
                * 0.3).astype(np.float32)
        media.save_wav({"waveform": np.stack([tone, tone])[None, ...],
                        "sample_rate": sr}, wav)

        out = os.path.join(tmp, "clip.mp4")
        started = time.time()
        w, h, count = media.encode_clip(frames, fps, out, wav_path=wav)
        elapsed = time.time() - started
        check("encoded at the planned size", (w, h, count), (width, height, n))
        ok("it did not hang", elapsed < 120.0, "%.1fs" % elapsed)
        size = media.file_size(out)
        ok("the file holds real video", size > 100000, "%.1f MB" % (size / 1e6))
        with open(out, "rb") as f:
            blob = f.read()
        ok("the mp4 was finalised (moov atom present)", b"moov" in blob)

        back = media.decode_clip(out, w, h)
        ok("it reads back", back.shape[0] >= n - 2, "%d frames" % back.shape[0])

        # A truncated encode must raise rather than be filed as a good take.
        broken = os.path.join(tmp, "broken.mp4")
        try:
            media.encode_clip(frames[:2], fps, broken,
                              wav_path=os.path.join(tmp, "missing.wav"))
            ok("a failed encode raises", False, "it returned quietly")
        except RuntimeError:
            ok("a failed encode raises", True)
        ok("and leaves no decoy clip in the vault", not os.path.exists(broken))

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("\n%s" % ("FAILED: " + ", ".join(FAILED) if FAILED else "all checks passed"))
sys.exit(1 if FAILED else 0)
