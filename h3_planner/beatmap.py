"""Find the musical grid of a track, and cut a timeline on it.

A music video is judged on whether the picture changes when the music does. The
planner's even beats cannot know that, and a treatment's timestamps are only as
good as whoever typed them, so this reads the boundaries out of the audio.

Nothing here decides what a shot contains — only where the cuts fall.

The one thing that makes it usable: musical time is not renderable time. Two
bars at 92.3 BPM is 5.201s, and H3 renders 5.167s or 5.875s, nothing between.
So a cut's *musical* length goes in ``target_duration`` and the rendered length
is snapped separately; the stitcher already trims every clip back to its
planned length, so the cut lands on the beat to the millisecond and the ladder
overshoot is discarded. Sync is exact and does not drift down a long track.

Pure functions over numpy. librosa is imported lazily so the pack still loads
without it.
"""

import math

# One bar of 4/4 is the unit a listener actually feels. Two bars is the usual
# shot length in a music video; one is frantic, four is a held moment.
DEFAULT_BEATS_PER_BAR = 4


def _mono(audio, target_sr=22050):
    """ComfyUI AUDIO -> (mono float32 numpy, sample rate)."""
    import numpy as np

    waveform = audio["waveform"]
    arr = (waveform.cpu().numpy() if hasattr(waveform, "cpu")
           else np.asarray(waveform))
    while arr.ndim > 2:          # (batch, channels, samples)
        arr = arr[0]
    if arr.ndim == 2:            # (channels, samples)
        arr = arr.mean(axis=0)
    arr = arr.astype(np.float32, copy=False)
    sr = int(audio.get("sample_rate") or target_sr)
    if sr != target_sr and arr.size:
        import librosa
        arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return arr, sr


def analyse(audio, beats_per_bar=DEFAULT_BEATS_PER_BAR, target_sr=22050):
    """Tempo, beat times, bar lines and the track's biggest energy rises."""
    import librosa
    import numpy as np

    y, sr = _mono(audio, target_sr)
    if y.size < sr:
        raise RuntimeError("the audio is too short to find a tempo in")

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beats = np.asarray(beats, dtype=float)
    if beats.size < 4:
        raise RuntimeError(
            "no steady beat was found in this audio. It may be speech, or too "
            "quiet — plan the cuts from the treatment instead.")
    tempo = float(np.atleast_1d(tempo)[0])
    beat_gap = float(np.median(np.diff(beats)))

    # Which beat of the bar carries the kick. librosa gives beats but not
    # downbeats, and a bar line half a bar out puts every cut off the one.
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    onset_t = librosa.frames_to_time(np.arange(len(onset)), sr=sr)
    strength = np.interp(beats, onset_t, onset)
    per_bar = max(1, int(beats_per_bar))
    scores = [float(strength[p::per_bar].mean()) if strength[p::per_bar].size
              else 0.0 for p in range(per_bar)]
    phase = int(np.argmax(scores)) if scores else 0
    bars = [float(b) for b in beats[phase::per_bar]]

    duration = float(len(y)) / sr
    return {
        "tempo": round(tempo, 2),
        "beat_seconds": round(beat_gap, 4),
        "bar_seconds": round(beat_gap * per_bar, 4),
        "beats_per_bar": per_bar,
        "phase": phase,
        "phase_scores": [round(s, 3) for s in scores],
        "beats": [round(float(b), 3) for b in beats],
        "bars": [round(b, 3) for b in bars],
        "duration": round(duration, 3),
        "drops": _drops(y, sr, bars, beat_gap * per_bar, duration),
    }


def _drops(y, sr, bars, bar_seconds, duration):
    """Bar lines where sustained energy jumps — a chorus or a drop.

    Measured over two bars either side rather than instantaneously: a snare
    hit is not a drop, and a naive difference of smoothed RMS just finds the
    moment the intro starts.
    """
    import librosa
    import numpy as np

    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    if not len(rms):
        return []
    span = max(0.5, bar_seconds * 2)

    def mean_between(a, b):
        mask = (times >= a) & (times < b)
        return float(rms[mask].mean()) if mask.any() else 0.0

    found = []
    for bar in bars:
        if bar < span or bar > duration - span * 0.5:
            continue
        before = mean_between(bar - span, bar)
        after = mean_between(bar, bar + span)
        if before > 1e-6 and after / before >= 1.25:
            found.append({"at": round(bar, 3), "lift": round(after / before, 2)})
    found.sort(key=lambda d: -d["lift"])
    return found[:8]


# A bar can miss a window by a millisecond — two bars at 92.3 BPM is
# 5.201s against a 5.2s ceiling — and refusing that is pedantry, not
# accuracy. The render snaps to the frame ladder regardless.
WINDOW_TOLERANCE = 0.05


def group_size(bar_seconds, low, high, prefer=0, tolerance=None):
    """How many bars make a clip inside the allowed length window.

    Returns 0 when no whole number of bars fits, which is a real answer: at
    150 BPM one bar is 1.6s and four are 6.4s, so a 5-6s window contains no
    whole number of bars at all.
    """
    slack = WINDOW_TOLERANCE if tolerance is None else float(tolerance)
    if prefer and low - slack <= bar_seconds * prefer <= high + slack:
        return int(prefer)
    best, best_gap = 0, None
    middle = (float(low) + float(high)) / 2.0
    for count in (1, 2, 3, 4, 6, 8, 12, 16):
        span = bar_seconds * count
        if span < low - slack or span > high + slack:
            continue
        gap = abs(span - middle)
        if best_gap is None or gap < best_gap:
            best, best_gap = count, gap
    return best


# A clip shorter than this is not worth a card of its own: it renders to a
# handful of frames and reads as a glitch rather than a shot.
MIN_PARTIAL = 1.0


def plan_cuts(analysis, low, high, prefer_bars=0, snap_to_drops=True,
              start=0.0, end=None, cover=True):
    """Bar-aligned spans covering the track, each inside [low, high].

    Every boundary is a real bar line, so a cut always lands on the one.

    The first downbeat is almost never at 0.000 — on a 92 BPM rap it was at
    2.554s — and the grid can stop short of the end. With ``cover`` on, the
    lead-in and the tail are covered anyway, because the alternative is
    silence: those seconds of song never render, and any shot the treatment
    placed in them is dropped without appearing anywhere at all.
    """
    begin_at = float(start)
    bars = [b for b in analysis["bars"] if b >= begin_at - 1e-6]
    finish = float(end) if end else analysis["duration"]
    bar_seconds = analysis["bar_seconds"]
    per_clip = group_size(bar_seconds, low, high, prefer_bars)
    if not per_clip:
        raise RuntimeError(
            "no whole number of bars fits between %.2fs and %.2fs at %.1f BPM "
            "(one bar is %.3fs). Widen the segment length window."
            % (low, high, analysis["tempo"], bar_seconds))

    drops = {d["at"] for d in analysis["drops"]} if snap_to_drops else set()
    cuts, missed, leftover, i = [], [], [], 0

    head = (bars[0] - begin_at) if bars else 0.0
    if cover and head > 0.05:
        # Absorb the lead-in into the opening clip by taking FEWER bars, so
        # everything after it still starts on a bar line. A 2.554s lead-in in
        # front of a 3-bar clip is 10.38s, over a 10s ceiling; in front of a
        # 2-bar clip it is 7.77s, which fits and keeps the grid intact.
        for k in range(per_clip, -1, -1):
            if k >= len(bars):
                continue
            span = bars[k] - begin_at
            if low - WINDOW_TOLERANCE <= span <= high + WINDOW_TOLERANCE:
                cuts.append({"start": round(begin_at, 3),
                             "end": round(bars[k], 3),
                             "duration": round(span, 3), "bars": k,
                             "drop": False, "partial": "pickup"})
                i = k
                break
        else:
            # No grouping fits around it. Give it a card of its own, short as
            # it is — the alternative is the opening of the song not existing.
            cuts.append({"start": round(begin_at, 3),
                         "end": round(bars[0], 3),
                         "duration": round(head, 3), "bars": 0,
                         "drop": False, "partial": "pickup"})

    while i < len(bars) - 1:
        begin = bars[i]
        if begin >= finish - 0.05:
            break
        step = per_clip
        # A drop deserves its own cut. Shorten the clip before it so the next
        # one starts exactly on the lift rather than a bar or two late.
        for ahead in range(1, per_clip):
            candidate = bars[i + ahead] if i + ahead < len(bars) else None
            if candidate is not None and candidate in drops:
                if bar_seconds * ahead >= low - WINDOW_TOLERANCE:
                    step = ahead
                else:
                    # Honouring it would need a clip under the minimum
                    # length, so the drop passes mid-shot. Worth saying.
                    missed.append(round(candidate, 3))
                break
        stop = bars[i + step] if i + step < len(bars) else finish
        stop = min(stop, finish)
        length = stop - begin
        if length < low - WINDOW_TOLERANCE:
            break               # the tail is handled once, below
        cuts.append({
            "start": round(begin, 3),
            "end": round(stop, 3),
            "duration": round(length, 3),
            "bars": step,
            "drop": any(abs(begin - d) < 1e-3 for d in drops),
        })
        i += step

    if cuts:
        # A lead-in of a few frames is rounding, not a pickup: swallow it.
        slip = cuts[0]["start"] - begin_at
        if 1e-6 < slip <= 0.05:
            cuts[0]["start"] = round(begin_at, 3)
            cuts[0]["duration"] = round(cuts[0]["end"] - begin_at, 3)

        tail = finish - cuts[-1]["end"]
        if tail > 0.05:
            merged = cuts[-1]["duration"] + tail
            edge = cuts[-1]["end"]
            if merged <= high + WINDOW_TOLERANCE:
                # Merging blindly once made a final clip of 10.4s under a 10s
                # ceiling, which is a length the card cannot render.
                cuts[-1]["end"] = round(finish, 3)
                cuts[-1]["duration"] = round(merged, 3)
            elif cover and tail >= MIN_PARTIAL:
                last = {"start": edge, "end": round(finish, 3),
                        "duration": round(tail, 3),
                        "bars": int(tail // bar_seconds), "drop": False}
                if tail < low - WINDOW_TOLERANCE:
                    last["partial"] = "tail"
                cuts.append(last)
            else:
                leftover.append(round(tail, 3))

    return cuts, sorted(set(missed)), leftover


def label(cut, index):
    partial = cut.get("partial")
    if partial == "pickup":
        return ("pickup + %d bar(s)" % cut["bars"]) if cut["bars"] else "pickup"
    if partial == "tail":
        return "tail"
    if cut.get("drop"):
        return "DROP - %d bar(s)" % cut["bars"]
    return "%d bar(s) from %.2fs" % (cut["bars"], cut["start"])


def as_segments(cuts):
    """Cut list -> authored segments, ready for normalize_timeline.

    ``target_duration`` is the EXACT musical span and ``audio_start`` the exact
    position in the track. The renderer snaps to the frame ladder separately
    and the stitcher trims back, so the picture change lands on the beat.
    """
    return [{
        "id": "seg_%02d" % (i + 1),
        "beat": label(cut, i),
        "target_duration": cut["duration"],
        "audio_start": cut["start"],
        "link": "cut",
        "prompt": "",
    } for i, cut in enumerate(cuts)]


def lyric_lines(audio, model_size="small", language=None, max_gap=0.55):
    """Where each sung or rapped line ends, from the vocal.

    Whisper gives word times; a line is a run of words with no long gap. This
    is the "when does the rhyming line end" question — in rap a line is one bar,
    so the bar grid usually answers it more reliably than the transcript. Use
    this to confirm the grid, not to replace it.
    """
    import tempfile
    import os
    import numpy as np
    import soundfile as sf
    from faster_whisper import WhisperModel

    y, sr = _mono(audio, 16000)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, y, sr)
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(path, word_timestamps=True,
                                       language=language)
        words = [w for s in segments for w in (s.words or [])]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    lines, current = [], []
    for word in words:
        if current and word.start - current[-1].end > max_gap:
            lines.append(current)
            current = []
        current.append(word)
    if current:
        lines.append(current)
    return [{"start": round(float(ln[0].start), 3),
             "end": round(float(ln[-1].end), 3),
             "text": "".join(w.word for w in ln).strip()} for ln in lines]
