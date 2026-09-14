"""ffmpeg plumbing: encode a clip, read it back, trim-and-concat a timeline.

Everything the stitcher does happens on *files*. A three-minute 1080p timeline
held as IMAGE tensors is tens of gigabytes of RAM, which would undo the entire
low-VRAM design, so clips move through this module as paths.
"""

import os
import re
import shutil
import subprocess
import tempfile
import wave

import numpy as np

_FFMPEG = None


def ffmpeg_bin():
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    for probe in ("ffmpeg", "ffmpeg.exe"):
        found = shutil.which(probe)
        if found:
            _FFMPEG = found
            return _FFMPEG
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        return _FFMPEG
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install it and put it on PATH, or "
        "`pip install imageio-ffmpeg` into the ComfyUI environment.")


def _stderr_file():
    """A real file for ffmpeg's stderr instead of a pipe.

    A pipe has a fixed OS buffer of a few tens of KB. Whenever we also hold
    ffmpeg's stdin open we cannot drain that buffer while feeding frames in, so
    ffmpeg blocks writing a warning, stops reading stdin, and the two processes
    wait on each other forever — a silent hang that leaves a header-only mp4 on
    disk. A file has no such limit and needs no reader thread to stay empty.
    """
    fd, path = tempfile.mkstemp(suffix=".ffmpeg.log")
    return os.fdopen(fd, "w+b"), path


def _read_log(handle, path):
    try:
        handle.seek(0)
        return handle.read().decode(errors="ignore")
    except Exception:
        return ""
    finally:
        try:
            handle.close()
        except Exception:
            pass
        try:
            os.remove(path)
        except OSError:
            pass


def _wait(proc, timeout, what, handle, path):
    """Wait on ffmpeg, but never forever. Returns (exit_code, stderr_text)."""
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        err = _read_log(handle, path)
        raise RuntimeError(
            "ffmpeg hung %s and was killed after %.0fs. Last output:%s%s"
            % (what, timeout, os.linesep, err[-1500:] or "(nothing)"))
    return code, _read_log(handle, path)


def _run(cmd, what, timeout=1800.0):
    handle, log = _stderr_file()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=handle)
    code, err = _wait(proc, timeout, what, handle, log)
    if code != 0:
        raise RuntimeError("ffmpeg failed %s:%s%s"
                           % (what, os.linesep, err[-1500:]))


# --------------------------------------------------------------------------
# tensors <-> files
# --------------------------------------------------------------------------

def _to_uint8(images):
    frames = images.cpu().numpy() if hasattr(images, "cpu") else np.asarray(images)
    frames = (np.clip(frames, 0.0, 1.0) * 255.0).astype(np.uint8)
    if frames.ndim != 4:
        raise ValueError("expected IMAGE batch (N,H,W,C), got %r" % (frames.shape,))
    if frames.shape[3] == 4:
        frames = frames[:, :, :, :3]
    return frames


def save_wav(audio, path):
    waveform = audio["waveform"]
    waveform = waveform.cpu().float().numpy() if hasattr(waveform, "cpu") \
        else np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(pcm.shape[0])
        wf.setsampwidth(2)
        wf.setframerate(int(audio["sample_rate"]))
        wf.writeframes(pcm.T.reshape(-1).tobytes())
    return path


def read_wav(path):
    import torch
    with wave.open(path, "rb") as wf:
        channels, sr = wf.getnchannels(), wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    pcm = np.frombuffer(raw, dtype="<i2").reshape(-1, channels).T
    waveform = torch.from_numpy(pcm.astype(np.float32) / 32767.0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sr}


def trim_args(start, end):
    """ffmpeg input arguments for a [start, end) window, in seconds.

    ``-ss`` goes BEFORE ``-i`` so ffmpeg seeks instead of decoding and throwing
    away, and the length is given as ``-t`` rather than ``-to`` because after a
    seek ``-to`` is measured from the seek point and silently means something
    else.
    """
    args = []
    start = max(0.0, float(start or 0.0))
    if start:
        args += ["-ss", "%.3f" % start]
    if end:
        span = float(end) - start
        if span <= 0:
            raise ValueError(
                "the trim ends at %.3fs but starts at %.3fs" % (float(end), start))
        args += ["-t", "%.3f" % span]
    return args


def load_audio_file(path, sample_rate=44100, start=0.0, end=None):
    """Decode any audio (or a video's audio track) into ComfyUI's AUDIO dict.

    Goes through ffmpeg rather than torchaudio so the pack keeps its promise of
    no pip dependencies, and so mp3/m4a work the same as wav.
    """
    import torch
    cmd = [ffmpeg_bin()] + trim_args(start, end) + ["-i", path, "-vn",
           "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", "2", "-ar", str(int(sample_rate)), "-"]
    handle, log = _stderr_file()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=handle)
    raw = proc.stdout.read()
    code, err = _wait(proc, 900.0, "decoding audio from %s" % path, handle, log)
    if code != 0 or not raw:
        raise RuntimeError("could not decode audio from %s:%s%s"
                           % (path, os.linesep, err[-600:]))
    samples = np.frombuffer(raw, dtype="<f4")
    samples = samples[:len(samples) // 2 * 2].reshape(-1, 2).T.copy()
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def encode_clip(images, fps, out_path, wav_path=None, crf=17):
    """Write an IMAGE batch (+ optional wav) to h264 mp4. Returns (w, h, n)."""
    frames = _to_uint8(images)
    n, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    even_w, even_h = width - (width % 2), height - (height % 2)
    if (even_w, even_h) != (width, height):
        frames = frames[:, :even_h, :even_w, :]
        width, height = even_w, even_h

    cmd = [ffmpeg_bin(), "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-"]
    if wav_path:
        cmd += ["-i", wav_path]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-crf", str(int(crf)), "-movflags", "+faststart"]
    if wav_path:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [out_path]

    # stderr goes to a file, never a pipe. We hold stdin open for the whole
    # feed — 141 frames of 416x736 is 129 MB — and cannot read a stderr pipe
    # at the same time, so a pipe here deadlocks the moment ffmpeg emits more
    # warnings than the buffer holds. That is what left a 48-byte mp4 with an
    # empty mdat box in the vault and hung the queue after the first segment.
    handle, log = _stderr_file()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=handle)
    broken = False
    try:
        for i in range(frames.shape[0]):
            proc.stdin.write(frames[i].tobytes())
    except (BrokenPipeError, OSError):
        # ffmpeg died mid-feed. Its own stderr says why, so fall through and
        # report that instead of this write error.
        broken = True
    try:
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        broken = True

    code, err = _wait(proc, max(180.0, n * 2.0),
                      "storing clip %s" % out_path, handle, log)
    if code != 0 or broken:
        raise RuntimeError("ffmpeg failed storing clip (exit %s):%s%s"
                           % (code, os.linesep, err[-1500:] or "(no output)"))

    # An mp4 this small is a bare header with no moov atom: the file exists but
    # holds no video. Unchecked, it is filed in the vault as a good take and
    # only surfaces much later as a card that will not play.
    size = file_size(out_path)
    if size < 1024:
        try:
            os.remove(out_path)  # never leave a decoy take in the vault
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg wrote only %d bytes to %s — the clip has no video track.%s%s"
            % (size, out_path, os.linesep, err[-1500:] or "(no output)"))
    return width, height, n


def decode_clip(path, width, height):
    import torch
    cmd = [ffmpeg_bin(), "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    handle, log = _stderr_file()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=handle)
    raw = proc.stdout.read()
    _wait(proc, 900.0, "decoding %s" % path, handle, log)
    stride = width * height * 3
    count = len(raw) // stride
    if count == 0:
        raise RuntimeError("could not decode stored clip: %s" % path)
    arr = np.frombuffer(raw[:count * stride], dtype=np.uint8)
    arr = arr.reshape(count, height, width, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def probe_size(path):
    """(width, height) of a video file, read off ffmpeg's own banner.

    ffprobe is not guaranteed to sit beside ffmpeg in every ComfyUI install, so
    this parses the line ffmpeg already prints when it opens the file:

        Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661),
        yuv420p(progressive), 832x1504, 8008 kb/s, 24 fps, ...

    The two-digit minimum on each side is what keeps the codec tag
    ``0x31637661`` from reading as a resolution.
    """
    handle, log = _stderr_file()
    proc = subprocess.Popen([ffmpeg_bin(), "-i", path],
                            stdout=subprocess.DEVNULL, stderr=handle)
    # ffmpeg exits non-zero when given no output file, which is exactly what
    # this call does, so the banner is the result and the exit code is noise.
    # _wait already drains and deletes the log, so read it from there rather
    # than calling _read_log again on a handle it has closed.
    _code, text = _wait(proc, 60.0, "probing %s" % path, handle, log)
    for line in text.splitlines():
        if ": Video:" not in line:
            continue
        found = re.search(r"[,\s](\d{2,5})x(\d{2,5})[\s,]", line)
        if found:
            return int(found.group(1)), int(found.group(2))
    raise RuntimeError(
        "no video stream found in %s. ffmpeg said:\n%s"
        % (os.path.basename(path), text[-400:]))


# A reference video is conditioning, not footage: H3 wants a couple of seconds
# and downscales anyway. Decoding a 30s 1080p file whole is 5 GB of float
# tensor, which is a hang rather than a reference.
REF_VIDEO_SECONDS = 15.0
REF_VIDEO_SHORT_EDGE = 512


def decode_video(path, max_seconds=REF_VIDEO_SECONDS,
                 short_edge=REF_VIDEO_SHORT_EDGE, start=0.0, end=None):
    """A reference video as IMAGE frames, capped in length and in size."""
    width, height = probe_size(path)
    scale = min(1.0, float(short_edge) / max(1, min(width, height)))
    out_w = max(2, int(round(width * scale)) // 2 * 2)
    out_h = max(2, int(round(height * scale)) // 2 * 2)
    # The cap is a backstop, not the trim: an untrimmed card still must not
    # decode a four-minute file into memory.
    span = (float(end) - max(0.0, float(start or 0.0))) if end else None
    keep = min(float(max_seconds), span) if span else float(max_seconds)
    cmd = ([ffmpeg_bin()] + trim_args(start, None)
           + ["-t", "%.3f" % keep, "-i", path,
              "-vf", "scale=%d:%d" % (out_w, out_h),
              "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    handle, log = _stderr_file()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=handle)
    raw = proc.stdout.read()
    _wait(proc, 900.0, "decoding reference video %s" % path, handle, log)
    return frames_from_raw(raw, out_w, out_h, path)


def frames_from_raw(raw, width, height, path):
    """Raw rgb24 bytes to a float IMAGE tensor.

    ``.copy()`` because ``frombuffer`` hands back a read-only view of the pipe
    buffer, and a node downstream that writes into its input then fails deep in
    torch with nothing naming this function.
    """
    import torch
    stride = width * height * 3
    count = len(raw) // stride
    if count == 0:
        raise RuntimeError("could not decode %s" % path)
    arr = np.frombuffer(raw[:count * stride], dtype=np.uint8)
    arr = arr.reshape(count, height, width, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr.copy())


def write_png(frame_hwc, path):
    """Save one float HWC frame as PNG (used for last-frame chaining)."""
    from PIL import Image
    arr = frame_hwc.cpu().numpy() if hasattr(frame_hwc, "cpu") \
        else np.asarray(frame_hwc)
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(arr).save(path)
    return path


def write_thumbnail(images, path, max_width=384, quality=80):
    from PIL import Image
    frames = _to_uint8(images)
    mid = frames[frames.shape[0] // 2]
    img = Image.fromarray(mid)
    if img.width > max_width:
        h = max(1, int(round(img.height * max_width / float(img.width))))
        img = img.resize((max_width, h), Image.LANCZOS)
    img.save(path, quality=quality)
    return path


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------

def concat_trimmed(clips, out_path, fps, width, height,
                   sample_rate=48000, crf=18, include_audio=True):
    """Trim each clip to its planned duration and join them.

    ``clips`` is a list of dicts: {path, duration, has_audio}. The trim is what
    makes the 17-frame ladder invisible — every segment is cut back to the
    length the plan asked for, so nothing drifts against the audio.

    One ffmpeg pass, frame accurate. Hard cuts only in this version.
    """
    if not clips:
        raise RuntimeError("nothing to stitch — no clips selected")

    want_audio = include_audio and any(c.get("has_audio") for c in clips)
    cmd = [ffmpeg_bin(), "-y"]
    for clip in clips:
        cmd += ["-i", clip["path"]]

    silence_index = None
    if want_audio and not all(c.get("has_audio") for c in clips):
        silence_index = len(clips)
        cmd += ["-f", "lavfi", "-t", "1",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=%d" % sample_rate]

    parts, labels = [], []
    for i, clip in enumerate(clips):
        dur = float(clip["duration"])
        parts.append(
            "[%d:v]trim=end=%.6f,setpts=PTS-STARTPTS,"
            "scale=%d:%d:force_original_aspect_ratio=decrease,"
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=%s[v%d]"
            % (i, dur, width, height, width, height, fps, i))
        labels.append("[v%d]" % i)
        if not want_audio:
            continue
        if clip.get("has_audio"):
            parts.append(
                "[%d:a]atrim=end=%.6f,asetpts=PTS-STARTPTS,"
                "aresample=%d,aformat=channel_layouts=stereo[a%d]"
                % (i, dur, sample_rate, i))
        else:
            parts.append(
                "[%d:a]atrim=end=%.6f,asetpts=PTS-STARTPTS,"
                "aformat=channel_layouts=stereo[a%d]"
                % (silence_index, dur, i))
        labels.append("[a%d]" % i)

    parts.append("%sconcat=n=%d:v=1:a=%d[vout]%s"
                 % ("".join(labels), len(clips), 1 if want_audio else 0,
                    "[aout]" if want_audio else ""))

    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if want_audio:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium",
            "-crf", str(int(crf)), "-r", str(fps), "-movflags", "+faststart",
            out_path]

    _run(cmd, "stitching %d clips" % len(clips))
    return out_path


def contact_sheet(image_paths, out_path, columns=4, tile_width=320):
    """Grid of segment thumbnails, for a quick look at the whole timeline."""
    from PIL import Image
    if not image_paths:
        return None
    tiles = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        h = max(1, int(round(img.height * tile_width / float(img.width))))
        tiles.append(img.resize((tile_width, h), Image.LANCZOS))
    if not tiles:
        return None
    rows = (len(tiles) + columns - 1) // columns
    cell_h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (columns * tile_width, rows * cell_h), (16, 16, 16))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * tile_width, (i // columns) * cell_h))
    sheet.save(out_path, quality=85)
    return out_path


def file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def mux_track(video_path, audio, out_path, start_seconds=0.0,
              sample_rate=48000):
    """Lay one continuous audio track over a finished video.

    For a music video this is what you want in the end: every clip was
    generated against its own window of the song, and each was trimmed back to
    the exact length that window occupied, so the original track drops straight
    back on top and stays in sync for the whole runtime — at full quality,
    rather than 21 separately generated audio fragments butted together.
    """
    fd, wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav(audio, wav)
        cmd = [ffmpeg_bin(), "-y", "-i", video_path]
        if start_seconds:
            cmd += ["-ss", "%.6f" % float(start_seconds)]
        cmd += ["-i", wav,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-ar", str(int(sample_rate)), "-shortest",
                "-movflags", "+faststart", out_path]
        _run(cmd, "laying the track over the stitched video")
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    return out_path
