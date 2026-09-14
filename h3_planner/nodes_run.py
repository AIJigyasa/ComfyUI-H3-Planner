"""Runtime nodes: Dispatcher, Track Slice, Vault Write, Stitcher."""

import json
import os
import time
import urllib.request

from . import ladder, media, paths, store, vault
from .nodes_plan import CATEGORY, flatten_prompt

try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:  # older ComfyUI
    ExecutionBlocker = None

try:
    # ComfyUI's native VIDEO object. It wraps a path and streams from it, so
    # handing one downstream costs nothing — decoding a 50s render into IMAGE
    # frames would be 13 GB of tensor before anything else happened.
    from comfy_api.input_impl import VideoFromFile
except Exception:  # older ComfyUI, or a build without the video API
    VideoFromFile = None


def _blocked(count, message):
    """Skip the sampler branch cleanly when there is nothing to render.

    Returns a UI payload as well as the blockers. A blocked run finishes in
    0.05s with no error and no output, which from the canvas is indomitably
    identical to nothing happening at all — the reason has to be visible
    somewhere other than the console.
    """
    if ExecutionBlocker is None:
        raise RuntimeError(message)
    print("[H3Planner] %s" % message.replace("\n", " | "))
    return {"ui": {"text": [message]},
            "result": tuple(ExecutionBlocker(None) for _ in range(count))}


# --------------------------------------------------------------------------

class H3PlannerDispatcher:
    """Hand the sampler exactly one segment per queue run.

    The cursor is a state machine, not a counter. This node claims the first
    segment that has no clip for the current pass and marks it running; Vault
    Write is what completes it. So queueing forty runs against a twelve-segment
    timeline renders twelve and no-ops the rest, an interrupted render resumes
    where it stopped, and nothing is ever rendered twice.
    """

    # Ten, and every one is meant to be wired. Anything that was only
    # informational (megapixels, pass, audio_start, render_seconds) moved into
    # `info` or travels inside `shot`.
    OUTPUTS = ("H3_SHOT", "STRING", "INT", "INT", "INT", "FLOAT",
               "INT", "INT", "IMAGE", "STRING")
    NAMES = ("shot", "prompt", "length", "width", "height", "fps",
             "seed", "refine_seed", "first_frame", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "timeline": ("H3_TIMELINE",),
                "mode": (["next_pending", "manual_index", "range",
                          "failed_only"], {"default": "next_pending"}),
                "manual_index": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "range_start": ("INT", {"default": 1, "min": 1, "max": 9999}),
                "range_end": ("INT", {"default": 0, "min": 0, "max": 9999,
                                      "tooltip": "0 = to the end"}),
                "force_rerender": ("BOOLEAN", {"default": False,
                                               "tooltip": "discard the stored clip for this pass and shoot another take"}),
                "prompt_format": (["full_reference_6", "video_3field", "flat"],
                                  {"default": "full_reference_6"}),
                "reset_stuck": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "force every segment stuck on 'running' back to pending. A cancelled or crashed run leaves one behind, and enough of those stall the queue."}),
            },
            "optional": {"cast": ("H3_CAST",)},
        }

    RETURN_TYPES = OUTPUTS
    RETURN_NAMES = NAMES
    FUNCTION = "dispatch"
    CATEGORY = CATEGORY
    # An output node so it still runs — and still reports — when everything
    # downstream of it is blocked.
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return store.run_token()  # must re-run every queue to advance the cursor

    def dispatch(self, project, timeline, mode, manual_index, range_start,
                 range_end, force_rerender, prompt_format, reset_stuck=False,
                 cast=None):
        n = len(self.OUTPUTS)
        tl_path = project["timeline_path"]
        pass_name = project["render_pass"]

        # Always work from disk: earlier runs in this queue have moved it on.
        live = store.load(tl_path)
        if live is None:
            live = timeline
            store.save(tl_path, live)
        revived = (store.unstick(live) if reset_stuck
                   else store.sweep_stale(live, project.get("stale_seconds", 600)))
        if revived:
            print("[H3Planner] returned to pending: %s" % ", ".join(revived))
            store.save(tl_path, live)

        seg, why = store.select(live, pass_name, mode, manual_index,
                                range_start, range_end)
        if seg is None:
            # Say exactly why. "Nothing to render" while segments sit on
            # `running` looks identical to "all done" from the outside, and
            # that is what a stalled queue feels like.
            c = store.counts(live, pass_name)
            detail = ", ".join("%d %s" % (c[k], k) for k in
                               ("done", "pending", "running", "failed", "locked")
                               if c[k])
            if c["running"]:
                hint = ("%d segment(s) are still marked running from an earlier "
                        "run. Tick reset_stuck on this node to free them."
                        % c["running"])
            elif c["failed"]:
                hint = "Set mode to failed_only to retry them."
            elif c["done"] == c["total"] and pass_name == "draft":
                hint = ("This pass is finished. Switch H3 Project render_pass "
                        "to 'final' to upscale them, or hit Redo on a card to "
                        "reshoot one.")
            elif c["done"] == c["total"]:
                hint = ("This pass is finished — stitch it, or hit Redo on a "
                        "card to reshoot one.")
            elif c["locked"]:
                hint = "The rest are locked. Unlock a card to render it."
            else:
                hint = ""

            message = "\n".join(filter(None, [
                "Nothing to render.",
                "%s: %s of %d" % (pass_name, detail, c["total"]),
                why,
                hint,
            ]))
            return _blocked(n, message)

        force = force_rerender or (mode == "manual_index"
                                   and bool(seg.get("clips", {}).get(pass_name)))

        frames = ladder.snap_frames(
            seg["target_duration"], project["fps"],
            modulus=project["frame_modulus"],
            remainder=project["frame_remainder"],
            minimum=project["frame_minimum"], direction=project["snap"])
        render_seconds = frames / float(project["fps"])

        live, seg = store.claim(tl_path, seg["id"], pass_name, force=force,
                                render_frames=frames,
                                render_duration=render_seconds)

        prompt = flatten_prompt(seg["prompt"], prompt_format)

        first_frame = None
        if seg.get("first_frame_path"):
            fp = seg["first_frame_path"]
            if not os.path.isabs(fp):
                fp = os.path.join(paths.frames_dir(project["name"]), fp)
            try:
                import numpy as np
                import torch
                from PIL import Image
                arr = np.asarray(Image.open(fp).convert("RGB"),
                                 dtype=np.float32) / 255.0
                first_frame = torch.from_numpy(arr).unsqueeze(0)
            except Exception as ex:
                print("[H3Planner] first frame unreadable (%s): %s" % (fp, ex))

        shot = {
            "project": project["name"],
            "timeline_path": tl_path,
            "segment_id": seg["id"],
            "index": seg["index"],
            "pass": pass_name,
            "take": int(seg.get("take", 0)),
            "seed": seg["seed"],
            "refine_seed": seg["refine_seed"],
            "fps": float(project["fps"]),
            "target_duration": seg["target_duration"],
            "render_frames": frames,
            "render_duration": render_seconds,
            "pass_mode": project.get("pass_mode", pass_name),
            "audio_start": seg.get("audio_start"),
            "link": seg.get("link", "cut"),
            "spec_hash": seg.get("spec_hash", ""),
        }

        done, total, _ = store.progress(live, pass_name)
        overshoot = render_seconds - seg["target_duration"]
        info = "\n".join([
            "segment  %s (%d of %d)  beat %s" % (seg["id"], seg["index"] + 1,
                                                 total, seg.get("beat") or "-"),
            "pass     %s (mode %s) take %d   base %dx%d @ %.2f MP -> %.2f MP"
            % (pass_name, project.get("pass_mode", pass_name), shot["take"],
               project["width"], project["height"], project["megapixels"],
               project.get("upscale_megapixels", project["megapixels"])),
            "length   %d frames = %.3fs  (planned %.3fs, stitcher trims %.3fs)"
            % (frames, render_seconds, seg["target_duration"], overshoot),
            "seeds    %d / %d" % (seg["seed"], seg["refine_seed"]),
            "audio    %s" % ("from %.3fs" % seg["audio_start"]
                             if seg.get("audio_start") is not None else "none"),
            "progress %d/%d %s clips before this run  (%s)"
            % (done, total, pass_name, why),
            "remaining%s" % self._eta_line(live, pass_name),
        ])
        print("[H3Planner] rendering %s (%s pass, %d frames)"
              % (seg["id"], pass_name, frames))

        return {"ui": {"text": [info]},
                "result": (shot, prompt, frames, project["width"],
                           project["height"], float(project["fps"]),
                           int(seg["seed"]), int(seg["refine_seed"]),
                           first_frame, info)}

    @staticmethod
    def _eta_line(timeline, pass_name):
        """What is left, from measured rate — including the run starting now."""
        seconds, samples = store.eta(timeline, pass_name)
        if not seconds:
            return "  (timing after the first segment finishes)"
        return "  ~%s of rendering left%s" % (
            store.format_duration(seconds),
            "" if samples > 1 else " (from one segment so far)")


# --------------------------------------------------------------------------

class H3PlannerPassGate:
    """Decide which half of the sampling graph runs.

    ComfyUI cannot bypass nodes from Python, but it can refuse to execute a
    branch: anything downstream of an ExecutionBlocker is skipped entirely, so
    a blocked upscale branch costs no VRAM and no time.

    Sits straight after the AV latent is split. The draft branch decodes the
    base latent as it is; the final branch feeds the latent upscaler and the
    second sampler. Each branch gets its own shot token with the right pass
    stamped on it, so the two Vault Write nodes file into the right slots with
    nothing to configure.
    """

    OUTPUTS = ("LATENT", "H3_SHOT", "LATENT", "H3_SHOT", "STRING")
    NAMES = ("draft_latent", "draft_shot", "final_latent", "final_shot", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("H3_SHOT",),
                "latent": ("LATENT", {
                    "tooltip": "the base video latent, straight off LTXVSeparateAVLatent"}),
            },
        }

    RETURN_TYPES = OUTPUTS
    RETURN_NAMES = NAMES
    FUNCTION = "gate"
    CATEGORY = CATEGORY

    def gate(self, shot, latent):
        mode = shot.get("pass_mode") or shot.get("pass") or "draft"
        draft_live = mode in ("draft", "one_go")
        final_live = mode in ("final", "one_go")

        draft_shot = dict(shot)
        draft_shot["pass"] = "draft"
        final_shot = dict(shot)
        final_shot["pass"] = "final"

        info = "\n".join([
            "mode     %s" % mode,
            "draft    %s" % ("decoding the base latent"
                             if draft_live else "skipped"),
            "final    %s" % ("upscaling and re-sampling"
                             if final_live else "skipped"),
            "segment  %s" % shot.get("segment_id", "?"),
        ])
        print("[H3Planner] pass gate: %s (draft=%s final=%s)"
              % (mode, draft_live, final_live))

        def live(value, enabled):
            if enabled:
                return value
            if ExecutionBlocker is None:
                return None
            return ExecutionBlocker(None)

        return (live(latent, draft_live), live(draft_shot, draft_live),
                live(latent, final_live), live(final_shot, final_live), info)


# --------------------------------------------------------------------------

class H3PlannerTrackSlice:
    """The window of the reference track that belongs to this segment.

    Slices ``render_duration`` — not target — so H3 hears audio for every frame
    it renders and the stitcher's trim removes picture and sound together.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "shot": ("H3_SHOT",),
                "pad_short": ("BOOLEAN", {"default": True,
                                          "tooltip": "pad with silence when the track runs out"}),
            },
            "optional": {
                "offset_seconds": ("FLOAT", {"default": 0.0, "min": -600.0,
                                             "max": 600.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "start", "end", "info")
    FUNCTION = "slice_track"
    CATEGORY = CATEGORY

    def slice_track(self, audio, shot, pad_short, offset_seconds=0.0):
        import torch

        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        squeezed = waveform[0] if waveform.dim() == 3 else waveform
        total = squeezed.shape[-1]

        start_s = float(shot.get("audio_start") or 0.0) + float(offset_seconds)
        span_s = float(shot.get("render_duration") or 0.0)
        if span_s <= 0:
            span_s = total / float(sr)

        start = max(0, int(round(start_s * sr)))
        want = int(round(span_s * sr))
        chunk = squeezed[..., start:start + want]

        short = want - chunk.shape[-1]
        note = ""
        if short > 0:
            if pad_short:
                pad = torch.zeros(chunk.shape[0], short, dtype=chunk.dtype,
                                  device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=-1)
                note = "  (padded %.3fs of silence — track ended)" % (short / sr)
            else:
                note = "  (%.3fs short — track ended)" % (short / sr)

        out = {"waveform": chunk.unsqueeze(0), "sample_rate": sr}
        end_s = start_s + chunk.shape[-1] / float(sr)
        info = ("%s: %.3fs -> %.3fs (%.3fs of %.3fs track)%s"
                % (shot.get("segment_id", "?"), start_s, end_s,
                   chunk.shape[-1] / float(sr), total / float(sr), note))
        return (out, start_s, end_s, info)


# --------------------------------------------------------------------------

class H3PlannerVaultWrite:
    """Store the rendered segment and advance the timeline.

    This is the only node that knows a run succeeded, so completion — and
    optional auto-advance — belongs here rather than in the dispatcher.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot": ("H3_SHOT",),
                "images": ("IMAGE",),
                "save_last_frame": ("BOOLEAN", {"default": True,
                                                "tooltip": "write the final frame for the next segment's continue link"}),
                "auto_advance": ("BOOLEAN", {"default": False,
                                             "tooltip": "queue the next segment automatically until the timeline is done"}),
                "max_auto_runs": ("INT", {"default": 24, "min": 1, "max": 999}),
            },
            "optional": {"audio": ("AUDIO",)},
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("status", "clip_path", "images")
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return store.run_token()

    def write(self, shot, images, save_last_frame, auto_advance,
              max_auto_runs, audio=None, prompt=None, extra_pnginfo=None):
        project = shot["project"]
        seg_id = shot["segment_id"]
        pass_name = shot["pass"]
        tl_path = shot["timeline_path"]

        try:
            entry = vault.store(
                project, seg_id, pass_name, shot.get("take", 0), images,
                shot["fps"], audio=audio, spec_hash=shot.get("spec_hash", ""),
                seed=shot.get("seed"))
        except Exception as ex:
            store.fail(tl_path, seg_id, str(ex))
            raise

        timeline, seg = store.complete(
            tl_path, seg_id, pass_name, entry,
            frames=entry["frames"], duration=entry["duration"])

        if save_last_frame:
            try:
                self._chain_last_frame(project, timeline, seg, images, tl_path)
            except Exception as ex:
                print("[H3Planner] last frame not saved: %s" % ex)

        done, total, failed = store.progress(timeline, pass_name)
        clip_path = vault.abs_path(project, entry)

        lines = [
            "stored %s  %s take %d" % (seg_id, pass_name, entry["take"]),
            "%d frames  %dx%d  %.3fs rendered, %.3fs after trim"
            % (entry["frames"], entry["width"], entry["height"],
               entry["duration"], shot.get("target_duration", entry["duration"])),
            "%s  (%.1f MB)" % (clip_path, entry["bytes"] / 1e6),
            "progress %d/%d %s clips%s"
            % (done, total, pass_name, ", %d failed" % failed if failed else ""),
        ]

        if auto_advance:
            lines.append(self._auto_advance(
                timeline, pass_name, prompt, max_auto_runs,
                key=(tl_path, seg_id, pass_name, entry["take"])))
        elif done < total:
            left, samples = store.eta(timeline, pass_name)
            lines.append("queue again for the next segment (%d left%s)"
                         % (total - done,
                            ", ~%s" % store.format_duration(left) if left else ""))
        else:
            lines.append("timeline complete for the %s pass — stitch it" % pass_name)

        status = "\n".join(lines)
        print("[H3Planner] %s" % status.replace("\n", " | "))
        return {"ui": {"text": [status]},
                "result": (status, clip_path, images)}

    # -- helpers ----------------------------------------------------------

    def _chain_last_frame(self, project, timeline, seg, images, tl_path):
        nxt = None
        for candidate in timeline["segments"]:
            if candidate["index"] == seg["index"] + 1:
                nxt = candidate
                break
        if nxt is None or nxt.get("link") != "continue":
            return
        name = "%s_last.png" % seg["id"]
        media.write_png(images[-1],
                        os.path.join(paths.frames_dir(project), name))
        if nxt.get("first_frame_path") != name:
            # first_frame_path is derived, not authored — deliberately outside
            # SPEC_FIELDS, so writing it here never invalidates a stored clip.
            nxt["first_frame_path"] = name
            store.save(tl_path, timeline)
        print("[H3Planner] %s will start from %s" % (nxt["id"], name))

    # Seatbelt against the runaway. If the executor ever serves the dispatcher
    # from cache again, this node re-stores the SAME segment and would queue
    # another run for it, forever. Refusing to advance twice for one segment
    # stops that dead even if the cause comes back.
    _last_advanced = {}

    def _auto_advance(self, timeline, pass_name, prompt_graph, max_auto_runs,
                      key=None):
        outstanding = [s for s in timeline["segments"]
                       if store.is_outstanding(s, pass_name)]
        if not outstanding:
            return "auto-advance: timeline complete, stopping"

        if key is not None:
            path = key[0]
            if self._last_advanced.get(path) == key:
                # This used to blame ComfyUI's cache, which cost an afternoon
                # looking in the wrong place. The usual cause is upstream: a
                # planner in the render graph rewrites the prompts every queue,
                # the timeline discards the clip that no longer matches, and
                # this segment becomes outstanding again seconds after it was
                # rendered.
                return ("auto-advance STOPPED: %s was stored twice in a row, so "
                        "advancing would loop forever. Something is putting it "
                        "back to pending after each render — check the H3 "
                        "Timeline report for DISCARDED clips, which means a "
                        "node upstream is rewriting the prompts on every queue "
                        "(a planner in the render graph with reuse_existing "
                        "OFF does this). Queue manually until it is fixed."
                        % key[1])
            self._last_advanced[path] = key
        if prompt_graph is None:
            return ("auto-advance: unavailable (no PROMPT from the server) — "
                    "queue manually")

        try:
            from server import PromptServer
            server = PromptServer.instance
            running, pending = server.prompt_queue.get_current_queue()
            if pending:
                return ("auto-advance: %d run(s) already queued, leaving them "
                        "to it" % len(pending))
            if len(outstanding) > int(max_auto_runs):
                return ("auto-advance: %d segments outstanding, over the "
                        "max_auto_runs ceiling of %d — queue manually"
                        % (len(outstanding), max_auto_runs))

            port = getattr(server, "port", 8188) or 8188
            body = json.dumps({
                "prompt": prompt_graph,
                "client_id": "h3planner-auto",
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:%d/prompt" % port, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp.read()
            return "auto-advance: queued the next segment (%d left)" % len(outstanding)
        except Exception as ex:
            return "auto-advance failed (%s) — queue manually" % ex


# --------------------------------------------------------------------------

class H3PlannerStitch:
    """Trim every stored clip back to its planned length and join them.

    The trim is the whole point: H3 renders on a 17-frame ladder and always
    overshoots what was planned, so cutting each segment back to
    ``target_duration`` is what keeps a long video in sync with its track
    instead of drifting a fraction of a second per cut.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "timeline": ("H3_TIMELINE",),
                "prefer": (["chosen", "final", "draft"], {"default": "chosen"}),
                "trim_to_plan": ("BOOLEAN", {"default": True,
                                             "tooltip": "off = keep H3's ladder overshoot; sync will drift"}),
                "filename_prefix": ("STRING", {"default": "h3planner"}),
                "min_clips": ("INT", {
                    "default": 2, "min": 0, "max": 999,
                    "tooltip": "wait quietly until this many segments have rendered. Leaving the node in the render graph then stitches a growing preview instead of erroring on the first run. 0 = wait for EVERY segment in the timeline, so it stitches itself the moment the last card is done. With prefer set to draft or final it counts only clips of THAT pass, so a final stitch waits for finals instead of joining drafts."}),
                "contact_sheet": ("BOOLEAN", {"default": True}),
                "audio": (["from clips", "from track", "silent"], {
                    "default": "from clips",
                    "tooltip": "from track lays the original reference audio over the whole video — for a music video that is what keeps it in sync and at full quality"}),
            },
            "optional": {
                "music_track": ("AUDIO", {
                    "tooltip": "the full reference track, straight off the Cast Board"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "FLOAT", "STRING", "VIDEO")
    RETURN_NAMES = ("video_path", "contact_sheet_path", "duration", "report",
                    "video")
    FUNCTION = "stitch"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return store.run_token()

    def stitch(self, project, timeline, prefer, trim_to_plan, filename_prefix,
               min_clips, contact_sheet, audio="from clips",
               music_track=None):
        name = project["name"]
        live = store.load(project["timeline_path"]) or timeline
        # 0 means "all of them", so the node can be left wired into the render
        # graph and stitch itself the moment the last card finishes, with
        # nothing to wire up and no number to keep in step with the timeline.
        want = int(min_clips) or len(live["segments"])

        clips, missing, rows, thumbs, short = [], [], [], [], []
        # vault.resolve falls back to the other pass so a half-promoted
        # timeline still stitches. That fallback is silent, and asking for
        # finals while every clip is a draft would otherwise hand back a draft
        # video called final. Count what actually matched the pass asked for.
        strict = prefer in ("final", "draft")
        matched, borrowed = 0, []
        for seg in live["segments"]:
            entry = vault.resolve(name, seg, prefer)
            if entry is None:
                missing.append(seg["id"])
                rows.append("  %-8s MISSING" % seg["id"])
                continue
            duration = (seg["target_duration"] if trim_to_plan
                        else entry["duration"])
            if trim_to_plan and entry["duration"] < seg["target_duration"] - 0.05:
                # The stored clip is shorter than the plan, so the cut will run
                # short here and everything after it slides early against the
                # track. Almost always means the segment was rendered before
                # its duration was changed.
                short.append("%s: clip is %.3fs but the plan wants %.3fs — "
                             "re-render it (Redo on the card)"
                             % (seg["id"], entry["duration"],
                                seg["target_duration"]))
            if entry["pass"] == prefer:
                matched += 1
            elif strict:
                borrowed.append("%s (%s)" % (seg["id"], entry["pass"]))
            duration = min(duration, entry["duration"])
            clips.append({"path": vault.abs_path(name, entry),
                          "duration": duration,
                          "width": entry.get("width", 0),
                          "height": entry.get("height", 0),
                          "has_audio": entry.get("has_audio", False)})
            thumb = vault.thumb_path(name, entry)
            if thumb:
                thumbs.append(thumb)
            rows.append("  %-8s %-5s take %d  %.3fs rendered -> %.3fs used%s"
                        % (seg["id"], entry["pass"], entry["take"],
                           entry["duration"], duration,
                           "" if entry.get("has_audio") else "  (silent)"))

        # Never error for want of clips. This node usually sits in the render
        # graph, so on the first queue there is nothing to join yet — that is
        # the normal state of things, not a failure.
        have = matched if strict else len(clips)
        if have < max(1, want):
            waiting = "\n".join([
                "waiting — %d of %d segment(s) have a %s clip, need %d to stitch"
                % (have, len(live["segments"]),
                   prefer if strict else "usable", want),
                "still to render: %s"
                % (", ".join(missing + [b.split(" ")[0] for b in borrowed])[:400]
                   or "none"),
                "",
                "Queue the render graph again; this node joins them as they "
                "arrive and re-stitches every run.",
            ])
            print("[H3Planner] stitch %s" % waiting.splitlines()[0])
            # There is no video yet, so whatever is wired to that output must
            # not run. Blocking the branch is silent and correct; handing on a
            # None would make a Save Video node throw on every early queue.
            blank = (ExecutionBlocker(None) if ExecutionBlocker is not None
                     else None)
            return {"ui": {"text": [waiting]},
                    "result": ("", "", 0.0, waiting, blank)}

        # A half-promoted timeline mixes 0.3 MP drafts with 1.2 MP finals, so
        # the canvas is the largest clip present rather than the current pass —
        # never downscale a final to match a draft.
        width = max([c["width"] for c in clips if c["width"]] or [project["width"]])
        height = max([c["height"] for c in clips if c["height"]] or [project["height"]])
        width += width % 2
        height += height % 2
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(paths.renders_dir(name),
                           "%s_%s_%s.mp4" % (paths.safe_name(filename_prefix),
                                             name, stamp))
        use_track = audio == "from track" and music_track is not None
        if audio == "from track" and music_track is None:
            raise RuntimeError(
                "audio is 'from track' but no music_track is connected — wire "
                "the Cast Board's audio output, or switch to 'from clips'.")

        media.concat_trimmed(clips, out, project["fps"], width, height,
                             include_audio=(audio == "from clips"))
        if use_track:
            first = next((s for s in live["segments"]
                          if s.get("audio_start") is not None), None)
            start = float(first["audio_start"]) if first else 0.0
            muxed = out.replace(".mp4", "_track.mp4")
            media.mux_track(out, music_track, muxed, start_seconds=start)
            try:
                os.remove(out)
            except OSError:
                pass
            store._replace_with_retry(muxed, out)

        sheet = None
        if contact_sheet and thumbs:
            sheet = media.contact_sheet(
                thumbs, os.path.join(paths.renders_dir(name),
                                     "%s_%s_sheet.jpg" % (name, stamp)))

        total = sum(c["duration"] for c in clips)
        planned = sum(s["target_duration"] for s in live["segments"])
        report = "\n".join([
            "stitched %d clip(s) -> %.3fs" % (len(clips), total),
            "planned  %.3fs across %d segment(s)%s"
            % (planned, len(live["segments"]),
               "; %d not rendered yet" % len(missing) if missing else ""),
            "audio    %s" % {
                "from clips": "each clip's own generated audio",
                "from track": "the original track, laid over the whole video",
                "silent": "none",
            }[audio],
            "trim     %s" % ("on — each clip cut back to its planned length"
                             if trim_to_plan else "OFF — ladder overshoot kept, "
                             "expect drift against audio"),
            "output   %s (%.1f MB)" % (out, media.file_size(out) / 1e6),
            "sheet    %s" % (sheet or "-"),
            "",
            "segments:",
        ] + rows
            + (["", "OUT OF SYNC", "! " + "\n! ".join(short)] if short else [])
            + (["", "MIXED PASSES",
                "! %d segment(s) have no %s clip, so the take from the other "
                "pass was used instead: %s"
                % (len(borrowed), prefer, ", ".join(borrowed[:12])),
                "! render those on the %s pass, or set prefer to 'chosen' if "
                "a mixed cut is what you want." % prefer]
               if borrowed else []))

        print("[H3Planner] stitched %d clips -> %s" % (len(clips), out))

        # Show the finished video on the node. Returning only a path meant the
        # result was easy to miss entirely — it is sitting in renders/ and
        # nothing on the canvas says so.
        sub, fn = paths.relative_to_output(out)
        ui = {"text": [report],
              "gifs": [{"filename": fn, "subfolder": sub, "type": "output",
                        "format": "video/h264-mp4"}]}
        if sheet:
            ssub, sfn = paths.relative_to_output(sheet)
            ui["images"] = [{"filename": sfn, "subfolder": ssub,
                             "type": "output"}]
        # A VIDEO, not IMAGE frames: it connects straight to Save Video or
        # Preview Video without decoding the whole render into memory.
        video = VideoFromFile(out) if VideoFromFile is not None else None
        return {"ui": ui,
                "result": (out, sheet or "", float(total), report, video)}


NODE_CLASS_MAPPINGS = {
    "H3PlannerDispatcher": H3PlannerDispatcher,
    "H3PlannerPassGate": H3PlannerPassGate,
    "H3PlannerTrackSlice": H3PlannerTrackSlice,
    "H3PlannerVaultWrite": H3PlannerVaultWrite,
    "H3PlannerStitch": H3PlannerStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PlannerDispatcher": "H3 Shot Dispatcher",
    "H3PlannerPassGate": "H3 Pass Gate",
    "H3PlannerTrackSlice": "H3 Track Slice",
    "H3PlannerVaultWrite": "H3 Vault Write",
    "H3PlannerStitch": "H3 Stitch Timeline",
}
