"""H3 Beat Map — cut the timeline where the music changes.

Wire the Cast Board's audio in. It finds the tempo, the bar lines and the
track's biggest energy rises, then lays out segments that begin on a downbeat.

The cuts it produces carry the EXACT musical span in ``target_duration`` and the
exact position in ``audio_start``. H3 renders the nearest frame-ladder length,
which is always a little longer, and the stitcher trims each clip back to its
planned length — so every picture change lands on the one, and the error does
not accumulate down a four-minute track.

It writes no prompts. Feed its ``cuts`` to the H3 Segment Slicer to fill them
from a treatment, or wire the timeline straight in and write them on the cards.
"""

import json

from . import beatmap, ladder, store
from .nodes_plan import CATEGORY


class H3PlannerBeatMap:
    """Audio in, a bar-aligned timeline out. No model, no key, no network."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "audio": ("AUDIO", {
                    "tooltip": "the full track, straight off the Cast Board's audio output"}),
                "min_segment_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.5, "max": 30.0, "step": 0.1}),
                "max_segment_seconds": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "a whole number of bars must fit between these. At 92 BPM a bar is 2.6s, so 5-10s allows 2 or 3 bars; a narrow window may allow none."}),
                "beats_per_bar": ("INT", {
                    "default": 4, "min": 1, "max": 12,
                    "tooltip": "4 for almost everything. 3 for a waltz, 6 for some drill and trap."}),
                "bars_per_clip": ("INT", {
                    "default": 0, "min": 0, "max": 16,
                    "tooltip": "0 = choose the grouping that best fits the length window. Set it to force 2-bar or 4-bar shots."}),
                "cut_on_drops": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "shorten the clip before a big energy rise so the next one starts exactly on it"}),
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "skip an intro"}),
                "end_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "0 = to the end of the track"}),
                "cover_whole_track": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "the first downbeat is rarely at 0.000. On = the lead-in before it and the tail after the last bar are covered too, so no second of the track goes unrendered and no shot is silently dropped. Off = start on the first downbeat and lose whatever comes before it."}),
            },
        }

    RETURN_TYPES = ("H3_TIMELINE", "STRING", "STRING")
    RETURN_NAMES = ("timeline", "cuts", "report")
    FUNCTION = "map_beats"
    CATEGORY = CATEGORY

    def map_beats(self, project, audio, min_segment_seconds,
                  max_segment_seconds, beats_per_bar, bars_per_clip,
                  cut_on_drops, start_seconds, end_seconds,
                  cover_whole_track=True):
        low = min(float(min_segment_seconds), float(max_segment_seconds))
        high = min(float(max_segment_seconds), float(project["max_seconds"]))

        try:
            analysis = beatmap.analyse(audio, beats_per_bar)
        except ImportError as ex:
            # librosa is opt-in on purpose (Manager would install a
            # requirements.txt for every user), so this message is the
            # install instruction and has to name the file that exists.
            raise RuntimeError(
                "the beat map needs librosa, which is not installed in this "
                "ComfyUI environment (%s). It is an optional install: run "
                "`python -m pip install -r requirements-audio.txt` from this "
                "pack's folder, using the Python that runs ComfyUI "
                "(the python.exe inside python_embeded on the Windows portable "
                "build), "
                "then restart. Every other node works without it; to cut "
                "without the music, feed a treatment to the Segment Slicer "
                "instead." % ex)

        cuts, missed, leftover = beatmap.plan_cuts(
            analysis, low, high, prefer_bars=bars_per_clip,
            snap_to_drops=bool(cut_on_drops), start=float(start_seconds),
            end=float(end_seconds) or None, cover=bool(cover_whole_track))
        if not cuts:
            raise RuntimeError(
                "no clips could be laid out. The track is %.1fs at %.1f BPM "
                "with a %.3fs bar; between %.2fs and %.2fs that leaves nothing "
                "to work with."
                % (analysis["duration"], analysis["tempo"],
                   analysis["bar_seconds"], low, high))

        timeline = store.normalize_timeline(
            {"name": project["name"],
             "context": {
                 "tempo": analysis["tempo"],
                 "bar_seconds": analysis["bar_seconds"],
                 "beats_per_bar": analysis["beats_per_bar"],
                 "total_duration": round(cuts[-1]["end"] - cuts[0]["start"], 3),
             },
             "segments": beatmap.as_segments(cuts)}, project)

        rows, drift, short = [], 0.0, 0.0
        for seg, cut in zip(timeline["segments"], cuts):
            frames = ladder.snap_frames(
                seg["target_duration"], project["fps"],
                modulus=project["frame_modulus"],
                remainder=project["frame_remainder"],
                minimum=project["frame_minimum"], direction=project["snap"])
            seg["render_frames"] = frames
            seg["render_duration"] = frames / float(project["fps"])
            # The ladder rounds to the nearest whole frame, so a rung can
            # sit a fraction under the musical span. The stitcher trims to
            # target, and a target longer than the clip would silently
            # keep the whole clip and slide everything after it early.
            if seg["render_duration"] < seg["target_duration"]:
                short += seg["target_duration"] - seg["render_duration"]
                seg["target_duration"] = seg["render_duration"]
            trim = seg["render_duration"] - seg["target_duration"]
            drift += trim
            rows.append(
                "  %-8s %7.3f -> %7.3f  %d bar(s) %6.3fs  renders %3df "
                "(%.3fs, trim %.3fs)%s"
                % (seg["id"], cut["start"], cut["end"], cut["bars"],
                   seg["target_duration"], frames, seg["render_duration"],
                   trim, "  <- DROP" if cut["drop"] else
                   ("  <- %s" % cut["partial"].upper()
                    if cut.get("partial") else "")))

        store.save(project["timeline_path"], timeline)

        covered = cuts[-1]["end"] - cuts[0]["start"]
        # cuts[0] can be the pickup, which is off the grid by design.
        grid = next((c for c in cuts if not c.get("partial")), cuts[0])
        partials = ["%s %.2fs" % (c["partial"], c["duration"])
                    for c in cuts if c.get("partial")]
        report = "\n".join([
            "%d clip(s) on the bar grid — no model, no key, no network"
            % len(cuts),
            "tempo      %.1f BPM, %d/4, bar = %.3fs (downbeat on beat %d of %d)"
            % (analysis["tempo"], analysis["beats_per_bar"],
               analysis["bar_seconds"], analysis["phase"] + 1,
               analysis["beats_per_bar"]),
            "grid       %d beats, %d bars over %.2fs of audio"
            % (len(analysis["beats"]), len(analysis["bars"]),
               analysis["duration"]),
            "clips      %d bar(s) each = %.3fs, inside the %.2f-%.2fs window"
            % (grid["bars"], grid["duration"], low, high),
            "coverage   %s"
            % ("the whole track from %.2fs — %s"
               % (float(start_seconds), ", ".join(partials) or "grid only")
               if cover_whole_track else
               "the bar grid only; anything before the first downbeat at "
               "%.2fs is not rendered" % cuts[0]["start"]),
            "covers     %.2fs from %.2fs (%.2fs of track left over)"
            % (covered, cuts[0]["start"], analysis["duration"] - cuts[-1]["end"]),
            "trim       %.3fs total is cut back by the stitcher, so every "
            "boundary stays on the beat" % drift,
            "rounding   %.3fs lost to whole-frame rounding across %d clip(s)"
            % (short, len(cuts)),
            "drops      %s" % (", ".join("%.2fs (x%.2f)" % (d["at"], d["lift"])
                                         for d in analysis["drops"][:6])
                               or "none found"),
            "",
            "clips:",
        ] + rows
            + (["", "NOTES",
                "- %d drop(s) could not start a clip without going under the "
                "%.2fs minimum, so the music lifts mid-shot: %s"
                % (len(missed), low,
                   ", ".join("%.2fs" % m for m in missed[:6]))]
               if missed else [])
            + (["- %.2fs of track at the end was left uncovered: it is "
                "too short for a clip of its own and merging it would "
                "push the last clip past %.2fs"
                % (sum(leftover), high)] if leftover else [])
            + ["", "Feed `cuts` to the H3 Segment Slicer to fill these from a "
               "treatment, or write the prompts on the cards."])

        return (timeline, json.dumps({"analysis": {
            k: analysis[k] for k in ("tempo", "bar_seconds", "beats_per_bar",
                                     "phase", "duration")},
            "cuts": cuts}, indent=1), report)


NODE_CLASS_MAPPINGS = {"H3PlannerBeatMap": H3PlannerBeatMap}
NODE_DISPLAY_NAME_MAPPINGS = {"H3PlannerBeatMap": "H3 Beat Map"}
