"""Setup nodes: Project and Timeline. The Cast Board lives in nodes_cast."""

import json
import os
import re

from . import ladder, paths, resolution, store

CATEGORY = "H3 Planner"

TAG_RE = re.compile(r"<\s*(Subject|Picture|Video|Audio)\s*(\d+)\s*>", re.I)

SECTION_ORDER = ("subject_definitions", "summary", "retention_analysis",
                 "detailed_description", "overall_soundscape",
                 "non_diegetic_music")


def flatten_prompt(prompt, style="full_reference_6"):
    """Six named sections -> the one string MiniMaxH3ReferenceToVideo wants.

    Matches `_render_full_ref` in ComfyUI-H3-Prompt-Creator exactly, so a
    prompt authored by that node and one assembled here are byte-identical.
    """
    if isinstance(prompt, str):
        return prompt.strip()
    if not isinstance(prompt, dict):
        return ""
    if style == "flat":
        return "\n\n".join(str(prompt[k]).strip()
                           for k in prompt if str(prompt[k]).strip())
    if style == "video_3field":
        order = ("integrated_multimodal_description", "overall_soundscape",
                 "non_diegetic_music")
    else:
        order = SECTION_ORDER
    out = []
    for i, key in enumerate(order):
        out.append("%s:" % key)
        out.append(str(prompt.get(key, "")).strip() or "N/A")
        if i < len(order) - 1:
            out.append("")
    return "\n".join(out)


def cited_tags(prompt):
    text = flatten_prompt(prompt)
    found = {}
    for kind, number in TAG_RE.findall(text):
        found.setdefault(kind.capitalize(), set()).add(int(number))
    return found


# --------------------------------------------------------------------------

class H3PlannerProject:
    """Project-wide settings: resolutions, fps, the frame ladder, seeds.

    Every other node in the pack reads its numbers from here, so a change to
    the ladder or the draft resolution takes effect everywhere at once.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "h3_project"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                                  "step": 1.0}),
                "aspect_ratio": (resolution.ASPECTS, {"default": "9:16"}),
                "render_pass": (["draft", "final", "one_go"], {
                    "default": "draft",
                    "tooltip": "draft = base sample only, the upscale branch is skipped. final = base + upscale, for segments you have approved. one_go = both in a single run, storing a draft and a final for every segment."}),
                "draft_megapixels": ("FLOAT", {"default": 0.3, "min": 0.05,
                                               "max": 8.0, "step": 0.05}),
                "final_megapixels": ("FLOAT", {"default": 1.2, "min": 0.05,
                                               "max": 8.0, "step": 0.05}),
                "align": ("INT", {"default": 32, "min": 1, "max": 128}),
                "base_seed": ("INT", {"default": 12345, "min": 0,
                                      "max": 0xFFFFFFFFFFFFFF}),
                "frame_modulus": ("INT", {"default": 17, "min": 1, "max": 256,
                                          "tooltip": "H3 needs length %% modulus == remainder"}),
                "frame_remainder": ("INT", {"default": 5, "min": 0, "max": 255}),
                "frame_minimum": ("INT", {"default": 5, "min": 1, "max": 4096}),
                "snap": (["up", "nearest"], {"default": "up",
                                             "tooltip": "up never renders less than planned; the stitcher trims the overshoot"}),
                "max_segment_seconds": ("FLOAT", {
                    "default": 6.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "the longest single clip your GPU can render in one go. H3 itself tops out near 15s; drop this to 6 on a smaller card and the planner cuts more, shorter segments instead."}),
                "default_duration": ("FLOAT", {"default": 6.0, "min": 0.2,
                                               "max": 60.0, "step": 0.1}),
                "stale_minutes": ("INT", {"default": 10, "min": 0, "max": 1440,
                                          "tooltip": "a run left unfinished this long is reclaimed"}),
            },
        }

    RETURN_TYPES = ("H3_PROJECT", "FLOAT", "STRING")
    RETURN_NAMES = ("project", "upscale_megapixels", "report")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, project_name, fps, aspect_ratio, render_pass,
              draft_megapixels, final_megapixels, align, base_seed,
              frame_modulus, frame_remainder, frame_minimum, snap,
              max_segment_seconds, default_duration, stale_minutes):
        name = paths.safe_name(project_name)
        draft_wh, final_wh, scale = resolution.pass_sizes(
            draft_megapixels, final_megapixels, aspect_ratio, align)

        # The base sampler ALWAYS runs at draft size. A final is not a bigger
        # base render — it is the draft latent put through the upscaler. Sizing
        # the base pass by render_pass, as this used to, meant a "promoted"
        # segment was a different video rather than the one you approved.
        width, height = draft_wh
        megapixels = width * height / 1e6
        upscale_megapixels = final_wh[0] * final_wh[1] / 1e6

        # The pass a run claims: in one_go you are ultimately after the final,
        # and the draft falls out of the same base sample for free.
        claim_pass = "draft" if render_pass == "draft" else "final"

        project = {
            "name": name, "fps": float(fps), "aspect_ratio": aspect_ratio,
            "render_pass": claim_pass, "pass_mode": render_pass,
            "align": int(align),
            "final_width": final_wh[0], "final_height": final_wh[1],
            "upscale_megapixels": float(upscale_megapixels),
            "upscale_scale": scale,
            "draft_megapixels": float(draft_megapixels),
            "final_megapixels": float(final_megapixels),
            "megapixels": float(megapixels),
            "width": width, "height": height,
            "base_seed": int(base_seed),
            "frame_modulus": int(frame_modulus),
            "frame_remainder": int(frame_remainder),
            "frame_minimum": int(frame_minimum),
            "snap": snap,
            "max_seconds": float(max_segment_seconds),
            "default_duration": float(default_duration),
            "stale_seconds": int(stale_minutes) * 60,
            "dir": paths.project_dir(name),
            "timeline_path": paths.timeline_path(name),
        }

        top_frames, top_seconds = ladder.ceiling(
            fps, max_segment_seconds, modulus=frame_modulus, remainder=frame_remainder,
            minimum=frame_minimum)
        grid = ladder.rungs(fps, max_segment_seconds, modulus=frame_modulus,
                            remainder=frame_remainder, minimum=frame_minimum)
        step = (grid[1][1] - grid[0][1]) if len(grid) > 1 else 0.0

        report = "\n".join([
            "project      %s" % name,
            "folder       %s" % project["dir"],
            "mode         %s%s" % (render_pass, {
                "draft": "  — base sample only, upscale branch skipped",
                "final": "  — base sample + upscale, claims segments with no final",
                "one_go": "  — both branches, stores a draft and a final per segment",
            }[render_pass]),
            "base         %dx%d (%.2f MP, %s) — every pass samples at this size"
            % (width, height, megapixels, aspect_ratio),
            "upscale to   %dx%d (%.2f MP, exactly %dx — same aspect, clean for "
            "the latent upscaler)"
            % (final_wh[0], final_wh[1], upscale_megapixels, scale),
            "fps          %g" % fps,
            "ladder       length %% %d == %d, min %d  (%d rungs, %.4fs apart)"
            % (frame_modulus, frame_remainder, frame_minimum, len(grid), step),
            "segment cap  %.1fs asked -> longest usable clip is %d frames = %.3fs"
            % (max_segment_seconds, top_frames, top_seconds),
            "seeds        derived from base %d, fixed at plan time" % base_seed,
        ])
        return (project, float(upscale_megapixels), report)


# --------------------------------------------------------------------------

class H3PlannerTimeline:
    """Author, validate and persist the timeline.

    Re-running with an edited prompt resets only the segments that actually
    changed — everything already rendered keeps its clips. That is what makes
    it safe to keep tweaking a long timeline mid-render.
    """

    EXAMPLE = json.dumps({
        "name": "delhi-rap",
        "segments": [
            {"id": "s1", "beat": "hook", "duration": 3.0,
             "prompt": "subject_definitions:\n<Subject 1> is the rapper from the reference photo...",
             "audio_start": 0.0},
            {"id": "s2", "beat": "verse", "duration": 8.0,
             "prompt": "...", "audio_start": 3.0},
        ],
    }, indent=1)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project": ("H3_PROJECT",),
                "source": (["inline", "saved", "file"], {"default": "inline",
                           "tooltip": "saved = reuse the timeline already on disk for this project, so planning and rendering can live in separate workflows"}),
                "timeline_json": ("STRING", {"multiline": True,
                                             "default": cls.EXAMPLE}),
                "on_change": (["merge", "replace"], {"default": "merge",
                                                     "tooltip": "merge keeps clips for segments whose spec is unchanged"}),
                "reset_failed": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "timeline": ("H3_TIMELINE", {
                    "tooltip": "from the Segment Prompter or Treatment Splitter; takes precedence over the JSON below"}),
                "file_path": ("STRING", {"default": ""}),
                "cast": ("H3_CAST",),
            },
        }

    # segment_count exists to be wired into the Stitcher's min_clips, so the
    # stitch waits for the whole timeline without a number to keep in step by
    # hand. Appended, never inserted: the slot index is what a saved workflow
    # stores, so an existing graph keeps its timeline and report links.
    RETURN_TYPES = ("H3_TIMELINE", "STRING", "INT")
    RETURN_NAMES = ("timeline", "report", "segment_count")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return store.run_token()

    # Every authored field a planner hands over. A whitelist, because runtime
    # state (clips, take, claimed_at) must not become authored truth. It has
    # to list every authored field though: the Story Planner's scene grouping,
    # its opens/ends handover and its spoken lines were all silently dropped
    # here once, so a replan lost the script and the card strip never saw a
    # scene.
    AUTHORED_FIELDS = ("id", "beat", "prompt", "target_duration", "link",
                       "cast_used", "audio_start", "notes", "source",
                       "prompt_fingerprint", "scene", "scene_name",
                       "opens_from", "ends_with", "dialogue", "speaker",
                       "refine_note")

    @staticmethod
    def _previous_authored(timeline_json, names):
        """The card strip's own segments, by id, from the widget.

        Only the widget knows about a refine or a hand edit by the time the
        graph runs: the planner upstream has usually saved its own prompts
        over the timeline on disk a moment earlier. A widget written for a
        different project is ignored rather than leaking its edits across.
        """
        try:
            doc = json.loads(timeline_json or "")
        except (TypeError, ValueError):
            return {}
        if isinstance(doc, list):
            doc = {"segments": doc}
        if not isinstance(doc, dict):
            return {}
        owner = doc.get("name") or doc.get("project")
        if owner and owner not in names:
            return {}
        return {s["id"]: s for s in doc.get("segments") or []
                if isinstance(s, dict) and s.get("id")}

    @classmethod
    def _adopt(cls, upstream, previous):
        """Upstream segments, keeping any prompt the user refined or edited.

        The rule, per segment, compares three prompts: the one the planner
        delivered LAST time (recorded as planned_prompt_hash), the one on the
        card now, and the one the planner delivers now.

        - card unchanged since last delivery -> take the planner's prompt
        - card edited, planner delivered the same as last time -> keep the edit
        - card edited, planner delivered something new -> the planner re-wrote
          this segment on purpose, so its prompt wins and the lost edit is
          named in the report

        Without this, a refined prompt lasted exactly until the next queue:
        the planner in front of the Timeline handed its original prompt back,
        the refine was overwritten, and the segment rendered the old shot
        again.
        """
        segments, kept, replaced = [], [], []
        for seg in upstream:
            row = {k: seg[k] for k in cls.AUTHORED_FIELDS if k in seg}
            delivered = store.prompt_hash(seg.get("prompt"))
            row["planned_prompt_hash"] = delivered
            before = previous.get(seg.get("id"))
            if before is not None and before.get("prompt") not in (None, ""):
                mine = store.prompt_hash(before.get("prompt"))
                recorded = before.get("planned_prompt_hash")
                if mine != delivered:
                    if recorded:
                        edited = mine != recorded
                        planner_moved = delivered != recorded
                    else:
                        # A timeline from before this was recorded. A refine
                        # leaves its note behind; a plain hand edit cannot be
                        # told apart from a stale card, so the planner wins
                        # once and the hash is recorded from here on.
                        edited = bool((before.get("refine_note") or "").strip())
                        planner_moved = False
                    if edited and not planner_moved:
                        row["prompt"] = before["prompt"]
                        if before.get("refine_note"):
                            row["refine_note"] = before["refine_note"]
                        kept.append(row["id"])
                    elif edited:
                        # The note described the prompt that was just lost.
                        row["refine_note"] = ""
                        replaced.append(row["id"])
            segments.append(row)
        return segments, kept, replaced

    def build(self, project, source, timeline_json, on_change, reset_failed,
              file_path="", cast=None, timeline=None):
        kept_edits, lost_edits = [], []
        if timeline is not None:
            # A prompted timeline arrived from upstream. Adopt it and mirror it
            # into the widget, so the card strip shows the generated prompts,
            # but keep any prompt refined or edited on a card since the
            # planner last delivered it.
            previous = self._previous_authored(
                timeline_json, {project["name"], timeline.get("name")})
            segments, kept_edits, lost_edits = self._adopt(
                timeline["segments"], previous)
            authored = {
                "name": timeline.get("name", project["name"]),
                "context": timeline.get("context", {}),
                "segments": segments,
            }
            raw = json.dumps(authored, indent=1, ensure_ascii=False)
        elif source == "saved":
            existing_doc = store.load(project["timeline_path"])
            if existing_doc is None:
                # Name the projects that DO have a timeline. This nearly always
                # means project_name here does not match the one the planning
                # workflow ran under, and going to look for that by hand is a
                # miserable way to find out.
                known = paths.planned_projects()
                if known:
                    hint = ("Projects that already have one: %s. Set "
                            "project_name on H3 Project to match the planning "
                            "workflow." % ", ".join(repr(k) for k in known))
                else:
                    hint = ("No project has a timeline yet — run "
                            "workflow_plan.json first, with the same "
                            "project_name.")
                raise ValueError(
                    "H3 Timeline source is 'saved', but project %r has no "
                    "timeline at %s. %s"
                    % (project["name"], project["timeline_path"], hint))
            raw = json.dumps({
                "name": existing_doc.get("name", project["name"]),
                "context": existing_doc.get("context", {}),
                "segments": existing_doc["segments"],
            }, ensure_ascii=False)
        elif source == "file":
            path = file_path.strip()
            if not path:
                raise ValueError("source is 'file' but file_path is empty")
            if not os.path.isabs(path):
                path = os.path.join(project["dir"], path)
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
        else:
            raw = timeline_json

        try:
            authored = json.loads(raw)
        except json.JSONDecodeError as ex:
            raise ValueError("timeline JSON is invalid: %s (line %d, col %d)"
                             % (ex.msg, ex.lineno, ex.colno))

        fresh = store.normalize_timeline(authored, project)

        # ladder: plan in target_duration, render the rung at or above it
        warnings = []
        for seg in fresh["segments"]:
            frames = ladder.snap_frames(
                seg["target_duration"], project["fps"],
                modulus=project["frame_modulus"],
                remainder=project["frame_remainder"],
                minimum=project["frame_minimum"], direction=project["snap"])
            seg["render_frames"] = frames
            seg["render_duration"] = frames / float(project["fps"])
            if seg["render_duration"] > project["max_seconds"] + 1e-6:
                warnings.append(
                    "%s: %.3fs needs %d frames = %.3fs, past the %.1fs segment cap"
                    % (seg["id"], seg["target_duration"], frames,
                       seg["render_duration"], project["max_seconds"]))

        if cast:
            size = cast.get("size", 0)
            for seg in fresh["segments"]:
                for kind, numbers in cited_tags(seg["prompt"]).items():
                    # Pictures only: subjects live inside pictures and a
                    # two-picture cast can hold five of them.
                    if kind == "Picture":
                        over = sorted(n for n in numbers if n > size)
                        if over:
                            cites = ", ".join("<%s %d>" % (kind, n) for n in over)
                            warnings.append(
                                "%s cites %s but the cast has %d image slot(s)"
                                % (seg["id"], cites, size))

        for seg in fresh["segments"]:
            if not str(seg["prompt"]).strip():
                warnings.append("%s has an empty prompt" % seg["id"])

        tl_path = project["timeline_path"]
        existing = store.load(tl_path)
        timeline = fresh if on_change == "replace" else store.merge(existing, fresh)

        # Rendered clips thrown away because the prompt changed under them.
        # This is the loop that looks like a caching bug: a planner left inside
        # the render graph with reuse_existing off rewrites every prompt on
        # every queue, so each clip is discarded the moment after it is made
        # and the dispatcher renders the same segment again, forever. Naming it
        # is the whole fix — the behaviour itself is correct.
        discarded = timeline.pop("_discarded", [])
        if discarded:
            note = ("DISCARDED %d rendered clip(s) — their prompts changed "
                    "since they were rendered: %s.\n"
                    "If you did not edit them, something upstream is rewriting "
                    "the timeline on every queue. A planner wired into the "
                    "render graph with reuse_existing OFF will do exactly "
                    "this, and the same segment will render forever. Turn "
                    "reuse_existing ON, or keep planning in its own workflow."
                    % (len(discarded),
                       ", ".join("%s (%s)" % (d["id"], "+".join(d["passes"]))
                                 for d in discarded[:6])))
            print("[H3Planner] %s" % note.replace("\n", " "))
            warnings.append(note)

        if reset_failed:
            for seg in timeline["segments"]:
                if seg.get("state") == store.FAILED:
                    seg["state"] = store.PENDING
                    seg["last_error"] = ""

        store.save(tl_path, timeline)

        target_total = sum(s["target_duration"] for s in timeline["segments"])
        render_total = sum(s["render_duration"] or 0.0
                           for s in timeline["segments"])
        carried = sum(1 for s in timeline["segments"] if s.get("clips"))

        edits = []
        if kept_edits:
            edits.append("kept your edited prompt on %s: the planner "
                         "upstream delivered the same prompt as before, so "
                         "the edit stands and renders"
                         % ", ".join(kept_edits))
        if lost_edits:
            warnings.append(
                "REPLACED your edited prompt on %s: the planner re-wrote %s, "
                "so its new prompt is used. Refine again if you still want "
                "the change."
                % (", ".join(lost_edits),
                   "that segment" if len(lost_edits) == 1 else "those segments"))

        report = "\n".join([
            store.summary(timeline, project["render_pass"]),
            "",
            "target %.3fs over %d segments; H3 renders %.3fs, stitcher trims "
            "back %.3fs" % (target_total, len(timeline["segments"]),
                            render_total, render_total - target_total),
            "%d segment(s) kept existing clips" % carried,
        ] + edits + [
            "saved to %s" % tl_path,
        ] + (["", "WARNINGS", "! " + "\n! ".join(warnings)] if warnings else []))

        # Hand the authored JSON back to the card strip. Without this the
        # cards stay empty whenever the timeline arrives through the input
        # wire, because they read the widget and nothing had written to it.
        return {
            # The project name travels with the payload. The card strip used
            # to guess it by walking the graph back to H3 Project, and when
            # that failed the server fell back to the first project on disk —
            # so the cards quietly showed a DIFFERENT project's clips.
            # width/height so the card strip can shape its cards like the
            # video: a 9:16 clip in a fixed landscape card showed a cropped
            # slice of the shot in a box twice as wide as it needed to be.
            "ui": {"h3_timeline": [{"authored": raw,
                                    "project": project["name"],
                                    "pass": project["render_pass"],
                                    "width": int(project["width"]),
                                    "height": int(project["height"]),
                                    "from_input": timeline is not None}]},
            "result": (timeline, report, len(timeline["segments"])),
        }


NODE_CLASS_MAPPINGS = {
    "H3PlannerProject": H3PlannerProject,
    "H3PlannerTimeline": H3PlannerTimeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3PlannerProject": "H3 Project",
    "H3PlannerTimeline": "H3 Timeline",
}
