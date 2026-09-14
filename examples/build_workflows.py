"""Rewrite the reference H3 workflow to run through H3 Planner.

Takes `video_minimax_h3_r2v_test.json` (ComfyUI API format) and produces:

  workflow_plan.json    — idea -> treatment -> segments -> prompts -> timeline
  workflow_story.json   - one brief -> one model call -> the whole timeline
  workflow_render.json  — the loop: one segment per queue run, into the vault
  workflow_stitch.json  — the assembly: timeline + vault -> one mp4

API format is used deliberately. It keys inputs by name rather than by widget
order, so nothing here depends on the widget layout of third-party nodes like
ImageResizeKJv2 or DiffusionModelLoaderKJ — those pass through untouched.

    python examples/build_workflows.py <path-to-original.json>
"""

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# New node ids, chosen well clear of the originals.
PROJECT, CAST, TIMELINE, DISPATCH, SLICE, VAULT, STITCH = (
    "300", "301", "302", "303", "304", "305", "306")
PROMPTER, GATE, VAULT_FINAL = "308", "309", "310"
STORY = "311"
RESIZE_2 = "312"
SLICER = "313"

# Dispatcher output indices — must match H3PlannerDispatcher.RETURN_NAMES.
D_SHOT, D_PROMPT, D_LENGTH, D_WIDTH, D_HEIGHT = 0, 1, 2, 3, 4
D_FPS, D_SEED, D_REFINE, D_FIRST, D_INFO = 5, 6, 7, 8, 9

# Original node ids we rely on.
O_RESOLUTION, O_MATH, O_DURATION = "115", "131", "132"
O_H3COND, O_PROMPT_STR, O_LOADIMAGE, O_RESIZE = "136", "138", "139", "143"
O_LOADAUDIO, O_PURGE, O_PROMPT_NODE, O_DISPLAY = "157", "184", "193", "194"
O_NOISE_BASE, O_NOISE_REFINE = "129", "207"
O_DECODE_AUDIO, O_DECODE_VIDEO = "121", "122"
O_SEPARATE, O_UPSCALER, O_DECODE_DRAFT = "220", "221", "218"


# The Cast Board's output slots, by index. Output links bind to an index,
# not a name, so this constant is the only thing keeping the examples
# pointing at audio rather than at image_7.
AUDIO_1 = 9

def load_example_timeline():
    with open(os.path.join(HERE, "timeline_example.json"), encoding="utf-8") as f:
        doc = json.load(f)
    doc.pop("_note", None)
    return json.dumps(doc, indent=1, ensure_ascii=False)


def project_node():
    return {
        "inputs": {
            "project_name": "delhi_rap",
            "fps": 24.0,
            "aspect_ratio": "9:16",
            "render_pass": "draft",
            "draft_megapixels": 0.3,
            "final_megapixels": 1.2,
            "align": 32,
            "base_seed": 12345,
            "frame_modulus": 17,
            "frame_remainder": 5,
            "frame_minimum": 5,
            "snap": "up",
            "max_segment_seconds": 6.0,
            "default_duration": 6.0,
            "stale_minutes": 30,
        },
        "class_type": "H3PlannerProject",
        "_meta": {"title": "H3 Project"},
    }


def cast_node():
    """Self-contained now — references are dropped onto the node itself.

    Starts empty on purpose: drop at least one image on the Cast Board before
    queueing, or ref_image_0 has nothing to resize.
    """
    return {
        "inputs": {"cast_json": json.dumps({"entries": []})},
        "class_type": "H3PlannerCastBoard",
        "_meta": {"title": "H3 Cast Board"},
    }


def timeline_node(timeline_json, with_cast=True, source="inline",
                  from_upstream=None):
    node = {
        "inputs": {
            "project": [PROJECT, 0],
            "source": source,
            "timeline_json": timeline_json,
            "on_change": "merge",
            "reset_failed": True,
            "file_path": "",
        },
        "class_type": "H3PlannerTimeline",
        "_meta": {"title": "H3 Timeline"},
    }
    if with_cast:
        node["inputs"]["cast"] = [CAST, 0]
    if from_upstream:
        node["inputs"]["timeline"] = [from_upstream, 0]
    return node


def plan_nodes():
    """One node: treatment in, per-segment prompts out."""
    return {
        PROMPTER: {
            "inputs": {
                "project": [PROJECT, 0],
                "h3_prompt": [O_PROMPT_NODE, 0],
                "total_seconds": 120.0,
                "segment_seconds": 10.0,
                "shots_per_segment": 1,
                "format": "music video",
                "audio_role": "performed on camera",
                "provider": "Ollama (Local)",
                "ollama_url": "http://127.0.0.1:11434",
                "ollama_model": "qwen3-vl:8b",
                "temperature": 0.25,
                "reuse_existing": True,
                "seed": 0,
                "cast": [CAST, 0],
                "idea": "",
                "style_prefix_override": "",
                "api_key": "",
                "api_model": "",
                "max_output_tokens": 3072,
                "num_ctx": 8192,
                "keep_alive": "10m",
                "request_timeout": 600,
            },
            "class_type": "H3PlannerSegmentPrompter",
            "_meta": {"title": "H3 Segment Prompter"},
        },
    }


def build_render(original, timeline_json):
    g = copy.deepcopy(original)

    # --- out: what H3 Planner replaces ------------------------------------
    # LoadImage goes too: references now live on the Cast Board itself.
    for dead in (O_RESOLUTION, O_MATH, O_DURATION, O_PROMPT_NODE,
                 O_LOADIMAGE, O_LOADAUDIO):
        g.pop(dead, None)

    # --- in: the planner --------------------------------------------------
    g[PROJECT] = project_node()
    g[CAST] = cast_node()
    g[TIMELINE] = timeline_node(timeline_json, source="saved")
    g[DISPATCH] = {
        "inputs": {
            "project": [PROJECT, 0],
            "timeline": [TIMELINE, 0],
            "mode": "next_pending",
            "manual_index": 1,
            "range_start": 1,
            "range_end": 0,
            "force_rerender": False,
            "prompt_format": "full_reference_6",
            "reset_stuck": False,
            "cast": [CAST, 0],
        },
        "class_type": "H3PlannerDispatcher",
        "_meta": {"title": "H3 Shot Dispatcher"},
    }
    g[SLICE] = {
        "inputs": {
            "audio": [CAST, AUDIO_1],    # the Cast Board's first audio output
            "shot": [DISPATCH, D_SHOT],
            "pad_short": True,
            "offset_seconds": 0.0,
        },
        "class_type": "H3PlannerTrackSlice",
        "_meta": {"title": "H3 Track Slice"},
    }
    # The gate sits on the base video latent and blocks one branch or the
    # other, so a draft run never touches the upscaler or the second sampler.
    g[GATE] = {
        "inputs": {
            "shot": [DISPATCH, D_SHOT],
            "latent": [O_SEPARATE, 0],
        },
        "class_type": "H3PlannerPassGate",
        "_meta": {"title": "H3 Pass Gate"},
    }
    g[VAULT] = {
        "inputs": {
            "shot": [GATE, 1],                    # draft_shot
            "images": [O_DECODE_DRAFT, 0],        # pre-upscale decode
            "audio": [O_DECODE_AUDIO, 0],
            "save_last_frame": True,
            "auto_advance": False,
            "max_auto_runs": 24,
        },
        "class_type": "H3PlannerVaultWrite",
        "_meta": {"title": "H3 Vault Write (draft)"},
    }
    g[VAULT_FINAL] = {
        "inputs": {
            "shot": [GATE, 3],                    # final_shot
            "images": [O_DECODE_VIDEO, 0],        # after upscale + re-sample
            "audio": [O_DECODE_AUDIO, 0],
            "save_last_frame": True,
            "auto_advance": False,
            "max_auto_runs": 24,
        },
        "class_type": "H3PlannerVaultWrite",
        "_meta": {"title": "H3 Vault Write (final)"},
    }

    # --- rewire the existing graph ----------------------------------------
    cond = g[O_H3COND]["inputs"]
    cond["width"] = [DISPATCH, D_WIDTH]
    cond["height"] = [DISPATCH, D_HEIGHT]
    cond["length"] = [DISPATCH, D_LENGTH]
    cond["ref_audios.ref_audio_0"] = [SLICE, 0]

    # the prompt primitive stays, so the dispatched prompt is visible on canvas
    g[O_PROMPT_STR]["inputs"]["value"] = [DISPATCH, D_PROMPT]

    # both decodes now hang off the gate rather than straight off the latent
    g[O_DECODE_DRAFT]["inputs"]["samples"] = [GATE, 0]      # draft_latent
    g[O_UPSCALER]["inputs"]["latent"] = [GATE, 2]           # final_latent
    # and the upscaler's target comes from the project, not a hand-set widget
    g[O_UPSCALER]["inputs"]["mode.megapixels"] = [PROJECT, 1]

    # cast slot 1 feeds the existing resize, which still feeds ref_image_0.
    # Its width/height were hardcoded to 1376x768 — a landscape box for a
    # portrait video — so the reference reached H3 in a different shape to the
    # frame it had to match. Drive it from the same size the sampler uses.
    g[O_RESIZE]["inputs"]["image"] = [CAST, 1]
    g[O_RESIZE]["inputs"]["width"] = [DISPATCH, D_WIDTH]
    g[O_RESIZE]["inputs"]["height"] = [DISPATCH, D_HEIGHT]

    # A SECOND reference. Only ref_image_0 was ever wired, so a cast with
    # a character AND a product handed H3 one image while the prompt cited
    # <Subject 1> and <Subject 2> — the second reference simply was not
    # there, and the product never appeared in the video.
    # ref_images is an autogrow group (ref_image_0..8), so more can be
    # added the same way: copy the resize node, take the next cast slot,
    # and wire it to the next ref_image_N.
    g[RESIZE_2] = {
        "inputs": {
            "image": [CAST, 2],
            "width": [DISPATCH, D_WIDTH],
            "height": [DISPATCH, D_HEIGHT],
            **{k: v for k, v in g[O_RESIZE]["inputs"].items()
               if k not in ("image", "width", "height")},
        },
        "class_type": g[O_RESIZE]["class_type"],
        "_meta": {"title": "Resize reference 2"},
    }
    g[O_H3COND]["inputs"]["ref_images.ref_image_1"] = [RESIZE_2, 0]

    # both passes take their seed from the plan, never from a roll
    g[O_NOISE_BASE]["inputs"]["noise_seed"] = [DISPATCH, D_SEED]
    g[O_NOISE_REFINE]["inputs"]["noise_seed"] = [DISPATCH, D_REFINE]

    # purge VRAM once the plan is resolved, before the model loads
    g[O_PURGE]["inputs"]["anything"] = [DISPATCH, D_INFO]
    g[O_DISPLAY]["inputs"]["source"] = [DISPATCH, D_INFO]

    return g


def build_plan(original, timeline_json):
    """Idea in, prompted timeline out. Run once; no model weights load."""
    g = {
        PROJECT: project_node(),
        CAST: cast_node(),
        # your existing prompt creator, kept whole — run it at the WHOLE
        # video's duration so its [Shot N] markers become the cut list
        O_PROMPT_NODE: copy.deepcopy(original[O_PROMPT_NODE]),
    }
    creator = g[O_PROMPT_NODE]["inputs"]
    creator["target_duration"] = 120.0
    creator["api_key"] = ""           # never ship a key in a template
    creator.pop("reference_image_1", None)
    creator["reference_audio"] = [CAST, AUDIO_1]  # first audio output

    g.update(plan_nodes())
    g[TIMELINE] = timeline_node(timeline_json, from_upstream=PROMPTER)
    return g


def build_story(timeline_json):
    """One brief in, the whole prompted timeline out — in one model call.

    No prompt creator in this graph at all: the Story Planner writes the
    treatment and the per-segment prompts together, which is what keeps an ad
    or a story continuous across the cuts. Three nodes, no model weights.
    """
    return {
        PROJECT: project_node(),
        CAST: cast_node(),
        STORY: {
            "inputs": {
                "project": [PROJECT, 0],
                "idea": ("A rapper crosses a Delhi rooftop at golden hour, "
                         "the city behind him, and takes the stairwell down "
                         "into the street."),
                "total_seconds": 30.0,
                "min_segment_seconds": 5.0,
                "max_segment_seconds": 10.0,
                "format": "cinematic ad",
                "dialogue": "spoken lines",
                "audio_role": "performed on camera",
                "window": 4,
                "single_call_max": 8,
                "chain_first_frames": False,
                "provider": "Ollama (Local)",
                "ollama_url": "http://127.0.0.1:11434",
                "ollama_model": "qwen3-vl:8b",
                "temperature": 0.35,
                "reuse_existing": True,
                "seed": 0,
                "cast": [CAST, 0],
                "style_prefix_override": "",
                "api_key": "",
                "api_model": "",
                "max_output_tokens": 8192,
                "num_ctx": 16384,
                "keep_alive": "10m",
                "request_timeout": 900,
            },
            "class_type": "H3PlannerStoryPlanner",
            "_meta": {"title": "H3 Story Planner"},
        },
        TIMELINE: timeline_node(timeline_json, from_upstream=STORY),
    }

def build_slice(original, timeline_json):
    """A treatment into segment prompts with no model in the graph at all.

    Same shape as the plan workflow, with the Segment Prompter swapped for the
    Slicer: no provider, no key, no Ollama. Runs in milliseconds.
    """
    g = {
        PROJECT: project_node(),
        CAST: cast_node(),
        O_PROMPT_NODE: copy.deepcopy(original[O_PROMPT_NODE]),
    }
    # Same sanitising as the plan graph: never ship a key, and drop the wires
    # that pointed at nodes this graph does not contain.
    creator = g[O_PROMPT_NODE]["inputs"]
    creator["target_duration"] = 52.0
    creator["api_key"] = ""
    creator.pop("reference_image_1", None)
    creator["reference_audio"] = [CAST, AUDIO_1]

    g.update({
        SLICER: {
            "inputs": {
                "project": [PROJECT, 0],
                "h3_prompt": [O_PROMPT_NODE, 0],
                "total_seconds": 52.0,
                "shots_per_segment": 2,
                "min_segment_seconds": 2.0,
                "audio_role": "performed on camera",
                "carry_sections": True,
                "cast": [CAST, 0],
                "style_prefix_override": "",
            },
            "class_type": "H3PlannerSegmentSlicer",
            "_meta": {"title": "H3 Segment Slicer"},
        },
        TIMELINE: timeline_node(timeline_json, from_upstream=SLICER),
    })
    return g


def build_stitch(original, timeline_json):
    """A four-node graph: nothing loads, nothing samples, ffmpeg does the work."""
    return {
        PROJECT: project_node(),
        TIMELINE: timeline_node(timeline_json, with_cast=False,
                                source="saved"),
        STITCH: {
            "inputs": {
                "project": [PROJECT, 0],
                "timeline": [TIMELINE, 0],
                "prefer": "chosen",
                "trim_to_plan": True,
                "filename_prefix": "h3planner",
                "min_clips": 2,
                "contact_sheet": True,
                "audio": "from clips",
            },
            "class_type": "H3PlannerStitch",
            "_meta": {"title": "H3 Stitch Timeline"},
        },
    }


def verify(graph, label):
    """Every link must point at a node that exists, on a plausible slot."""
    problems = []
    for node_id, node in graph.items():
        for key, value in node.get("inputs", {}).items():
            if not (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str)):
                continue
            target, slot = value
            if target not in graph:
                problems.append("%s.%s -> missing node %s"
                                % (node_id, key, target))
            elif not isinstance(slot, int) or slot < 0:
                problems.append("%s.%s -> bad slot %r" % (node_id, key, slot))
    print("%-16s %2d nodes, %s" % (label, len(graph),
                                   "OK" if not problems else "PROBLEMS"))
    for p in problems:
        print("    ! " + p)
    return not problems


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.expanduser("~"), "Downloads", "video_minimax_h3_r2v_test.json")
    with open(src, encoding="utf-8") as f:
        original = json.load(f)

    timeline_json = load_example_timeline()
    outputs = {
        "workflow_plan.json": build_plan(original, timeline_json),
        "workflow_story.json": build_story(timeline_json),
        "workflow_slice.json": build_slice(original, timeline_json),
        "workflow_render.json": build_render(original, timeline_json),
        "workflow_stitch.json": build_stitch(original, timeline_json),
    }

    ok = True
    for name, graph in outputs.items():
        ok &= verify(graph, name)
        with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
