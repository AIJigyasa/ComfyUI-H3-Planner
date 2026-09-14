# The three workflows

Generated from your `video_minimax_h3_r2v_test.json` by `build_workflows.py`.
All in **ComfyUI API format** — load with the normal Load button; ComfyUI
detects the format and lays the graph out automatically, so drag things where
you want them and re-save as your own workflow.

API format is deliberate: it keys inputs by name rather than widget order, so
nothing depends on the internal layout of `ImageResizeKJv2`,
`DiffusionModelLoaderKJ`, `LoadAudioUI` or any other third-party node. They all
pass through byte-identical.

| File | Runs | What it does |
|---|---|---|
| `workflow_plan.json` | once | idea → treatment → segments → prompts → timeline |
| `workflow_render.json` | once per segment | one segment per queue run, into the vault |
| `workflow_stitch.json` | once | timeline + vault → one mp4 |

They share state through `output/h3_planner/<project>/timeline.json`, so keep
`project_name` identical in all three. Plan and stitch load the saved timeline;
only the planning workflow writes it.

---

## 1. workflow_plan.json — seven nodes, no model weights

```
H3 Cast Board ──┬──────────────────────────────────────────────┐
                │                                              │
LoadAudioUI ──► H3 Full-Reference Prompt Creator (yours) ──► H3 Treatment Splitter
                                                                   │        │
                                                            timeline│        │style_prefix
                                                                   ▼        ▼
                                                            H3 Segment Prompter ──► H3 Timeline
```

**Your prompt creator is kept whole** — it writes the treatment. The only
change is `target_duration: 60.0` instead of 6.0: run it at the **whole video's
length**, so its `[Shot N]` markers become the cut list. It caps at 120 s, so
anything longer needs several treatments chained, or the Segment Prompter's
`from_idea` mode.

**H3 Treatment Splitter** is pure arithmetic — no model, no cost. It groups the
treatment's shots into segments that fit the 14.375 s ceiling, never splitting a
shot, and lifts the style prefix off `[Shot 1]`.

**H3 Segment Prompter** then writes a complete standalone six-section prompt per
segment: style prefix copied verbatim into every `[Shot 1]`, timestamps rebased
to zero and validated against that segment's rendered length, `subject_definitions`
limited to the tags that segment actually cites, and `(S1)`/`(S2)` speaker IDs
held stable across segments. One call per segment; segments already written from
the same inputs are reused, so re-running after one edit costs one call.

Before queueing: **drop your references onto the Cast Board.** The old
`LoadImage` node is gone — the board holds them itself.

## 2. workflow_render.json

Same as before with `source: saved` on the Timeline, so it picks up whatever the
planning workflow wrote.

**Removed**: `115` Resolution Selector, `131` Math Expression, `132` Float
(Duration), `193` prompt creator, `139` LoadImage.

**Rewired** — everything from `MiniMaxH3ReferenceToVideo` rightward untouched:

```
303.width/height/length  -> 136 MiniMaxH3ReferenceToVideo
303.prompt               -> 138 PrimitiveStringMultiline -> 136.prompt
303.seed                 -> 129 RandomNoise   (base pass)
303.refine_seed          -> 207 RandomNoise   (upscale pass)
303.shot                 -> 304 Track Slice and 305 Vault Write
301.image_1              -> 143 ImageResizeKJv2 -> 136.ref_image_0
304.audio                -> 136.ref_audio_0
122 / 121 (decodes)      -> 305 Vault Write
303.info                 -> 184 PurgeVRAM, 194 Display Any
```

`shot` is the wire that matters: it runs from the dispatcher **around** the
sampler into Vault Write, carrying segment id, pass, take and seed past the part
of the graph that knows nothing about timelines.

Set batch count to the number of segments and queue. Extra runs no-op.

## 3. workflow_stitch.json

Project, Timeline (`saved`), Stitch. Nothing loads, nothing samples — it reads
the vault and runs one ffmpeg pass. Output in
`output/h3_planner/<project>/renders/`, with a contact sheet beside it.

---

## Two things to set for yourself

**1. The audio loader is still trimmed to one clip.**

`157 LoadAudioUI` keeps your `start_time 73.03 / end_time 83.12`. That was right
for a single 10-second clip, but H3 Track Slice does the windowing now, so the
loader should hand it the **whole track** and each segment's `audio_start` picks
its own window. Left as-is, `audio_start` becomes relative to that 10-second
slice. I did not change it because I cannot verify what that node treats as
"load everything".

**2. The draft pass still pays for the upscale.**

Your graph runs both passes in one go: base sample → `220 LTXVSeparateAVLatent`
→ `221 MinimaxH3LatentUpscaler3D` (fixed at 1.2 MP) → `222` → `209` → `122`.
Switching **H3 Project** to `render_pass: draft` changes the *base* resolution,
but `221` still upscales to its own widget value.

To make drafting actually cheap: bypass `221, 222, 207, 208, 209, 211, 215`
(Ctrl-B) and point `305 Vault Write.images` at `218 VAEDecode`, which already
decodes the pre-upscale latent. Un-bypass and point back at `122` for finals.
Or just set `221 mode.megapixels` by hand per pass.

Either way the bookkeeping is unaffected — drafts and finals are separate vault
entries and the stitcher takes whichever exists.

---

## Regenerating

```bash
python examples/build_workflows.py path/to/your_workflow_api.json
```

Verifies every link resolves before writing, and prints any that don't.

## A note on the original file

The workflow you sent had a live Google API key in node `193`'s `api_key`
widget. The generated files clear it, so these are clean — but the original
still has it, and it travels with the file. Worth rotating.
