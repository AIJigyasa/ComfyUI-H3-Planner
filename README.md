# ComfyUI H3 Planner

Long-form video orchestration for **MiniMax H3**. You plan a video as segments;
the pack hands your sampler one segment per queue run, files every result in a
timeline-addressed vault, and stitches the lot back to the durations you planned.

**Sampling is not part of this pack.** It emits plain primitives that wire
straight into `MiniMaxH3ReferenceToVideo`, and takes frames back afterwards.
Your checkpoint, LoRA, sigmas and latent upscaler stay exactly as they are.

**Twelve nodes, from a one-line brief to a finished video.**

- **Plan** a whole piece: the **Story Planner** writes an advert, short film or
  UGC piece from one brief; the **Segment Slicer** cuts an existing timestamped
  treatment into segments with no model at all; the **Beat Map** lays a music
  video's cuts on the actual bar lines of the track.
- **Cast** it on the **Cast Board**, which uploads references into the node and
  tags them `<Subject N>`, `<Picture N>`, `<Audio N>`, `<Video N>` for you.
- **Render** one segment per queue run with the **Shot Dispatcher**, **Track
  Slice**, **Pass Gate** and **Vault Write**, with draft and upscaled final
  passes kept separately.
- **Review and fix** on the **Timeline** card strip, including rewriting one
  shot from a plain-English note without re-planning the rest.
- **Stitch** it back to the exact planned durations, in sync with the track.

**Ready-made workflows** built from a working single-clip H3 graph live in
`examples/` — see `examples/WORKFLOW.md`.

---

## Why it exists

Three problems show up the moment you try to make something longer than one clip:

1. **ComfyUI runs a graph once.** Twelve segments means twelve runs, and
   something has to remember which ones are done — through interruptions,
   restarts, and edits made halfway through.
2. **H3 renders on a 17-frame ladder.** Ask for 6.000 s and you get 6.583 s.
   Over a dozen segments that is up to 8 s of drift against your track.
3. **Peak VRAM must not grow with length.** A three-minute video has to cost
   the same VRAM as a fifteen-second one.

The pack answers all three: a claim/complete state machine, target-vs-render
durations with trim-on-stitch, and one segment per queue run with everything
file-backed.

---

## Install

Install **H3 Planner** from ComfyUI Manager, or clone it into
`ComfyUI/custom_nodes/`:

```bash
git clone https://github.com/AIJigyasa/ComfyUI-H3-Planner
```

then restart ComfyUI.

### Requirements

| What | Needed for | Notes |
|---|---|---|
| A ComfyUI with the **MiniMax H3** nodes | everything | The pack drives `MiniMaxH3ReferenceToVideo`; it does not sample on its own. Developed against ComfyUI 0.35.0. |
| **ffmpeg** | everything | Encodes every stored clip and does the stitching. Put it on PATH, or `pip install imageio-ffmpeg` into the ComfyUI environment. |
| [**ComfyUI-H3-Prompt-Creator**](https://github.com/AIJigyasa/ComfyUI-H3-Prompt-Creator) | Story Planner, Segment Prompter, Refine | Supplies the H3 system guide and the model providers. Install it the same way. Without it those three say so; every other node works. |
| A language model | Story Planner, Segment Prompter, Refine | A local **Ollama** vision model, or OpenAI, Anthropic, OpenRouter or Google Gemini through the Prompt Creator. |
| **librosa** | Beat Map only | Optional and not installed automatically. See below. |

No other Python packages: numpy, Pillow and torch already ship with ComfyUI.

**About API keys.** The planner never stores a key in a timeline, and Refine
reads it from the provider's environment variable. A key typed into a node's
`api_key` field is different: ComfyUI saves every widget value inside the
workflow file, so that key travels with any workflow you share or export. Set
the provider's environment variable instead, and leave the field empty.

### Optional: the Beat Map

The Beat Map analyses audio with librosa, which is heavy, so it is an opt-in
install rather than something every user gets. Run it with the Python that runs
ComfyUI:

```bash
python -m pip install -r requirements-audio.txt
```

On ComfyUI portable or Easy-Install for Windows that is the embedded Python:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-H3-Planner\requirements-audio.txt
```

Every other node works without it, and the Beat Map says what is missing if you
use it first.

---

## Wiring it into an existing H3 workflow

Starting from a working single-clip graph (`video_minimax_h3_r2v_test.json` is
the reference), you **remove three nodes** and add the planner around what is
left:

| Remove | Replace with |
|---|---|
| Float (Duration) | `H3 Shot Dispatcher` → `length` |
| Math Expression (the `% 17` one) | the dispatcher does this, and records it |
| the per-clip prompt creator | `H3 Timeline` → dispatcher → `prompt` |

Everything from `MiniMaxH3ReferenceToVideo` rightward is untouched.

```
H3 Project ─┬─────────────────────────────────────────────┐
            │                                             │
H3 Cast Board ──► image_1..8 ──► ImageResizeKJv2 ──► ref_images.ref_image_N
            │                                             │
            └─► H3 Timeline ──► H3 Shot Dispatcher ──┬──► prompt
                                                     ├──► length
                                                     ├──► width / height
                                                     ├──► seed        → RandomNoise (base)
                                                     ├──► refine_seed → RandomNoise (upscale)
                                                     ├──► shot ───────────────────┐
                                                     └──► (audio) H3 Track Slice ─┼─► ref_audio_0
                                                                                  │
   VAEDecode ──► images ────────────────────────────► H3 Vault Write ◄────────────┘
   VAEDecodeAudio ──► audio ───────────────────────►

   H3 Project + H3 Timeline ──► H3 Stitch Timeline ──► final mp4
```

The `shot` output is the important wire: route it **around** your sampler into
Vault Write. It carries segment id, pass, take and seed past the sampler, so the
vault knows what it is receiving without your subgraph knowing timelines exist.

### Running it

1. Author segments in `H3 Timeline`, queue once — check the report.
2. Set batch count to the number of segments and queue. Each run renders one.
   Queue too many and the extras no-op; queue too few and you carry on later.
3. Run `H3 Stitch Timeline`.
4. Happy with the cut? Add an **H3 Pass Gate** and a second Vault Write for the
   upscale branch (see the node below), set `render_pass` to `final` on
   `H3 Project`, and queue again — every segment is outstanding for the new
   pass. Changing `render_pass` alone is not enough: with one Vault Write there
   is one pass name in the graph, and the upscaled clip overwrites the draft.

Turn on `auto_advance` in Vault Write for one-button rendering. It stops when
the timeline is done, refuses to run if you already queued things manually, and
respects `max_auto_runs`.

---

## The frame ladder

H3 does not accept an arbitrary length. The reference workflow's expression

```
max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17
```

means **`length % 17 == 5`, minimum 5**. At 24 fps that is 21 usable rungs:

```
frames    5    22    39    56    73    90   107   124   141   158   175
sec    0.208 0.917 1.625 2.333 3.042 3.750 4.458 5.167 5.875 6.583 7.292

frames  192   209   226   243   260   277   294   311   328   345
sec    8.000 8.708 9.417 10.125 10.833 11.542 12.250 12.958 13.667 14.375
```

Note that **15.0 s is not reachable** — 14.375 s is the ceiling — and that
8.000 s is one of the few round numbers on the ladder.

Because the snap always rounds up, every segment renders slightly long. So each
segment carries two numbers:

- `target_duration` — what the cut actually needs. A free float; plan in this.
- `render_duration` — the rung at or above it. What H3 is asked for.

**The stitcher trims each clip back to `target_duration`.** That is what keeps a
long video in sync with its track instead of drifting a fraction of a second per
cut. Turning `trim_to_plan` off is supported and will drift; the report says so.

If a different VAE or LoRA changes the rule, `frame_modulus` / `frame_remainder`
/ `frame_minimum` on `H3 Project` are widgets. Set modulus to 1 to disable.

---

## Nodes

### H3 Project
Everything global: fps, aspect, draft/final megapixels, alignment, base seed, the
ladder, and the stale-run timeout. Its `render_pass` (`draft`, `final`,
`one_go`) is what the Pass Gate reads. The Shot Dispatcher outputs `width`/`height`
for the base sample, so `ResolutionSelector` is optional. The `report` output
prints the whole ladder —
read it once and you will know exactly what lengths you can ask for.

### H3 Cast Board
Self-contained: drop images, video and audio onto the node itself. Files upload
to `input/h3_planner/`, so a saved workflow reopens with its cast intact. Audio
and video cards play in place, and each has in and out points, so you can trim a
reference without leaving the node. The trimmed file is what leaves the board.

References can also be wired in. `ref_image_0`, `ref_audio_0` and `ref_video_0`
each grow a new socket as you fill one, the same way the H3 sampler's own inputs
do. `ref_video_` takes IMAGE frames, which is what H3 wants. Nothing is
required: a board that only uses uploads runs as it is.

**Tags are derived from the role, never typed.** The H3 guide is explicit that a
character or product image is *not* automatically a `<Picture N>` — if it only
supplies reusable identity it is a `<Subject N>`, and `<Picture N>` is reserved
for a concrete first/last/key frame or composition anchor. The four kinds are
numbered independently, so `<Subject 1>` and `<Picture 1>` can be different
assets. Wired references are numbered on from the uploaded cards.

| Role | Tag |
|---|---|
| character, product, style, wardrobe, environment, prop | `<Subject N>` |
| first_frame, last_frame, keyframe, composition | `<Picture N>` |
| video | `<Video N>` |
| audio | `<Audio N>` |

Numbering follows card order within each kind, so reordering the board renumbers
the tags *and* moves the wired outputs with them — which is what keeps a
prompt's `<Subject 2>` pointing at the slot it was written for. **Copy tags**
puts the whole block on the clipboard.

Outputs: `cast`, `image_1`…`image_8`, `audio_1`…`audio_4`, `video_1`…`video_4`,
`report`. A card past the last slot of its kind is left out of the cast and
named in the report, rather than tagged with nothing behind it.

### H3 Story Planner
One brief in, a whole planned and written timeline out. Built for adverts, short
films, UGC and product pieces. It makes two kinds of call: one to direct the
whole piece — the look, the world, the scenes, the script, who says each line —
and then the six-section H3 prompts, a few segments per call.

Everything a model is unreliable at is enforced in code afterwards:

- **Runtime.** `total_seconds` is a constraint. Segment lengths snap to real
  ladder rungs and the count is fitted so the plan adds up to what you asked.
- **One continuous piece.** Every segment's `summary` states the same location,
  time of day and light in identical words, which part it is, and what it opens
  and closes on. A shot that ignores where the previous clip ended is given that
  state; a segment that runs past its own ending into the next one has the
  extra shots removed and reported, so an action never renders twice.
- **Tags.** Every reference is bracketed and bound, in the section H3 actually
  reads, and a tag the cast does not have is stripped.
- **Dialogue.** Lines are budgeted for the runtime, written inside
  `<d>[English] …</d>` with a stable speaker ID, and a segment with no line is
  told that nobody speaks.

`audio_role` decides what a connected audio reference is **for**:

| Setting | Use it when |
|---|---|
| `performed on camera` | the audio is the soundtrack and a visible subject raps or sings it |
| `voice-over (off camera)` | the audio is the soundtrack and nobody on screen voices it |
| `background only` | the audio is the soundtrack and nobody performs it |
| `voice sample (clone the timbre)` | each audio is a **character's voice**; the script is written here and spoken in it |

For a two-character scene, connect one voice sample per character and say who is
who in the brief, for example *"use `<audio 1>` for the man and `<audio 2>` for
the woman"*. The report shows the casting and each voice's line count, and warns
when the role contradicts the brief.

With `reuse_existing` on, re-queueing with nothing changed costs no model calls. `chain_first_frames`
additionally hands each clip's last frame to the next clip inside a scene.

Outputs: `timeline`, `beat_sheet`, `report`.

### H3 Segment Slicer
Cuts a treatment you already have — a prompt with `[Shot N] 00:05.500` markers,
written at the whole video's length — into segments. **No model, no key, no
network.**

A segment ends on a shot's own timestamp and never inside a shot. Shots are
packed left to right while the segment stays under the length ceiling and under
`shots_per_segment`. Each segment gets the treatment's shared sections and its
own shots, with the clock rebased to that segment's zero.

Wire the Beat Map's `cuts` in and the boundaries come from the music instead:
shots land in whichever cut their timestamp falls into, and any shot outside
every cut is named in the report.

### H3 Beat Map
For music videos. Finds the tempo, the bar lines and the biggest energy rises in
a track, and lays out segments that start on a downbeat. Needs librosa; see
Install.

Each segment keeps the **exact** musical span as its planned length, and the
stitcher trims the ladder overshoot back off, so every cut lands on the one and
the error does not build up down a four-minute track. `cover_whole_track`
covers the lead-in before the first downbeat and the tail after the last bar, so
no second of the song goes unrendered. `cut_on_drops` shortens the clip before a
big lift so the next one starts exactly on it.

Outputs: `timeline`, `cuts` for the Segment Slicer, `report`.

### H3 Segment Prompter
Writes a complete standalone six-section H3 prompt for each segment — because a
*slice* of a long prompt is not a valid prompt. It has no style prefix, no
`subject_definitions` of its own, no soundscape, and its timestamps start
wherever the cut fell.

Per segment it enforces, in code rather than by asking:

- the style prefix appears verbatim at the head of `[Shot 1]`, so segment six
  cannot drift to a different look
- `[Shot 1]` carries no timestamp; later shots are respaced strictly inside the
  segment's *rendered* duration
- all essential action completes by the *planned* duration; the ladder overshoot
  is an explicit hold
- `subject_definitions` covers only the tags that segment cites, de-duplicated
- `(S1)`/`(S2)` speaker IDs stay stable across segments
- tags above the cast size are flagged

Two modes: from a timestamped treatment in `h3_prompt`, or from an `idea` with no treatment at all.
Segments already written from the same inputs are reused, and locked segments
are never touched — so re-running after one edit costs one call, not twelve.

It **imports** `ComfyUI-H3-Prompt-Creator` rather than copying it, reusing that
pack's system guide, providers, JSON repair and shot-timing enforcement. Soft
dependency, shared with the Story Planner and Refine: without it those say so
and everything else still works.

### H3 Timeline
A horizontal strip of segment cards — thumbnail, beat, planned vs rendered
duration, state chip, seed — with an editor underneath for the selected card:
id, beat, duration, `audio_start`, link type and the prompt itself. Buttons to
add, duplicate, delete, reorder, lock, and redo a segment.

Authored fields live in the `timeline_json` widget, so they save and travel with
the workflow (press **JSON** to edit it raw). Render state — clips, takes,
thumbnails, failures — is read live from disk, so the strip shows what has
actually rendered while you are still editing. The two never fight: the cards
only ever write authored fields.

Persists to `output/h3_planner/<project>/timeline.json`.

**Refine** rewrites one card's shot description from a plain-English note, for
example *"make it a low angle, she walks away from camera"*, without leaving the
timeline or re-planning anything else. Only `detailed_description` changes: the
tags, spoken lines, style prefix and every other section are guaranteed to
survive, and a revision that loses any of them is refused and the card left as
it was. It uses the provider that wrote the timeline, or local Ollama when the
timeline was made without a model.

The `segment_count` output is the number of segments, for wiring into the
Stitcher's `min_clips`.

Accepts a full object, or a bare array, or even a list of strings. Minimum useful
segment: `{"id": "s1", "duration": 6.0, "prompt": "..."}`.

Re-running with an edited prompt resets **only** the segments that changed —
everything already rendered keeps its clips. That is what makes it safe to keep
tweaking mid-render. It also warns when a prompt cites `<Subject 4>` and the cast
has three images, or when a duration will not fit the ladder.

### H3 Shot Dispatcher
Claims one segment per run and emits the primitives above.

A segment is *outstanding for a pass* when it is not locked, not running, and has
no stored clip for that pass. Claiming is idempotent, so:

- queue forty runs against twelve segments → twelve render, twenty-eight no-op
- interrupt, edit two prompts, resume → only those two re-render
- kill ComfyUI mid-render → the next run picks up exactly where it stopped
- a run that dies leaves its segment `running`; after `stale_minutes` it returns
  to `pending` on its own

When nothing is outstanding, the sampler branch is execution-blocked rather than
erroring, so a run with nothing to do finishes clean.

Modes: `next_pending`, `manual_index`, `range`, `failed_only`.

### H3 Track Slice
The window of your reference track that belongs to this segment. Slices
`render_duration`, not target, so H3 hears audio for every frame it renders and
the stitcher removes picture and sound overhang together. Pads with silence when
the track runs out.

### H3 Vault Write
The only node that knows a run succeeded, so completion lives here. Encodes the
clip to mp4 (plus wav and a thumbnail), files it under `(segment, pass, take)`,
and advances the timeline. Optionally writes the last frame for the next
segment's `continue` link, and optionally auto-queues the next run.

Promotion never overwrites: a draft and a final for the same segment are separate
entries, so a promote is reversible.

### H3 Pass Gate
Decides which half of the sampling graph runs. Sits straight after the base
sampler, takes its latent and the shot, and hands out two of each: one stamped
`draft` for the base decode, one stamped `final` for the latent upscaler and the
second sampler. Wire a Vault Write to each.

`render_pass` on H3 Project picks the mode. `draft` runs the base branch only,
`final` runs the upscale branch and stores finals, `one_go` stores both from a
single base sample. The branch that is not running is execution-blocked, so it
costs no VRAM and no time.

Without the gate there is only one pass name in the graph, and a second Vault
Write simply overwrites the first.

### H3 Stitch Timeline
Resolves each segment's clip, trims it to plan, and joins them in one ffmpeg
pass — frame accurate, and on **files**, never image tensors. The canvas is the
largest clip present, so a final is never downscaled to match a draft. Also
writes a contact sheet.

- `prefer` chooses `chosen`, `final` or `draft`. Set to `final` or `draft`, the
  node waits for clips **of that pass**, so a final stitch does not quietly join
  drafts; any segment that did fall back is named under MIXED PASSES.
- `min_clips` is how many clips to wait for. `0` means every segment, so the
  node can sit in the render graph and stitch itself the moment the last card
  finishes. The Timeline's `segment_count` output wires straight into it.
- `audio` takes each clip's own sound, lays the original track over the whole
  video, or leaves it silent.

The `video` output is a native VIDEO: wire it to **Save Video**. It is not
decoded back into frames, which for a long render would be many gigabytes of
memory for a file that already exists.

---

## Authoring a timeline

```json
{
  "name": "delhi-rap",
  "segments": [
    {
      "id": "s1",
      "beat": "hook",
      "duration": 3.0,
      "audio_start": 0.0,
      "prompt": {
        "subject_definitions": "<Subject 1> is the rapper from the reference photo, ...",
        "summary": "[reference generation] ...",
        "retention_analysis": "<Subject 1>: attribute_transfer, [Shot 1]",
        "detailed_description": "[Shot 1] 2D illustration, ...",
        "overall_soundscape": "Street ambience, ...",
        "non_diegetic_music": "N/A"
      }
    },
    { "id": "s2", "beat": "verse", "duration": 8.0, "audio_start": 3.0,
      "link": "continue", "prompt": "..." }
  ]
}
```

Per segment: `id`, `beat`, `duration` (or `target_duration`), `prompt` (string or
the six named sections), `audio_start`, `link` (`cut` / `continue` / `match`),
`seed`, `refine_seed`, `cast_used`, `locked`, `notes`. Only `duration` and
`prompt` really matter; seeds are derived deterministically from the project's
`base_seed` and stay fixed across re-plans, which is what makes a promoted
segment match the draft you approved.

A prompt given as an object is flattened exactly the way
`ComfyUI-H3-Prompt-Creator` renders full-reference mode, so prompts from that
node and prompts assembled here are byte-identical.

---

## On disk

```
output/h3_planner/<project>/
  timeline.json                     the spine — state survives restarts
  vault/vault.json                  clip manifest
  vault/s1__draft__t0.mp4|.wav|.jpg
  frames/s1_last.png                for continue links
  renders/h3planner_<project>_<stamp>.mp4
```

Delete `timeline.json` to start over; delete a clip from `vault/` and its segment
becomes outstanding again.

---

## Tests

```bash
sh run_tests.sh
```

Runs every suite: the browser modules parse as ES modules, the ladder and state
machine, prompt enforcement, the Story Planner and Slicer end to end against
adversarial model stubs, refine, the beat map, and file I/O. No ComfyUI needed;
the Beat Map's analysis half skips when librosa is not installed.

---

## License

GPL-3.0. See [LICENSE](LICENSE).

Copyright (C) 2026 AI Jigyasa.
