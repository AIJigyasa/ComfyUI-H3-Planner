# H3 Planner — design

A multi-segment timeline pack for MiniMax H3. One idea in, a finished multi-minute
video out: ads, music videos, short films, product films.

Status: **spine built** — Project, Cast Board, Timeline, Dispatcher, Track Slice,
Vault Write, Stitch are implemented and tested. Planner (§4), Continuity Pass
and the timeline UI (§5) are next. Sections below that are still design.

Pack name: **H3 Planner** (`ComfyUI-H3-Planner`).

**Scope.** H3 runs locally. Sampling — the checkpoint, the latent upscale, the
scheduler — is *outside* this pack and stays that way. This pack is the
orchestration around it: plan the segments, hand the sampler one segment at a time,
take the frames back, keep the book, stitch the result. The sampler subgraph is a
black box wired between two nodes (§2).

H3 accepts **any duration from 1 to 15 s**, continuous. That is a bigger deal than
it sounds — see §4.

---

## 0. The one naming decision everything else hangs off

H3 already has an internal shot grammar. A single 15 s H3 prompt contains
`[Shot 1]`, `[Shot 2] At 00:04.500`, … — the model cuts *inside* one generation.
So "shot" means two different things and the data model has to keep them apart:

| Term | What it is | Who owns it |
|---|---|---|
| **Beat** | a story unit — hook, reveal, CTA | Story Planner |
| **Segment** | **one generation** — one H3 prompt, ≤ 15 s, one vault clip | Timeline / Dispatcher |
| **Shot** | `[Shot N]` cut *inside* a segment's prompt | the H3 prompt itself |

Everything in this pack is **segment**-addressed. When the UI says "Shot 3" to the
user it means segment 3, but the JSON says `segments[2]` and never confuses the two.
A 90 s music video = 1 project → ~7 beats → 7–9 segments → ~20 internal shots.

---

## 1. Custom types

Four dict-backed types carried between nodes. All JSON-serialisable, all written to
disk so a restart or a crash mid-render loses nothing.

```
H3_PROJECT   name, fps, aspect, draft_res, final_res, base_seed, provider cfg, root dir
H3_CAST      ordered reference registry (see §3)
H3_TIMELINE  the whole thing — segments[], each with prompt/refs/seed/state/clip ids
H3_SHOT      one segment, unpacked, ready for the sampler
```

`H3_TIMELINE` is the spine. Sketch of one segment:

```json
{
  "id": "seg_03",
  "index": 3,
  "beat": "reveal",
  "duration": 11.5,
  "frames": 184,
  "link": "cut",
  "first_frame_path": null,
  "cast_used": ["hero", "bottle"],
  "tag_map": {"hero": "Subject 1", "bottle": "Subject 2"},
  "prompt": {
    "subject_definitions": "...",
    "summary": "...",
    "retention_analysis": "...",
    "detailed_description": "...",
    "overall_soundscape": "...",
    "non_diegetic_music": "..."
  },
  "seed": 884213,
  "state": "final",
  "take": 2,
  "clips": {"draft": "clip_1724....mp4", "final": "clip_1725....mp4"},
  "chosen": "final",
  "notes": "hand-edited the CTA line"
}
```

`link` is one of `cut | continue | match`. `state` is one of
`pending | running | draft | final | failed | locked`.

Two rules that make the whole pack behave:

- **Nothing mutates a segment silently.** Any node that rewrites a prompt bumps
  `prompt_rev` and sets `state: pending` — unless the segment is `locked`.
- **`seed` is written at plan time, never at render time.** That is what makes
  draft → final reproducible.

---

## 2. Orchestration — the loop, and the seam your sampler plugs into

ComfyUI runs a DAG once per queue press. Three ways to get a loop; the pack uses
the first and offers the third.

1. **Per-run cursor (core).** `H3 Shot Dispatcher` emits one segment per queue run.
   You queue with batch count = segment count, or press Queue N times. Works on
   every ComfyUI version, no execution-inversion nodes. Your `Video Gateway`
   already proves the pattern.
2. **Native for-loop nodes.** Version-fragile, and the whole timeline stays resident
   in memory for the entire run — exactly wrong for low VRAM. Rejected.
3. **Auto-queue driver (opt-in).** Ships as a toggle, not the default (§2.3).

**Why per-run is also the low-VRAM answer:** one segment per run means the graph
tears down between segments — model unload, latent freed, clip written to disk as
mp4. Peak VRAM is one 15 s clip whatever the timeline length. A 3-minute video
costs the same VRAM as a 15 s one.

### 2.1 The cursor is a state machine, not a counter

A counter is wrong, and this is the detail that decides whether the pack survives a
real render. **The dispatcher cannot know whether the run succeeded** — it finishes
executing long before the sampler does. So the advance is driven from the *far* end:

- **Dispatcher**, at run start: claim the first segment whose `state == pending`,
  mark it `running`, stamp `claimed_at`, emit it.
- **Vault Write**, at run end: write the clip, set `state` to `draft` or `final`,
  record the clip id and take number.
- A segment left `running` past a stale timeout (default 30 min) reverts to
  `pending` on the next dispatch, with a warning. A crashed or cancelled run
  self-heals; a killed ComfyUI resumes exactly where it stopped.

Consequences that fall out for free: queue 40 runs for a 12-segment timeline and the
extra 28 no-op instead of re-rendering; interrupt mid-render, edit two prompts in
the timeline UI, resume, and only those two are pending; nothing is ever rendered
twice or skipped.

Dispatcher selection modes: `next_pending` (default), `manual_index`, `range`,
`failed_only`, `all_again`. `IS_CHANGED` returns NaN so it re-runs every queue.

### 2.2 The sampler seam — measured against the real graph

Taken from `video_minimax_h3_r2v_test.json`. The pack replaces exactly three nodes
in it — the Float (Duration), the Math Expression, and the prompt creator — and
leaves everything from `MiniMaxH3ReferenceToVideo` rightward untouched.

| Dispatcher output | Type | Wires to |
|---|---|---|
| `prompt` | STRING | `MiniMaxH3ReferenceToVideo.prompt` (via the existing string primitive) |
| `length` | INT | `MiniMaxH3ReferenceToVideo.length` — replaces the Math Expression |
| `width`, `height` | INT | `MiniMaxH3ReferenceToVideo.width/height` |
| `megapixels` | FLOAT | `ResolutionSelector.megapixels` — the pass-aware alternative |
| `aspect_ratio` | STRING | `ResolutionSelector.aspect_ratio` |
| `seed` | INT | `RandomNoise.noise_seed` (base pass, node 129) |
| `refine_seed` | INT | `RandomNoise.noise_seed` (upscale pass, node 207) |
| `ref_1 … ref_6` | IMAGE | `ref_images.ref_image_0…N`, usually through `ImageResizeKJv2` |
| `ref_audio` | AUDIO | `ref_audios.ref_audio_0` — already sliced, see §2.5 |
| `first_frame` | IMAGE | only on `link: continue` |
| `fps` | FLOAT | `CreateVideo` / vault |
| `pass` | STRING | `draft` or `final` — drive a bypass switch on the upscale branch |
| `shot` | `H3_SHOT` | **opaque token — route around the sampler into Vault Write** |

That last row is the trick: `shot` carries segment id, pass, take and seed *past*
the sampler, so Vault Write knows what it is receiving without the sampler knowing
timelines exist. Your subgraph touches only the primitives.

**No negative prompt.** The graph runs `BasicGuider` with the 4-step turbo LoRA —
guidance-free, no negative conditioning anywhere. So the pack does not emit one, and
the planner must not write "avoid X" instructions expecting them to land somewhere.

**Prompt is one flat string.** `H3FullReferenceVideoPromptCreator` already returns
`h3_prompt` as a single STRING. The timeline stores the six sections separately (so
the Continuity Pass can rewrite one section without touching the rest) and flattens
on the way out using the same joiner that node uses. `prompt_format` picks
`full_reference_6` / `video_3field` / `flat`.

### 2.3 Length is a 17-frame ladder, and this is the sharpest constraint in the pack

The workflow's Math Expression is
`max(5, round(a*24)) + (5 - (max(5, round(a*24)) % 17)) % 17`, which is a long way
of saying:

> **`length ≡ 5 (mod 17)`, minimum 5, at 24 fps.**

So H3 does not accept any duration — it accepts 21 of them under 15 s, spaced
17/24 = **0.7083 s** apart:

```
frames    5    22    39    56    73    90   107   124   141   158   175
sec    0.208 0.917 1.625 2.333 3.042 3.750 4.458 5.167 5.875 6.583 7.292

frames  192   209   226   243   260   277   294   311   328   345  (362)
sec    8.000 8.708 9.417 10.125 10.833 11.542 12.250 12.958 13.667 14.375 (15.083)
```

Two consequences the current workflow silently eats:

1. **The expression always rounds up.** Ask for 6.000 s and you render 6.583 s. Over
   a 12-segment music video that is up to 8 s of accumulated drift against the track
   — every cut lands progressively later than the beat you planned it on.
2. **15 s is not reachable.** The ceiling under 15 is 345 frames = 14.375 s.

**The fix, and it is purely an orchestration fix:** the timeline stores two numbers
per segment.

- `target_duration` — the musical / narrative truth, a free float. What the cut
  actually needs to be.
- `render_frames` / `render_duration` — the next valid rung **at or above** target.

The sampler renders the rung. **The stitcher trims each clip back to
`target_duration`.** Sync is then exact and permanent, because every segment is cut
to plan rather than to whatever the ladder produced. The overshoot is at most
0.708 s of tail per segment, which is a rounding error of cost and a guarantee of
alignment.

This also feeds the prompter: it is told "the clip is 6.583 s, but every essential
action must complete by 6.000 s — the remainder is a hold." Timestamps stay legal
against the rendered length while the trim never cuts anything that matters.

### 2.4 Per-segment audio slicing — the music-video unlock

Your test graph loads the track with `LoadAudioUI` at `start_time 73.03 →
end_time 83.12` and feeds it to `ref_audios.ref_audio_0`. That is a hand-picked
10 s window of one song, for one clip.

A music video is that, N times, with **every segment taking a different window of
the same track** — and the windows have to be contiguous, gapless, and aligned to
the same clock the stitcher will use. Doing that by hand across 12 segments is
where the whole idea falls apart, so the pack owns it:

- **`H3 Track`** — load the song once, detect tempo and downbeats (librosa, already
  a dependency of the H3 pack), emit a `H3_TRACK` with the beat grid.
- The **Story Planner** lays segments onto that grid: `audio_start` /
  `audio_end` per segment, boundaries on downbeats, `target_duration` = the musical
  length, `render_frames` = the rung above it (§2.3).
- **`H3 Track Slice`** — takes `H3_TRACK` + the current `H3_SHOT`, returns the AUDIO
  for exactly that window, wired straight into `ref_audio_0`. It slices
  `render_duration` worth (not `target_duration`), so H3 hears audio for every frame
  it renders, and the stitcher's trim removes the overhang from both together.

It slices AUDIO rather than driving `LoadAudioUI`'s widgets, because widget values
are not connectable — this way it works regardless of which loader you use.

The payoff: lyrics land on the right segment, lip-sync has the right words in the
right window, cuts land on downbeats, and the assembled video is sample-accurate
against the original track because every segment was cut from the same timeline.

### 2.5 Auto-advance

The enqueue lives in **Vault Write**, not the dispatcher — only Vault Write knows the
run actually produced frames. It re-POSTs the captured graph (from the hidden
`PROMPT` input) to `/prompt`. Three guards, all necessary:

- stop when no segment is `pending`
- stop if the ComfyUI queue is non-empty, so manual batch-queueing and auto-advance
  can't compound into a runaway
- a `max_auto_runs` ceiling and a stop flag the UI can set mid-render

### 2.6 Chaining across runs

`link: continue` needs the last frame of segment *n* as the first frame of *n+1* —
across a process boundary, so it cannot be a tensor. Vault Write extracts the final
frame to a PNG and records `first_frame_path` on the *next* segment; the dispatcher
loads it. Same mechanism serves `H3 Retake` when you want a segment re-rendered
against a frame you picked by hand.

---

## 3. The reference cast — the part that makes it feel like Seedance

This is the hardest correctness problem in the pack, and the biggest reason to
build it rather than hand-write prompts.

H3 numbers references **by wired slot**: `<Picture 1>` is whatever is in input 1.
But a timeline has a *global* cast — hero, product, location plate, style ref,
song — and segment 5 might use only the hero and the product. In segment 5's own
call those become slots 1 and 2, so its prompt must say `<Subject 1>` and
`<Subject 2>` — different numbers than the same cast members carried in segment 2.

**`H3 Cast Board`** holds the global registry: each entry gets a stable key
(`hero`), a role (`character | product | location | style | motion | audio`), and a
retention default (`fully_preserved`, `attribute_transfer`, …). It also settles the
`<Subject>` vs `<Picture>` question once, globally — recurring identity is a
Subject; a concrete first/last frame is a Picture. Your existing pack already
enforces that distinction, and it is the difference between a consistent character
and background/pose leakage.

**`H3 Ref Binder`** sits between dispatcher and sampler and does the remap: takes
the segment's `cast_used`, packs those assets into slots 1..N in order, rewrites
every tag across all six prompt sections to the local numbering, and refuses the
run if a tag survives with no asset behind it. Nothing else in the pack is allowed
to touch tag numbers.

Consequence worth stating plainly: because the tag map is *derived*, you can
reorder segments in the timeline editor and every prompt stays correct.

---

## 4. Planning — idea to timeline

**`H3 Brief`** — idea, format (ad / music video / narrative / explainer / UGC),
target duration, language, tone, CTA, optional lyrics or script paste, optional
brand rules ("never show the cap off"). Plus the track when it's a music video.

**`H3 Story Planner`** — brief + cast + track → beats → segments. The planner works in
`target_duration` (free floats, whatever the beat needs); the ladder of §2.3 is
applied afterwards in code, never by the model. Duration is the planner's main
creative lever, not an arithmetic constraint:

- **Never pack to the ceiling.** `ceil(total / 14.375)` is the floor on segment
  count, not the plan. Segment length follows the beat: a 40 s ad is better as
  12 + 14 + 14 than 14 + 14 + 12 evenly, and better still as 3 + 11 + 12 + 14 when
  the hook is a 3-second stinger. The planner justifies each length by its beat.
- **Dialogue sets length.** A segment carrying a spoken line is sized to the line
  plus handles — measured from the script at ~2.6 words/second, not guessed. Get
  this right and `<cutoff>` is never needed; get it wrong and every third segment
  clips a word.
- **Music snaps to bars, not to the ladder.** With a track connected, boundaries land
  on detected downbeats and `target_duration` is a whole number of bars. The ladder
  overshoot is absorbed by the stitcher's trim (§2.3), so bar alignment is exact —
  which it could never be if the planner had to pick from 0.708 s rungs.
- **Floor of ~2 s.** Below that H3 has no room to establish a shot; the planner may
  still emit one as a deliberate stinger, but the timeline flags it.
- **Density check.** Roughly one internal `[Shot N]` per 3–4 s. A segment whose
  intent needs four cuts must be ≥ 12 s or get split — validated in code, because
  over-stuffing short segments is the most common way these prompts fail.
- The planner writes only the *skeleton*: beat, duration, link type, cast_used, and
  a one-line intent. It does not write H3 prose. Separating the two keeps the
  expensive call cheap to redo.

**`H3 Segment Prompter`** — walks the skeleton and writes the real H3 prompt for
each segment, reusing `VIDEO_SYSTEM` / `FULL_REF_SYSTEM` from the existing pack as
a library rather than a fork. Per segment it picks the task mode from what is
actually wired (T2VA / I2VA / FL2VA / L2VA, or the full-reference six-section form)
— the existing auto-mode logic, applied per segment.

**`H3 Continuity Pass`** — one call over the *whole* timeline after prompting.
This is what stops segment 6 putting the hero in a different jacket:

- one canonical wording per cast member, copied verbatim into every segment
- speaker IDs `(S1)`, `(S2)` stable across segments, not restarted per prompt
- `<scenetrans>` / `<cutoff>` inserted at boundaries where a line crosses a cut —
  H3's own mechanism for audio continuity, and the reason dialogue doesn't stutter
  at every stitch point
- identical style prefix everywhere ("Live-action, cinematic, …")
- validation: timestamps < duration, Shot 1 carries no timestamp, no orphan tags,
  ≤ 6 subjects, no duplicate subject definitions

Half of these are code checks, not LLM asks. Enforce in code what the model
reliably ignores — same discipline as the existing pack.

---

## 5. The timeline editor (the node people will actually love)

**`H3 Timeline`** — a JS widget node showing segment cards in a row: vault
thumbnail, duration bar, beat label, state chip, seed. Per card: edit prompt,
reorder by drag, lock, mute, redo, set duration, change link type, pick draft vs
final. The row total shows against target duration.

It is a real node with a real output (`H3_TIMELINE`), not a viewer — hand edits are
first-class and survive a re-plan of the untouched segments. `locked` segments are
never rewritten by the Continuity Pass.

Backed by `timeline.json` on disk beside the vault and served by aiohttp routes
(`/h3tl/timeline/get|save|reorder`), so the UI works without a queue run — the same
trick `clip_vault.js` already uses.

---

## 6. Draft → final passes (bookkeeping only)

Your graph already does both passes **in one run**: base sample at 0.3 MP →
`LTXVSeparateAVLatent` → `MinimaxH3LatentUpscaler3D` to 1.2 MP → re-concat → second
`SamplerCustomAdvanced` at ~0.85 denoise. So "how a segment reaches final res" is
solved and stays entirely yours. The pack's only job is to make the passes
*addressable* across a long timeline, so you are not paying for the upscale on all
twelve segments before you have seen the cut.

- `H3 Project` carries two megapixel targets and the pack tracks a current `pass`.
  The dispatcher emits `megapixels` and a `pass` string; you drive a bypass or
  switch on the upscale branch from it. That is the entire coupling.
- **Both seeds are fixed at plan time and never rolled at render time** — `seed` for
  node 129 and `refine_seed` for node 207. Rolling them is how a promoted segment
  stops matching the draft you approved; that is the pack's job to prevent, not
  the sampler's.
- `H3 Promote` takes a selection (`approved`, `failed`, `all`, or an index list),
  flips the pass and resets those segments to `pending`. The dispatcher then walks
  only those. Nothing else changes in the graph.
- Promotion **never overwrites the draft**. Both live in the vault as separate
  entries; `chosen` decides what stitches, so a promote is reversible.
- Optional dispatcher output: the draft clip path for the current segment, if you
  ever want to upscale from stored frames rather than re-running the base pass.

---

## 7. Vault — extend, don't replace

`AijigyasaClipVault` is already the right shape: manifest on disk, mp4 + wav, hash
dedup, survives restarts. What the timeline needs added:

- manifest entries carry `timeline_id`, `segment_id`, `pass` (draft|final), `seed`,
  `prompt_rev`
- retrieval by `(segment_id, pass)` instead of only "newest"
- variants: several takes per segment, one marked `chosen`
- a **thumbnail** per clip, so the timeline UI is not decoding mp4s to draw cards

Cleanest route: a `TimelineVault` in this pack writing the same layout plus the
extra keys, so existing vaults still open. `H3 Vault Write` is what the sampler
branch terminates into.

---

## 8. Assembly

**`H3 Stitcher`** — timeline + vault → one file. Works on **files via ffmpeg**, not
image tensors: a 3-minute 1080p timeline as IMAGE tensors is tens of GB of RAM,
which defeats the entire point. Concat demuxer when codec/res/fps match,
`filter_complex` only where a transition needs it.

- per-boundary transitions from the timeline (`cut` = hard, `continue` = none, plus
  dissolve / whip / dip-to-black where set)
- audio: H3 returns native audio per segment, so crossfade a few frames at each
  boundary; optional music bed with sidechain ducking under dialogue
- optional burned subtitles built from the `<d>[Language]…</d>` blocks — already
  parsed, already timed
- outputs the mp4 path plus a preview strip, never a full IMAGE batch

**`H3 Export`** — loudness normalise, aspect variants (16:9 / 9:16 / 1:1), and a
render report: per-segment seeds, prompts, durations, time and cost.

---

## 9. Low-VRAM rules the whole pack obeys

1. One segment per queue run. Peak VRAM is independent of timeline length.
2. Nothing holds an IMAGE batch it doesn't need — clips move as file paths.
3. Vault decode is on demand, and only for the segment being worked on.
4. Preview outputs are thumbnail strips, never full batches.
5. Timeline state lives on disk, so ComfyUI can restart mid-render and resume.
6. Draft at 0.2 MP is the default path; final res is opt-in per segment.

---

## 10. Node list (v1)

| # | Node | In → Out |
|---|---|---|
| 1 | H3 Project | — → `H3_PROJECT` |
| 2 | H3 Cast Board | IMAGE×N, VIDEO, AUDIO → `H3_CAST` |
| 2b | H3 Track | AUDIO → `H3_TRACK` (tempo, downbeat grid) |
| 3 | H3 Brief | text, `H3_TRACK` → `H3_BRIEF` |
| 4 | H3 Story Planner | brief + cast + project → `H3_TIMELINE` (skeleton) |
| 5 | H3 Segment Prompter | timeline → `H3_TIMELINE` (prompted) |
| 6 | H3 Continuity Pass | timeline → `H3_TIMELINE` (validated) |
| 7 | H3 Timeline | timeline → `H3_TIMELINE` (edited, UI node) |
| 8 | H3 Shot Dispatcher | timeline → `H3_SHOT` + the primitives of §2.2 |
| 9 | H3 Ref Binder | `H3_SHOT` + cast → IMAGE slots + retagged prompt |
| 9b | H3 Track Slice | `H3_TRACK` + `H3_SHOT` → AUDIO for this segment's window |
| 10 | H3 Vault Write | IMAGE, AUDIO, `H3_SHOT` → timeline (state updated) |
| 11 | H3 Promote | timeline + selection → filtered timeline |
| 12 | H3 Retake | timeline + index + note → timeline |
| 13 | H3 Stitcher | timeline → video path, preview strip |
| 14 | H3 Export | video path → final files + report |
| 15 | H3 Timeline Save / Load | disk ↔ `H3_TIMELINE` |

Build order: **1, 2, 8, 10, 13 first** — the smallest set that renders and stitches
a hand-written timeline end to end. Planning and UI go on top of a spine that
already works.

---

## 11. Settled / open

**Settled.** H3 is local; sampling stays outside the pack, and §2.2 is the measured
seam against `video_minimax_h3_r2v_test.json`. Length is the 17-frame ladder of
§2.3, not a free float — absorbed by target/render split plus trim-on-stitch. Both
passes already live in one graph (§6), so the pack only flips megapixels and a
bypass signal. No negative prompt anywhere. Draft 0.3 MP → final 1.2 MP, 24 fps.
The pack gets **its own `TimelineVault`**, standalone.

**Open.**

1. Reuse the H3 Prompt Creator engine by import (needs both packs installed), or
   vendor the system prompts? Import is cleaner, vendoring is more portable.
2. Longest realistic target — a 60 s ad, or a 3–5 minute music video? Drives whether
   the stitcher needs an intermediate-concat strategy.
3. Is the ladder rule `≡ 5 (mod 17)` fixed for H3, or does it shift with the VAE /
   turbo LoRA? The pack derives it from a widget so it can be changed, but a wrong
   value fails at the sampler rather than in the pack.
4. Does `ref_images.ref_image_N` accept more than a handful of slots, and does the
   4-step turbo LoRA hold identity across 6 references? Caps the practical cast size.
