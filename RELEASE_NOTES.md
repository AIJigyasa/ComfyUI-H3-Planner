# Release notes

## v0.2.0

### New
- **Segment Prompter: plans the whole video from an idea.** One planning pass fixes the cast (`<Subject N>` = who, from which picture, with looks read from the image) and what happens in each segment, before any segment is written.
- **The model now sees your reference pictures** when planning and when writing each segment.
- **Timing from your idea.** "First 28 seconds instrumental" becomes a vocal timeline: those clips keep every mouth closed, and a clip that crosses into the vocals is told exactly when singing starts.
- **One named performer.** Only the planned singer or rapper lip-syncs; everyone else just plays.
- **Picture jobs follow Cast Board roles.** A character sheet is used only for looks, never as a location or opening frame; the stage or location picture sets every shot.
- **Runtime is always honoured.** Segments add up to exactly `total_seconds`, and none is longer than `segment_seconds`. A treatment that ends early is continued; an overlong shot is split.
- **`shots_per_segment` works without a treatment** (the most shots per clip, each at least 1.5 s).
- **Cast Board:** every image is tagged `<Picture N>`; the people and objects inside are `<Subject N>`, as many as the analysis finds.
- **Report:** shows the plan, which pictures the model saw, the vocal timing, and a REPAIRED list of everything the code corrected.

### Fixed
- **Pictures described the wrong way round.** Tag renumbering in the formatting step turned the character sheet into `<Picture 1>` and the stage into `<Picture 2>`. Numbers are now kept exactly.
- **Subjects renumbered.** A segment defining `<Subject 2>` and `<Subject 4>` came out defining 1 and 2.
- **Summaries were "N/A"** in about half the segments. An undefined subject now keeps its sentence, and empty sections are asked for again.
- **Only segment 1's cast definitions reached the other segments**, so later segments cited subjects nobody defined.
- **Refined prompts were overwritten** on the next queue. Edits on a Timeline card now stick unless the planner genuinely rewrites that segment.
- **Refine had no model settings to use** for Segment Prompter timelines.
- **Placeholder dialogue** like `<d>'[lyrics]'</d>` is removed, and a summary no longer claims a video or keyframe task you didn't connect.
- **Stitched videos drifted out of sync with the audio.** Clips are now cut to whole frames against a running clock, so there is no drift across cuts.

### Upgrading
- Restart ComfyUI. Idea-mode prompts are rewritten once on the next run; after that, turn `reuse_existing` on.
- `shots_per_segment` = 1 now means one continuous shot per clip in idea mode.

## v0.1.0

First release.
