import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * H3 Timeline — a card strip instead of a JSON textarea.
 *
 * Authored data (id, beat, duration, prompt, audio_start, link) lives in the
 * node's `timeline_json` widget, so it saves and travels with the workflow.
 * Runtime state (clips, takes, thumbnails, failures) is pulled from disk, so
 * the strip shows what has actually rendered while you are still editing.
 * The two never fight: the UI only writes authored fields.
 */

const PASSES = ["draft", "final"];

const CSS = `
.h3p-wrap { display:flex; flex-direction:column; gap:6px; height:100%;
  font-family: system-ui, sans-serif; font-size:11px; color:#ddd; }
.h3p-bar { display:flex; gap:4px; align-items:center; flex-wrap:wrap;
  padding:2px 2px 0 2px; }
.h3p-bar .sp { flex:1 1 auto; }
.h3p-btn { background:#333; border:1px solid #555; color:#ddd; border-radius:4px;
  padding:2px 7px; cursor:pointer; font-size:11px; line-height:16px; }
.h3p-btn:hover { background:#444; border-color:#777; }
.h3p-btn.on { background:#2d4f6b; border-color:#4a86b8; }
.h3p-total { opacity:.75; font-variant-numeric:tabular-nums; }
/* Card width follows the video's shape. A fixed 168px card put a 9:16 clip in
   a landscape box, so the strip read as a row of wide cards with a small
   picture in each. --card-w and --aspect are set from the project's real
   width and height; the thumb then takes the frame's own proportions. */
.h3p-strip { display:flex; gap:6px; overflow-x:auto; overflow-y:hidden;
  padding:2px 2px 6px 2px; flex:0 0 auto; min-height:186px;
  align-items:stretch; scrollbar-width:thin; }
/* Fit: wrap instead of scrolling sideways, so a long timeline is visible at
   once rather than a scrollbar's worth at a time. */
.h3p-strip.fit { display:grid; overflow-x:hidden; overflow-y:auto;
  grid-template-columns:repeat(auto-fill, minmax(var(--card-w, 168px), 1fr));
  align-content:start; flex:1 1 auto; }
.h3p-card { flex:0 0 var(--card-w, 168px); min-width:0;
  display:flex; flex-direction:column; gap:3px;
  background:#282828; border:1px solid #3d3d3d; border-radius:6px; padding:5px;
  cursor:pointer; position:relative; }
.h3p-strip.fit .h3p-card { flex:1 1 auto; }
.h3p-card.sel { border-color:#5b9bd5; background:#2c3742; }
.h3p-card.locked { opacity:.62; }
.h3p-head { display:flex; justify-content:space-between; align-items:center;
  gap:4px; }
.h3p-id { font-weight:600; color:#fff; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.h3p-chip { font-size:9px; padding:1px 5px; border-radius:8px; flex:0 0 auto;
  text-transform:uppercase; letter-spacing:.4px; }
/* contain, not cover: a 9:16 clip in a landscape box was being cropped down
   its middle, so the card showed a slice of the shot rather than the shot. */
.h3p-thumb { width:100%; aspect-ratio:var(--aspect, 16 / 9); height:auto;
  object-fit:contain; border-radius:4px; background:#1b1b1b; display:block; }
.h3p-thumb.empty { display:flex; align-items:center; justify-content:center;
  color:#666; font-size:10px; }
.h3p-beat { color:#9ac; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.h3p-meta { display:flex; justify-content:space-between; opacity:.7;
  font-variant-numeric:tabular-nums; font-size:10px; }
.h3p-prompt { color:#aaa; font-size:10px; line-height:1.25; max-height:26px;
  overflow:hidden; }
.h3p-err { color:#e08a8a; font-size:10px; }
.h3p-edit { display:flex; flex-direction:column; gap:4px; min-height:0;
  flex:1 1 auto; border-top:1px solid #3d3d3d; padding-top:5px; }
.h3p-row { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
.h3p-row label { opacity:.65; }
.h3p-in { background:#1e1e1e; border:1px solid #444; color:#ddd; border-radius:3px;
  padding:2px 4px; font-size:11px; font-family:inherit; }
.h3p-in.num { width:62px; text-align:right; font-variant-numeric:tabular-nums; }
.h3p-in.txt { flex:1 1 90px; min-width:70px; }
.h3p-ta { background:#1e1e1e; border:1px solid #444; color:#ddd; border-radius:4px;
  padding:5px; font-size:11px; font-family:ui-monospace,Consolas,monospace;
  resize:none; flex:1 1 auto; min-height:60px; width:100%;
  box-sizing:border-box; line-height:1.35; }
.h3p-note { opacity:.6; font-size:10px; }
`;

/* -------------------------------------------------------------- helpers */

function ensureStyle() {
  if (document.getElementById("h3p-style")) return;
  const el = document.createElement("style");
  el.id = "h3p-style";
  el.textContent = CSS;
  document.head.appendChild(el);
}

const STATE_COLORS = {
  pending: ["#3a3a3a", "#bbb"],
  running: ["#5a4a1e", "#f0d071"],
  draft:   ["#254a2c", "#8fd39c"],
  final:   ["#1e4a55", "#7fd0e0"],
  failed:  ["#5a2626", "#f09b9b"],
  locked:  ["#3d3252", "#c0aae0"],
};

function chip(state) {
  const el = document.createElement("span");
  el.className = "h3p-chip";
  const [bg, fg] = STATE_COLORS[state] || STATE_COLORS.pending;
  el.style.background = bg;
  el.style.color = fg;
  el.textContent = state || "pending";
  return el;
}

function button(label, title, onClick) {
  const b = document.createElement("button");
  b.className = "h3p-btn";
  b.textContent = label;
  if (title) b.title = title;
  b.onclick = (e) => { e.stopPropagation(); onClick(e); };
  return b;
}

// Flatten a prompt (string, or the six named sections) for the card preview.
function promptPreview(prompt) {
  if (typeof prompt === "string") return prompt;
  if (prompt && typeof prompt === "object") {
    return prompt.detailed_description || prompt.summary ||
      Object.values(prompt).find((v) => typeof v === "string" && v.trim()) || "";
  }
  return "";
}

function promptText(prompt) {
  if (typeof prompt === "string") return prompt;
  if (prompt && typeof prompt === "object") return JSON.stringify(prompt, null, 1);
  return "";
}

// Round-trip the editor content back into whichever shape it came from.
function parsePrompt(text, wasObject) {
  if (wasObject) {
    try {
      const obj = JSON.parse(text);
      if (obj && typeof obj === "object" && !Array.isArray(obj)) return obj;
    } catch (e) { /* fall through — keep it as a plain string */ }
  }
  return text;
}

function thumbUrl(info) {
  if (!info) return null;
  const q = new URLSearchParams({
    filename: info.filename,
    subfolder: info.subfolder || "",
    type: info.type || "output",
  });
  // Re-rendering a segment reuses the same filename, so without this the
  // browser keeps serving the previous clip out of cache and the card looks
  // like nothing changed.
  if (info.v) q.set("h3v", String(info.v));
  return api.apiURL("/view?" + q.toString());
}

/* --------------------------------------------------------------- the UI */

class TimelineStrip {
  constructor(node, jsonWidget) {
    this.node = node;
    this.jsonWidget = jsonWidget;
    this.selected = 0;
    this.showJson = false;
    this.pass = "draft";
    this.live = {};          // segment id -> runtime info from disk
    this.doc = { segments: [] };

    this.root = document.createElement("div");
    this.root.className = "h3p-wrap";
    this.bar = document.createElement("div");
    this.bar.className = "h3p-bar";
    this.strip = document.createElement("div");
    this.strip.className = "h3p-strip";
    this.editor = document.createElement("div");
    this.editor.className = "h3p-edit";
    this.root.append(this.bar, this.strip, this.editor);

    this.read();
    this.render();
    this.refresh();
  }

  /* Card shape follows the video. Keeping the thumbnail area roughly constant
     means a 9:16 card is tall and narrow and a 16:9 card is short and wide,
     which is what makes a strip of them readable at a glance. */
  applyAspect(width, height) {
    const ratio = width / height;
    if (!isFinite(ratio) || ratio <= 0) return;
    const area = 132 * 132;
    const w = Math.round(Math.min(240, Math.max(104, Math.sqrt(area * ratio))));
    this.root.style.setProperty("--aspect", `${width} / ${height}`);
    this.root.style.setProperty("--card-w", `${w}px`);
  }

  applyFit() {
    this.strip.classList.toggle("fit", !!this.fitAll);
    for (const b of this.bar.querySelectorAll(".h3p-btn")) {
      if (b.textContent === "Fit") b.classList.toggle("on", !!this.fitAll);
    }
    this.node?.setDirtyCanvas(true, true);
  }
  /* ---- authored JSON <-> cards ---- */

  read() {
    let raw = this.jsonWidget?.value ?? "";
    try {
      const parsed = JSON.parse(raw);
      this.doc = Array.isArray(parsed) ? { segments: parsed } : (parsed || {});
      if (!Array.isArray(this.doc.segments)) this.doc.segments = [];
      this.parseError = null;
    } catch (e) {
      this.parseError = e.message;
      this.doc = { segments: [] };
    }
    return this.doc;
  }

  write() {
    if (!this.jsonWidget) return;
    this.jsonWidget.value = JSON.stringify(this.doc, null, 1);
    // keep the graph dirty so the change is saved with the workflow
    this.node.graph?.setDirtyCanvas(true, true);
    if (this.jsonWidget.callback) {
      try { this.jsonWidget.callback(this.jsonWidget.value); } catch (e) {}
    }
  }

  seg(i) { return this.doc.segments[i]; }
  duration(s) {
    const v = s.target_duration ?? s.duration ?? s.seconds;
    return Number.isFinite(+v) ? +v : 0;
  }

  /* ---- live state ---- */

  projectName() {
    // Three sources, best first. The name has to be right BEFORE the first
    // run of a reopened workflow: the server used to guess the first project
    // on disk when asked without one, so a strip that did not yet know its
    // own name came back showing another project's clips entirely.
    if (this.project) return this.project;

    // The wired H3 Project node is the truth, through any reroutes.
    try {
      const slot = this.node.inputs?.findIndex((i) => i.name === "project");
      if (slot >= 0) {
        let src = this.node.getInputNode(slot);
        for (let hop = 0; src && hop < 8; hop++) {
          const w = src.widgets?.find((x) => x.name === "project_name");
          if (w?.value) return String(w.value);
          // Reroute and friends carry no widgets; keep walking back.
          const next = src.getInputNode?.(0);
          if (!next || next === src) break;
          src = next;
        }
      }
    } catch (e) { /* no name is better than the wrong name */ }

    // Failing that, the authored JSON in the widget carries the name it
    // was written for, and it is there the moment the workflow loads.
    // Only a default example block can be wrong here, and a name that
    // matches no project on disk shows nothing rather than the wrong thing.
    try {
      const named = this.doc?.name || this.doc?.project;
      if (named) return String(named);
    } catch (e) { /* fall through */ }

    return "";
  }

  async refresh() {
    try {
      const name = this.projectName();
      if (!name) {
        // Asking without a name can only produce someone else's timeline.
        // The authored prompts still render; the clips simply wait.
        this.live = {};
        this.liveProject = "";
        this.render();
        return;
      }
      const q = new URLSearchParams({ project: name });
      const resp = await api.fetchApi("/h3planner/timeline?" + q.toString());
      const data = await resp.json();
      this.live = {};
      for (const s of data.segments || []) this.live[s.id] = s;
      this.liveProject = data.project || "";
      this.render();
    } catch (e) {
      // No server state yet is normal — the strip still edits fine.
    }
  }

  // The node hands back the authored JSON it actually used. Without this the
  // cards stay empty whenever the timeline arrives through the input wire,
  // because they read the widget and nothing had written to it.
  adopt(authoredJson, fromInput, project, pass, width, height) {
    if (project) this.project = project;
    if (pass) this.pass = pass;
    if (width > 0 && height > 0) this.applyAspect(width, height);
    if (!authoredJson || !this.jsonWidget) return;
    if (this.jsonWidget.value === authoredJson) { this.refresh(); return; }
    this.jsonWidget.value = authoredJson;
    this.fromInput = !!fromInput;
    this.read();
    this.render();
    this.refresh();
  }

  async mutate(segId, action, extra) {
    const project = this.projectName() || this.liveProject;
    if (!project) {
      alert("No project yet — queue the graph once so the node knows which "
            + "project it belongs to.");
      return null;
    }
    try {
      const res = await api.fetchApi("/h3planner/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(Object.assign(
          { project, segment_id: segId || "", action, pass: this.pass },
          extra || {})),
      });
      const data = await res.json().catch(() => null);
      // A refusal comes back as 400 with a reason. Swallowing it left the
      // card looking unchanged with no clue why.
      if (data && data.error) {
        alert("H3 Planner: " + data.error);
        return null;
      }
      await this.refresh();
      return data;
    } catch (e) {
      console.error("[H3Planner]", e);
      return null;
    }
  }

  /* ---- editing ---- */

  nextId() {
    const used = new Set(this.doc.segments.map((s) => s.id));
    for (let i = 1; i < 999; i++) {
      const id = "s" + i;
      if (!used.has(id)) return id;
    }
    return "s" + Date.now();
  }

  add(afterIndex) {
    const src = this.seg(afterIndex);
    const seg = {
      id: this.nextId(),
      beat: "",
      duration: src ? this.duration(src) : 6.0,
      audio_start: null,
      link: "cut",
      prompt: "",
    };
    const at = afterIndex == null ? this.doc.segments.length : afterIndex + 1;
    this.doc.segments.splice(at, 0, seg);
    this.selected = at;
    this.write();
    this.render();
  }

  duplicate(i) {
    const copy = JSON.parse(JSON.stringify(this.seg(i)));
    copy.id = this.nextId();
    delete copy.seed;
    delete copy.refine_seed;
    this.doc.segments.splice(i + 1, 0, copy);
    this.selected = i + 1;
    this.write();
    this.render();
  }

  remove(i) {
    const seg = this.seg(i);
    if (!confirm(`Delete segment ${seg.id}?\n\nStored clips stay in the vault.`))
      return;
    this.doc.segments.splice(i, 1);
    this.selected = Math.max(0, Math.min(this.selected, this.doc.segments.length - 1));
    this.write();
    this.render();
  }

  move(i, delta) {
    const j = i + delta;
    if (j < 0 || j >= this.doc.segments.length) return;
    const [seg] = this.doc.segments.splice(i, 1);
    this.doc.segments.splice(j, 0, seg);
    this.selected = j;
    this.write();
    this.render();
  }

  /* ---- rendering ---- */

  render() {
    this.renderBar();
    this.renderStrip();
    this.renderEditor();
  }

  renderBar() {
    this.bar.replaceChildren();
    const n = this.doc.segments.length;
    const total = this.doc.segments.reduce((a, s) => a + this.duration(s), 0);

    this.bar.append(button("+ Segment", "Append a new segment", () => this.add(null)));

    for (const p of PASSES) {
      const b = button(p, `Show ${p} clips on the cards`, () => {
        this.pass = p; this.render();
      });
      if (this.pass === p) b.classList.add("on");
      this.bar.append(b);
    }

    const sp = document.createElement("span");
    sp.className = "sp";
    this.bar.append(sp);

    const info = document.createElement("span");
    info.className = "h3p-total";
    const done = this.doc.segments.filter(
      (s) => this.live[s.id]?._vault?.[this.pass]).length;
    info.textContent = `${n} segment${n === 1 ? "" : "s"} · ${total.toFixed(2)}s · ${done}/${n} ${this.pass}`
      + (this.fromInput ? " · fed by an input" : "");
    info.title = this.fromInput
      ? "The timeline input is connected, so the upstream node owns these "
        + "prompts and will overwrite hand edits on the next queue. Lock a "
        + "segment to protect it."
      : "";
    this.bar.append(info);

    const fit = button("Fit", "Wrap the cards so the whole timeline is "
                       + "visible at once, instead of scrolling sideways",
                       () => {
      this.fitAll = !this.fitAll;
      this.applyFit();
    });
    if (this.fitAll) fit.classList.add("on");
    this.bar.append(fit);
    this.bar.append(button("Refresh", "Re-read render state from disk",
                           () => this.refresh()));
    this.bar.append(button("Redo all", `Clear every stored ${this.pass} clip so the whole timeline renders again. Prompts and seeds are kept; locked cards are left alone.`, () => {
      const n = this.doc.segments.length;
      if (!confirm(`Re-render all ${n} segment(s)?

Every stored ${this.pass} clip is cleared and the timeline goes back to pending. Prompts, seeds and locked cards are untouched.`))
        return;
      this.mutate(null, "redo_all");
    }));
    const j = button("JSON", "Show the raw timeline JSON widget", () => {
      this.showJson = !this.showJson;
      this.applyJsonVisibility();
      this.render();
    });
    if (this.showJson) j.classList.add("on");
    this.bar.append(j);
  }

  renderStrip() {
    this.strip.replaceChildren();

    if (this.parseError) {
      const err = document.createElement("div");
      err.className = "h3p-err";
      err.textContent = "timeline_json is not valid JSON: " + this.parseError +
        " — fix it in the JSON view.";
      this.strip.append(err);
      return;
    }

    this.doc.segments.forEach((seg, i) => {
      const live = this.live[seg.id] || {};
      const vaultInfo = live._vault?.[this.pass];
      const card = document.createElement("div");
      card.className = "h3p-card" + (i === this.selected ? " sel" : "") +
        (live.state === "locked" ? " locked" : "");
      card.onclick = () => { this.selected = i; this.render(); };

      const head = document.createElement("div");
      head.className = "h3p-head";
      const id = document.createElement("span");
      id.className = "h3p-id";
      id.textContent = `${i + 1}. ${seg.id ?? "?"}`;
      head.append(id, chip(live.state || "pending"));
      card.append(head);

      const url = thumbUrl(vaultInfo?.thumb);
      if (url) {
        const img = document.createElement("img");
        img.className = "h3p-thumb";
        img.src = url;
        img.loading = "lazy";
        img.title = "Click to play the rendered clip";
        img.onclick = (e) => {
          e.stopPropagation();
          const clip = thumbUrl(vaultInfo?.clip);
          if (!clip) return;
          const v = document.createElement("video");
          v.className = "h3p-thumb";
          v.style.objectFit = "cover";   // match the thumbnail exactly
          v.playsInline = true;
          v.muted = false;
          v.src = clip;
          v.controls = true;
          v.autoplay = true;
          v.loop = true;
          img.replaceWith(v);
        };
        card.append(img);
      } else {
        const ph = document.createElement("div");
        ph.className = "h3p-thumb empty";
        ph.textContent = "not rendered";
        card.append(ph);
      }

      if (seg.beat) {
        const beat = document.createElement("div");
        beat.className = "h3p-beat";
        beat.textContent = seg.beat;
        card.append(beat);
      }

      const meta = document.createElement("div");
      meta.className = "h3p-meta";
      const planned = this.duration(seg);
      const rendered = live.render_duration;
      meta.append(
        Object.assign(document.createElement("span"),
                      { textContent: planned.toFixed(2) + "s" }),
        Object.assign(document.createElement("span"), {
          textContent: rendered ? "→ " + Number(rendered).toFixed(2) + "s"
                                : (seg.link === "continue" ? "↩ continue" : ""),
          title: rendered ? "H3 rendered this much; the stitcher trims it back"
                          : "",
        }));
      card.append(meta);

      const preview = document.createElement("div");
      preview.className = "h3p-prompt";
      preview.textContent = promptPreview(seg.prompt).slice(0, 140);
      card.append(preview);

      if (live.last_error) {
        const err = document.createElement("div");
        err.className = "h3p-err";
        err.textContent = live.last_error.slice(0, 80);
        card.append(err);
      }

      this.strip.append(card);
    });

    const tail = document.createElement("div");
    tail.className = "h3p-card";
    tail.style.flex = "0 0 44px";
    tail.style.alignItems = "center";
    tail.style.justifyContent = "center";
    tail.style.fontSize = "20px";
    tail.style.color = "#777";
    tail.textContent = "+";
    tail.title = "Add a segment at the end";
    tail.onclick = () => this.add(null);
    this.strip.append(tail);
  }

  renderEditor() {
    this.editor.replaceChildren();
    const seg = this.seg(this.selected);
    if (!seg) {
      const note = document.createElement("div");
      note.className = "h3p-note";
      note.textContent = "No segments yet — press + Segment.";
      this.editor.append(note);
      return;
    }
    const live = this.live[seg.id] || {};

    const row1 = document.createElement("div");
    row1.className = "h3p-row";

    const mk = (label, value, cls, onChange, title) => {
      const l = document.createElement("label");
      l.textContent = label;
      const inp = document.createElement("input");
      inp.className = "h3p-in " + cls;
      inp.value = value ?? "";
      if (title) inp.title = title;
      inp.onchange = () => { onChange(inp.value); this.write(); this.render(); };
      row1.append(l, inp);
      return inp;
    };

    mk("id", seg.id, "txt", (v) => { seg.id = v.trim() || seg.id; });
    mk("beat", seg.beat ?? "", "txt", (v) => { seg.beat = v; });
    mk("sec", this.duration(seg), "num", (v) => {
      const n = parseFloat(v);
      if (Number.isFinite(n) && n > 0) {
        delete seg.duration; delete seg.seconds;
        seg.target_duration = n;
      }
    }, "What the cut needs. H3 rounds up to the next legal length; the stitcher trims back.");
    mk("audio@", seg.audio_start ?? "", "num", (v) => {
      const n = parseFloat(v);
      seg.audio_start = Number.isFinite(n) ? n : null;
    }, "Where this segment starts in the reference track, in seconds.");

    const linkSel = document.createElement("select");
    linkSel.className = "h3p-in";
    for (const opt of ["cut", "continue", "match"]) {
      const o = document.createElement("option");
      o.value = o.textContent = opt;
      if ((seg.link || "cut") === opt) o.selected = true;
      linkSel.append(o);
    }
    linkSel.onchange = () => { seg.link = linkSel.value; this.write(); this.render(); };
    const ll = document.createElement("label");
    ll.textContent = "link";
    row1.append(ll, linkSel);
    this.editor.append(row1);

    const row2 = document.createElement("div");
    row2.className = "h3p-row";
    row2.append(
      button("◀", "Move earlier", () => this.move(this.selected, -1)),
      button("▶", "Move later", () => this.move(this.selected, 1)),
      button("Duplicate", "Copy this segment (new id, fresh seeds)",
             () => this.duplicate(this.selected)),
      button("Delete", "Remove this segment", () => this.remove(this.selected)),
    );
    const sp2 = document.createElement("span");
    sp2.className = "sp";
    row2.append(sp2);
    row2.append(
      button("Redo", `Clear the stored ${this.pass} clip so it renders again`,
             () => this.mutate(seg.id, "redo")),
      button(live.state === "locked" ? "Unlock" : "Lock",
             "A locked segment is never claimed by the dispatcher",
             () => this.mutate(seg.id, live.state === "locked" ? "unlock" : "lock")),
    );
    if (live.state === "running") {
      row2.append(button("Reset", "Clear a stuck running state",
                         () => this.mutate(seg.id, "reset_running")));
    }
    this.editor.append(row2);

    // Say what is wrong in plain words and only this segment's shot
    // description is rewritten; cast, tags, dialogue and soundscape are
    // copied across untouched.
    const noteInput = document.createElement("input");
    noteInput.className = "h3p-in txt";
    noteInput.placeholder = "what should change in this shot? e.g. \"make it a "
      + "low angle, she walks away from camera\"";
    noteInput.value = seg.refine_note || "";
    const refineRow = document.createElement("div");
    refineRow.className = "h3p-row";
    const refineBtn = button("Refine", "Rewrite only this segment's shot "
      + "description from your note. Everything else — the cast wording, "
      + "the tags, the spoken line, the soundscape — is kept exactly as "
      + "it is. The stored clip is discarded, so it renders again.",
      async () => {
        const text = noteInput.value.trim();
        if (!text) { noteInput.focus(); return; }
        refineBtn.disabled = true;
        refineBtn.textContent = "Refining…";
        const out = await this.mutate(seg.id, "refine", { note: text });
        refineBtn.disabled = false;
        refineBtn.textContent = "Refine";
        if (out && out.prompt) {
          seg.prompt = out.prompt;
          seg.refine_note = text;
          this.write();
          this.render();
        }
      });
    refineRow.append(noteInput, refineBtn);
    this.editor.append(refineRow);

    const wasObject = seg.prompt && typeof seg.prompt === "object";
    const ta = document.createElement("textarea");
    ta.className = "h3p-ta";
    ta.spellcheck = false;
    ta.value = promptText(seg.prompt);
    ta.placeholder = wasObject
      ? "six H3 sections as JSON"
      : "H3 prompt for this segment";
    ta.onchange = () => {
      seg.prompt = parsePrompt(ta.value, wasObject);
      this.write();
      this.renderStrip();
    };
    this.editor.append(ta);

    const note = document.createElement("div");
    note.className = "h3p-note";
    const bits = [];
    if (live.seed) bits.push("seed " + live.seed);
    if (live.render_frames) bits.push(live.render_frames + " frames");
    const v = live._vault?.[this.pass];
    if (v) bits.push(`${this.pass} take ${v.take} · ${v.width}x${v.height}` +
                     (v.has_audio ? " · audio" : " · silent"));
    note.textContent = bits.join("  ·  ") ||
      "Not rendered yet — queue the graph once per segment.";
    this.editor.append(note);
  }

  applyJsonVisibility() {
    const el = this.jsonWidget?.element ||
      this.jsonWidget?.inputEl || null;
    if (el) el.style.display = this.showJson ? "" : "none";
    if (this.jsonWidget) {
      this.jsonWidget.computeSize = this.showJson
        ? undefined
        : () => [0, -4];  // collapse the row when hidden
    }
    this.node.setDirtyCanvas(true, true);
  }
}

/* ------------------------------------------------------------ extension */

app.registerExtension({
  name: "h3planner.timeline.cards",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "H3PlannerTimeline") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      ensureStyle();
      try {
        const jsonWidget = this.widgets?.find((w) => w.name === "timeline_json");
        const strip = new TimelineStrip(this, jsonWidget);
        this.h3pStrip = strip;

        const widget = this.addDOMWidget("h3p_cards", "div", strip.root,
                                         { serialize: false, hideOnZoom: false });

        // Take whatever height the node has left after the ordinary widgets,
        // so dragging the node bigger actually shows more. A fixed height is
        // what made this unreadable once a prompt was in it.
        const MIN_BODY = 300;
        const node = this;
        widget.computeSize = function (width) {
          let used = 0;
          for (const w of node.widgets ?? []) {
            if (w === widget) continue;
            const h = w.computeSize ? w.computeSize(width)[1]
                                    : (LiteGraph.NODE_WIDGET_HEIGHT ?? 20);
            used += Math.max(0, h) + 4;
          }
          const slots = Math.max(node.inputs?.length ?? 0,
                                 node.outputs?.length ?? 0);
          const chrome = (LiteGraph.NODE_SLOT_HEIGHT ?? 20) * slots + 30;
          return [width, Math.max(MIN_BODY, node.size[1] - used - chrome)];
        };

        const minH = MIN_BODY + 190;
        this.size = [Math.max(this.size[0], 640), Math.max(this.size[1], minH)];
        const onResize = this.onResize;
        this.onResize = function (size) {
          size[0] = Math.max(size[0], 460);
          size[1] = Math.max(size[1], minH);
          onResize?.call(this, size);
        };
        strip.applyJsonVisibility();
      } catch (e) {
        console.error("[H3Planner] card strip failed, JSON widget still works", e);
      }
      return r;
    };

    // Re-read the widget after a workflow load, and pull fresh render state.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        try {
          this.h3pStrip?.read();
          this.h3pStrip?.render();
          this.h3pStrip?.refresh();
        } catch (e) {}
      }, 60);
      return r;
    };
  },

  // After a run, the vault has new clips — refresh every strip on the canvas.
  setup() {
    // Refresh while anything is queued. Relying only on node events meant a
    // clip landed in the vault and the card kept showing "not rendered" until
    // Refresh was pressed by hand.
    let busy = false;
    let timer = null;
    const refreshAll = () => {
      for (const node of app.graph?._nodes ?? []) {
        if (node.h3pStrip) node.h3pStrip.refresh();
      }
    };
    api.addEventListener("status", (event) => {
      const left = event?.detail?.exec_info?.queue_remaining ?? 0;
      if (left > 0 && !busy) {
        busy = true;
        timer = setInterval(refreshAll, 2500);
      } else if (left === 0 && busy) {
        busy = false;
        clearInterval(timer);
        timer = null;
        setTimeout(refreshAll, 400);   // catch the last write
      }
    });
    api.addEventListener("execution_success", refreshAll);
    api.addEventListener("execution_error", refreshAll);

    api.addEventListener("executed", (event) => {
      const payload = event?.detail?.output?.h3_timeline?.[0];
      const target = app.graph?.getNodeById?.(event?.detail?.node);
      if (payload && target?.h3pStrip) {
        target.h3pStrip.adopt(payload.authored, payload.from_input,
                              payload.project, payload.pass,
                              payload.width, payload.height);
      }
      for (const node of app.graph?._nodes ?? []) {
        if (node.h3pStrip && node !== target) node.h3pStrip.refresh();
      }
    });
  },
});
