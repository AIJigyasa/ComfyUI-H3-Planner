import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * H3 Cast Board — drop references straight onto the node.
 *
 * Files upload to ComfyUI's input/h3_planner/ and the card list is stored in
 * the node's `cast_json` widget, so a saved workflow reopens with its cast
 * intact.
 *
 * Tags are derived, never typed. The H3 guide is explicit that a character or
 * product image is NOT automatically <Picture N> — that is reserved for a
 * concrete frame anchor — so the role decides the kind, and the kinds are
 * numbered independently.
 */

const ROLE_TAGS = {
  character: "Subject", product: "Subject", style: "Subject",
  wardrobe: "Subject", environment: "Subject", prop: "Subject",
  first_frame: "Picture", last_frame: "Picture",
  keyframe: "Picture", composition: "Picture",
  video: "Video", audio: "Audio",
};
const ROLES = Object.keys(ROLE_TAGS);
const IMAGE_ROLES = ROLES.filter((r) => ["Subject", "Picture"].includes(ROLE_TAGS[r]));
// Mirrors MAX_IMAGE_SLOTS in nodes_cast.py. The card footer promises a
// card an image_N output, so the two have to agree or the promise lies.
const MAX_IMAGE_SLOTS = 8;

const CSS = `
.h3c-wrap { display:flex; flex-direction:column; gap:6px; height:100%;
  font-family:system-ui,sans-serif; font-size:11px; color:#ddd; }
.h3c-bar { display:flex; gap:4px; align-items:center; flex-wrap:wrap; }
.h3c-bar .sp { flex:1 1 auto; }
.h3c-btn { background:#333; border:1px solid #555; color:#ddd; border-radius:4px;
  padding:2px 7px; cursor:pointer; font-size:11px; line-height:16px; }
.h3c-btn:hover { background:#444; border-color:#777; }
.h3c-btn.on { background:#2d4f6b; border-color:#4a86b8; }
.h3c-drop { border:1px dashed #555; border-radius:6px; padding:10px;
  text-align:center; color:#888; cursor:pointer; flex:0 0 auto; }
.h3c-drop.hot { border-color:#5b9bd5; color:#bcd; background:#243040; }
/* A wrapping grid, not a horizontal strip. The strip meant cards were clipped
   at the node's right edge however wide the node was dragged, and stretching
   them to the node's full height left a card with its controls at the top and
   a hand-span of dead space underneath. auto-fill reflows the columns as the
   node is resized; align-content:start keeps each card its natural height. */
.h3c-grid { display:grid; gap:6px; padding:2px; flex:1 1 auto;
  grid-template-columns:repeat(auto-fill, minmax(148px, 1fr));
  align-content:start; overflow-y:auto; overflow-x:hidden;
  min-height:120px; scrollbar-width:thin; }
.h3c-card { display:flex; flex-direction:column; gap:3px; min-width:0;
  background:#282828; border:1px solid #3d3d3d; border-radius:6px; padding:5px; }
.h3c-card.off { opacity:.45; }
.h3c-tag { font-weight:600; color:#8fd3ff; font-variant-numeric:tabular-nums; }
/* contain, not cover: this is a reference the model is being handed, so seeing
   all of it matters more than a tidy rectangle. cover silently hid the top and
   bottom of every portrait image on the board. */
.h3c-thumb { width:100%; aspect-ratio:1/1; min-height:72px; object-fit:contain;
  border-radius:4px; background:#1b1b1b; }
.h3c-ph { width:100%; aspect-ratio:1/1; min-height:72px; border-radius:4px;
  background:#1b1b1b; display:flex; align-items:center; justify-content:center;
  color:#777; font-size:20px; }
.h3c-in { background:#1e1e1e; border:1px solid #444; color:#ddd; border-radius:3px;
  padding:2px 4px; font-size:11px; font-family:inherit; width:100%;
  box-sizing:border-box; }
.h3c-row { display:flex; gap:3px; }
.h3c-row .h3c-btn { flex:1 1 auto; padding:1px 0; text-align:center; }
.h3c-file { font-size:9px; color:#777; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
/* A reference you cannot hear is a reference you cannot judge. The card
   used to show a music note for audio and a triangle for video, so the only
   way to check what had been uploaded was to find the file on disk. */
.h3c-video { width:100%; aspect-ratio:1/1; min-height:72px; object-fit:contain;
  border-radius:4px; background:#111; }
.h3c-audio { width:100%; height:32px; margin:1px 0; }
.h3c-trim { display:flex; gap:3px; align-items:center; }
.h3c-trim .h3c-in { min-width:0; flex:1 1 0; }
.h3c-trim .h3c-btn { flex:0 0 auto; padding:1px 5px; }
.h3c-note { font-size:10px; opacity:.65; }
.h3c-warn { font-size:10px; color:#e0b060; }
`;

function ensureStyle() {
  if (document.getElementById("h3c-style")) return;
  const el = document.createElement("style");
  el.id = "h3c-style";
  el.textContent = CSS;
  document.head.appendChild(el);
}

function button(label, title, onClick) {
  const b = document.createElement("button");
  b.className = "h3c-btn";
  b.textContent = label;
  if (title) b.title = title;
  b.onclick = (e) => { e.stopPropagation(); onClick(e); };
  return b;
}

function viewUrl(file) {
  const q = new URLSearchParams({
    filename: file, subfolder: "h3_planner", type: "input",
  });
  return api.apiURL("/view?" + q.toString());
}

function kindOf(name) {
  const ext = (name.split(".").pop() || "").toLowerCase();
  if (["png", "jpg", "jpeg", "webp", "bmp", "gif"].includes(ext)) return "image";
  if (["mp4", "mov", "mkv", "webm", "avi", "m4v"].includes(ext)) return "video";
  if (["mp3", "wav", "flac", "m4a", "ogg", "aac"].includes(ext)) return "audio";
  return "other";
}

function roleForKind(kind) {
  if (kind === "video") return "video";
  if (kind === "audio") return "audio";
  return "character";
}

class CastBoard {
  constructor(node, widget) {
    this.node = node;
    this.widget = widget;
    this.showJson = false;
    this.doc = { entries: [] };

    this.root = document.createElement("div");
    this.root.className = "h3c-wrap";
    this.bar = document.createElement("div");
    this.bar.className = "h3c-bar";
    this.drop = document.createElement("div");
    this.drop.className = "h3c-drop";
    this.drop.textContent = "Drop images, video or audio here — or click to browse";
    this.grid = document.createElement("div");
    this.grid.className = "h3c-grid";
    this.foot = document.createElement("div");
    this.foot.className = "h3c-note";
    this.root.append(this.bar, this.drop, this.grid, this.foot);

    this.wireDropTarget();
    this.read();
    this.render();
  }

  /* ---- persistence ---- */

  read() {
    try {
      const parsed = JSON.parse(this.widget?.value || "{}");
      this.doc = parsed && typeof parsed === "object" ? parsed : {};
      if (!Array.isArray(this.doc.entries)) this.doc.entries = [];
      this.parseError = null;
    } catch (e) {
      this.parseError = e.message;
      this.doc = { entries: [] };
    }
  }

  write() {
    if (!this.widget) return;
    this.widget.value = JSON.stringify(this.doc, null, 1);
    this.node.graph?.setDirtyCanvas(true, true);
    if (this.widget.callback) {
      try { this.widget.callback(this.widget.value); } catch (e) {}
    }
  }

  /* ---- tags ---- */

  // Numbered per kind, in card order — reorder the board and the tags follow,
  // which is what keeps a prompt's <Subject 2> pointing at the same slot.
  tagsFor() {
    const counters = { Subject: 0, Picture: 0, Video: 0, Audio: 0 };
    return this.doc.entries.map((entry) => {
      if (entry.disabled) return "";
      const kind = ROLE_TAGS[entry.role] || "Subject";
      counters[kind] += 1;
      return `<${kind} ${counters[kind]}>`;
    });
  }

  imageSlotFor(index) {
    let slot = 0;
    for (let i = 0; i <= index; i++) {
      const e = this.doc.entries[i];
      if (e.disabled) continue;
      const kind = ROLE_TAGS[e.role] || "Subject";
      if (kind === "Subject" || kind === "Picture") slot += 1;
    }
    return slot;
  }

  /* ---- uploads ---- */

  wireDropTarget() {
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
    for (const ev of ["dragenter", "dragover"]) {
      this.drop.addEventListener(ev, (e) => {
        stop(e); this.drop.classList.add("hot");
      });
    }
    for (const ev of ["dragleave", "drop"]) {
      this.drop.addEventListener(ev, (e) => {
        stop(e); this.drop.classList.remove("hot");
      });
    }
    this.drop.addEventListener("drop", (e) => {
      const files = [...(e.dataTransfer?.files || [])];
      if (files.length) this.upload(files);
    });
    this.drop.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.accept = "image/*,video/*,audio/*";
      input.onchange = () => input.files?.length && this.upload([...input.files]);
      input.click();
    });
  }

  async upload(files) {
    this.drop.textContent = `Uploading ${files.length} file(s)…`;
    let added = 0, failed = [];
    for (const file of files) {
      const body = new FormData();
      body.append("file", file, file.name);
      try {
        const resp = await api.fetchApi("/h3planner/upload", { method: "POST", body });
        const data = await resp.json();
        if (data.error) { failed.push(`${file.name}: ${data.error}`); continue; }
        this.doc.entries.push({
          key: (data.file.split(".")[0] || "ref").slice(0, 24),
          role: roleForKind(data.kind),
          note: "",
          file: data.file,
        });
        added++;
      } catch (err) {
        failed.push(`${file.name}: ${err}`);
      }
    }
    this.drop.textContent = "Drop images, video or audio here — or click to browse";
    if (added) this.write();
    this.render();
    if (failed.length) alert("Some references did not upload:\n" + failed.join("\n"));
  }

  /* ---- editing ---- */

  move(i, delta) {
    const j = i + delta;
    if (j < 0 || j >= this.doc.entries.length) return;
    const [e] = this.doc.entries.splice(i, 1);
    this.doc.entries.splice(j, 0, e);
    this.write();
    this.render();
  }

  remove(i) {
    const e = this.doc.entries[i];
    if (!confirm(`Remove ${e.key || e.file} from the cast?\n\nThe uploaded file stays in input/h3_planner/.`))
      return;
    this.doc.entries.splice(i, 1);
    this.write();
    this.render();
  }

  /* ---- rendering ---- */

  render() {
    this.renderBar();
    this.renderGrid();
    this.renderFoot();
  }

  renderBar() {
    this.bar.replaceChildren();
    const counts = { Subject: 0, Picture: 0, Video: 0, Audio: 0 };
    for (const e of this.doc.entries) {
      if (!e.disabled) counts[ROLE_TAGS[e.role] || "Subject"]++;
    }
    const label = document.createElement("span");
    label.textContent = Object.entries(counts)
      .filter(([, v]) => v).map(([k, v]) => `${v} ${k}`).join(" · ") || "no references yet";
    this.bar.append(label);

    const sp = document.createElement("span");
    sp.className = "sp";
    this.bar.append(sp);

    this.bar.append(button("Copy tags", "Copy the tag list for pasting into a prompt", () => {
      const tags = this.tagsFor();
      const text = this.doc.entries.map((e, i) =>
        e.disabled ? null : `${tags[i]} = ${e.key} (${e.role})${e.note ? "; " + e.note : ""}`)
        .filter(Boolean).join("\n");
      navigator.clipboard?.writeText(text);
    }));

    const j = button("JSON", "Show the raw cast_json widget", () => {
      this.showJson = !this.showJson;
      this.applyJsonVisibility();
      this.render();
    });
    if (this.showJson) j.classList.add("on");
    this.bar.append(j);
  }

  renderGrid() {
    this.grid.replaceChildren();
    if (this.parseError) {
      const err = document.createElement("div");
      err.className = "h3c-warn";
      err.textContent = "cast_json is not valid JSON: " + this.parseError;
      this.grid.append(err);
      return;
    }

    const tags = this.tagsFor();
    this.doc.entries.forEach((entry, i) => {
      const card = document.createElement("div");
      card.className = "h3c-card" + (entry.disabled ? " off" : "");
      const kind = ROLE_TAGS[entry.role] || "Subject";

      const tag = document.createElement("div");
      tag.className = "h3c-tag";
      tag.textContent = tags[i] || "(disabled)";
      card.append(tag);

      const assetKind = kindOf(entry.file || "");
      let player = null;
      if (assetKind === "image" && entry.file) {
        const img = document.createElement("img");
        img.className = "h3c-thumb";
        img.src = viewUrl(entry.file);
        img.loading = "lazy";
        card.append(img);
      } else if (assetKind === "video" && entry.file) {
        player = document.createElement("video");
        player.className = "h3c-video";
        player.src = viewUrl(entry.file);
        player.controls = true;
        player.preload = "metadata";
        card.append(player);
      } else if (assetKind === "audio" && entry.file) {
        const ph = document.createElement("div");
        ph.className = "h3c-ph";
        ph.textContent = "♪";
        card.append(ph);
        player = document.createElement("audio");
        player.className = "h3c-audio";
        player.src = viewUrl(entry.file);
        player.controls = true;
        player.preload = "metadata";
        card.append(player);
      } else {
        const ph = document.createElement("div");
        ph.className = "h3c-ph";
        ph.textContent = "?";
        card.append(ph);
      }

      const key = document.createElement("input");
      key.className = "h3c-in";
      key.value = entry.key || "";
      key.placeholder = "key";
      key.onchange = () => { entry.key = key.value.trim(); this.write(); };
      card.append(key);

      const role = document.createElement("select");
      role.className = "h3c-in";
      const allowed = assetKind === "video" ? ["video"]
        : assetKind === "audio" ? ["audio"] : IMAGE_ROLES;
      for (const r of allowed) {
        const o = document.createElement("option");
        o.value = o.textContent = r;
        if (entry.role === r) o.selected = true;
        role.append(o);
      }
      if (!allowed.includes(entry.role)) {
        entry.role = allowed[0];
        role.value = allowed[0];
      }
      role.title = "Role decides the tag kind. Identity that recurs is a Subject; " +
        "a concrete first/last/key frame is a Picture.";
      role.onchange = () => { entry.role = role.value; this.write(); this.render(); };
      card.append(role);

      const note = document.createElement("input");
      note.className = "h3c-in";
      note.value = entry.note || "";
      note.placeholder = "note (optional)";
      note.onchange = () => { entry.note = note.value; this.write(); };
      card.append(note);

      const file = document.createElement("div");
      file.className = "h3c-file";
      file.textContent = entry.file || "(no file)";
      file.title = entry.file || "";
      card.append(file);

      if (player) card.append(this.trimRow(entry, player));

      const row = document.createElement("div");
      row.className = "h3c-row";
      row.append(
        button("◀", "Move earlier — renumbers the tags", () => this.move(i, -1)),
        button("▶", "Move later — renumbers the tags", () => this.move(i, 1)),
        button(entry.disabled ? "On" : "Off", "Exclude without deleting", () => {
          entry.disabled = !entry.disabled; this.write(); this.render();
        }),
        button("✕", "Remove from the cast", () => this.remove(i)),
      );
      card.append(row);

      if (kind === "Subject" || kind === "Picture") {
        const slot = this.imageSlotFor(i);
        const s = document.createElement("div");
        s.className = slot > MAX_IMAGE_SLOTS ? "h3c-warn" : "h3c-file";
        s.textContent = slot > MAX_IMAGE_SLOTS
          ? `past image_${MAX_IMAGE_SLOTS} — not wired out`
          : `→ image_${slot}`;
        card.append(s);
      }

      this.grid.append(card);
    });
  }

  /* in / out points for one audio or video card.

     Stored on the entry as plain seconds, so the node trims with ffmpeg at
     load time rather than the whole file reaching the sampler. Blank or zero
     means "from the start" and "to the end". */
  trimRow(entry, player) {
    const wrap = document.createElement("div");
    wrap.className = "h3c-trim";

    const field = (key, placeholder, title) => {
      const el = document.createElement("input");
      el.className = "h3c-in";
      el.type = "number";
      el.step = "0.01";
      el.min = "0";
      el.placeholder = placeholder;
      el.title = title;
      el.value = entry[key] === undefined || entry[key] === null ? "" : entry[key];
      el.onchange = () => {
        const value = parseFloat(el.value);
        if (!isFinite(value) || value <= 0) delete entry[key];
        else entry[key] = Math.round(value * 1000) / 1000;
        this.write();
        this.render();
      };
      return el;
    };

    const stamp = (key) => {
      const at = player.currentTime;
      if (!isFinite(at) || at <= 0) delete entry[key];
      else entry[key] = Math.round(at * 1000) / 1000;
      this.write();
      this.render();
    };

    wrap.append(
      field("start", "in s", "Start here. Blank or 0 means the start of the file."),
      field("end", "out s", "End here. Blank means the end of the file."),
      button("⇤", "Set the in-point from where the player is now",
             () => stamp("start")),
      button("⇥", "Set the out-point from where the player is now",
             () => stamp("end")),
      button("↺", "Use the whole file", () => {
        delete entry.start;
        delete entry.end;
        this.write();
        this.render();
      }),
    );

    const span = document.createElement("div");
    span.className = "h3c-note";
    const from = entry.start || 0;
    if (entry.start || entry.end) {
      span.textContent = entry.end
        ? `using ${from.toFixed(2)}s to ${(+entry.end).toFixed(2)}s ` +
          `(${(entry.end - from).toFixed(2)}s)`
        : `using ${from.toFixed(2)}s to the end`;
    } else {
      span.textContent = "whole file";
    }

    const holder = document.createElement("div");
    holder.append(wrap, span);
    return holder;
  }

  renderFoot() {
    const images = this.doc.entries.filter(
      (e) => !e.disabled && ["Subject", "Picture"].includes(ROLE_TAGS[e.role] || "Subject")).length;
    const bits = [`${images} image slot(s) wired out`];
    if (images > MAX_IMAGE_SLOTS) {
      bits.push(`only the first ${MAX_IMAGE_SLOTS} reach the sampler`);
    }
    bits.push("Subject = recurring identity · Picture = a concrete frame anchor");
    this.foot.textContent = bits.join("  ·  ");
  }

  applyJsonVisibility() {
    const el = this.widget?.element || this.widget?.inputEl || null;
    if (el) el.style.display = this.showJson ? "" : "none";
    if (this.widget) {
      this.widget.computeSize = this.showJson ? undefined : () => [0, -4];
    }
    this.node.setDirtyCanvas(true, true);
  }
}

app.registerExtension({
  name: "h3planner.castboard",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "H3PlannerCastBoard") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      ensureStyle();
      try {
        const widget = this.widgets?.find((w) => w.name === "cast_json");
        const board = new CastBoard(this, widget);
        this.h3cBoard = board;
        const dom = this.addDOMWidget("h3c_board", "div", board.root,
                                      { serialize: false, hideOnZoom: false });
        const MIN_BODY = 260;
        const node = this;
        dom.computeSize = function (width) {
          let used = 0;
          for (const w of node.widgets ?? []) {
            if (w === dom) continue;
            const h = w.computeSize ? w.computeSize(width)[1]
                                    : (LiteGraph.NODE_WIDGET_HEIGHT ?? 20);
            used += Math.max(0, h) + 4;
          }
          const slots = Math.max(node.inputs?.length ?? 0,
                                 node.outputs?.length ?? 0);
          const chrome = (LiteGraph.NODE_SLOT_HEIGHT ?? 20) * slots + 30;
          return [width, Math.max(MIN_BODY, node.size[1] - used - chrome)];
        };
        const minH = MIN_BODY + 170;
        this.size = [Math.max(this.size[0], 560), Math.max(this.size[1], minH)];
        const onResize = this.onResize;
        this.onResize = function (size) {
          // 320 rather than 420: the grid wraps now, so one column is a
          // legitimate shape for a narrow node rather than a broken one.
          size[0] = Math.max(size[0], 320);
          size[1] = Math.max(size[1], minH);
          onResize?.call(this, size);
        };
        board.applyJsonVisibility();
      } catch (e) {
        console.error("[H3Planner] cast board failed, JSON widget still works", e);
      }
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        try { this.h3cBoard?.read(); this.h3cBoard?.render(); } catch (e) {}
      }, 60);
      return r;
    };
  },
});
